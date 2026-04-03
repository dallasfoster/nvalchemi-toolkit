# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Domain-parallel dynamics wrapper for spatial decomposition."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from nvalchemi.distributed.config import DomainConfig, HookScope, _GeometrySnapshot
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.hooks._context import HookContext

if TYPE_CHECKING:
    from nvalchemi.data.batch import Batch
    from nvalchemi.distributed.atom_migrator import AtomMigrator
    from nvalchemi.distributed.ghost_exchanger import GhostExchanger
    from nvalchemi.distributed.partitioner import SpatialPartitioner

logger = logging.getLogger(__name__)


class DomainParallel(BaseDynamics):
    """Wraps any ``BaseDynamics`` subclass with spatial domain decomposition.

    ``DomainParallel`` splits the global simulation box across ranks using
    a ``SpatialPartitioner``, manages ghost exchange and atom migration
    each step, and provides gather utilities to reconstruct the full
    system on a single rank when needed.

    The inner dynamics object retains its own hooks (e.g. ``NeighborListHook``).
    Hooks registered on the ``DomainParallel`` wrapper itself are *outer* hooks
    that fire around the entire domain-decomposed step.

    Parameters
    ----------
    dynamics : BaseDynamics
        The underlying single-GPU dynamics integrator or optimizer.
    config : DomainConfig
        Domain decomposition configuration.
    **kwargs : Any
        Additional keyword arguments forwarded to ``BaseDynamics.__init__``
        (e.g. ``hooks``, ``n_steps``, ``device_type``).
    """

    def __init__(
        self,
        dynamics: BaseDynamics,
        config: DomainConfig,
        **kwargs: Any,
    ) -> None:
        super().__init__(model=dynamics.model, **kwargs)
        self._dynamics: BaseDynamics = dynamics
        self._config: DomainConfig = config

        # Lazy-initialized in partition()
        self._partitioner: SpatialPartitioner | None = None
        self._ghost_exchanger: GhostExchanger | None = None
        self._migrator: AtomMigrator | None = None

        # Runtime state
        self._n_owned: int = 0
        self._geometry_snapshot: _GeometrySnapshot | None = None
        self._forces_primed: bool = False

        # AABB / NL persistence state.
        # Caching pos_min between steps keeps the local coordinate frame
        # stable so the NeighborListHook's skin check computes correct
        # displacements.  The NL ref is only invalidated when the padded
        # batch size changes (ghost count changed).
        self._cached_pos_min: torch.Tensor | None = None
        self._prev_n_padded: int = 0

        # Number of periodic image atoms added by _add_periodic_images.
        self._n_periodic_images: int = 0

        # Optional debug callback: called with (padded_batch, n_owned, step)
        # after compute but before restore_geometry/strip.  Set from outside
        # to capture padded-batch state including ghost atoms and NL.
        self._debug_post_compute_fn: Any = None

        # Determine rank from mesh, or fall back to dist / 0.
        if config.mesh is not None:
            try:
                self._domain_rank: int = config.mesh.get_local_rank()
            except Exception:
                self._domain_rank = 0
        elif dist.is_initialized():
            self._domain_rank = dist.get_rank()
        else:
            self._domain_rank = 0

    # ------------------------------------------------------------------
    # Properties delegated to inner dynamics
    # ------------------------------------------------------------------

    @property
    def __needs_keys__(self) -> set[str]:  # type: ignore[override]
        """Delegate to the inner dynamics."""
        return self._dynamics.__needs_keys__

    @property
    def __provides_keys__(self) -> set[str]:  # type: ignore[override]
        """Delegate to the inner dynamics."""
        return self._dynamics.__provides_keys__

    # ------------------------------------------------------------------
    # Partition
    # ------------------------------------------------------------------

    def partition(self, batch: Batch | None) -> Batch:
        """Partition the global batch across ranks.

        Must be called once before ``run()`` or the step loop.

        Parameters
        ----------
        batch : Batch | None
            The full-system batch on rank 0.  ``None`` on all other ranks.

        Returns
        -------
        Batch
            The local batch containing only this rank's owned atoms.
        """
        from nvalchemi.distributed.atom_migrator import AtomMigrator
        from nvalchemi.distributed.ghost_exchanger import GhostExchanger
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        # Resolve device: prefer the batch's device, fall back to the
        # current CUDA device (NCCL requires all tensors on GPU).
        if batch is not None:
            device = batch.positions.device
        elif torch.cuda.is_available():
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        else:
            device = torch.device("cpu")

        # --- Broadcast cell matrix and PBC from rank 0 ---
        # Shapes follow AtomicData/Batch convention:
        #   cell: (1, 3, 3)   pbc: (1, 3)
        # All tensors must have matching shape, dtype, and device across
        # ranks for NCCL broadcast.
        if batch is not None:
            cell_bcast = batch.cell.clone().to(
                device=device, dtype=torch.float32
            )  # (1, 3, 3)
            pbc_bcast = (
                batch.pbc.clone().to(device=device)
                if hasattr(batch, "pbc") and batch.pbc is not None
                else torch.ones(1, 3, dtype=torch.bool, device=device)
            )
        else:
            cell_bcast = torch.zeros(1, 3, 3, dtype=torch.float32, device=device)
            pbc_bcast = torch.ones(1, 3, dtype=torch.bool, device=device)

        if dist.is_initialized():
            dist.broadcast(cell_bcast, src=0)
            dist.broadcast(pbc_bcast, src=0)

        # Pass batch-convention shapes directly — SpatialPartitioner
        # normalizes (1, 3, 3) → (3, 3) and (1, 3) → (3,) internally.
        cell_matrix = cell_bcast
        pbc = pbc_bcast

        # --- Initialize components ---
        self._partitioner = SpatialPartitioner(
            config=self._config,
            cell_matrix=cell_matrix,
            pbc=pbc,
        )

        mesh = self._config.mesh
        if mesh is not None:
            self._ghost_exchanger = GhostExchanger(
                partitioner=self._partitioner,
                config=self._config,
                mesh=mesh,
            )
            self._migrator = AtomMigrator(
                partitioner=self._partitioner,
                config=self._config,
                mesh=mesh,
            )

        # --- Distribute atoms to ranks ---
        if not dist.is_initialized():
            # Single-process mode: rank 0 gets everything.
            if batch is None:
                raise ValueError("batch must be provided in single-process mode")
            self._n_owned = batch.positions.shape[0]
            return batch

        # Multi-process: use scatter_v to distribute atoms from rank 0.
        from physicsnemo.distributed.utils import scatter_v_wrapper

        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        group = self._config.mesh.get_group() if self._config.mesh is not None else None
        world_size = dist.get_world_size(group=group)

        # Rank 0: assign atoms to ranks, sort by rank, compute per-rank sizes.
        if self._domain_rank == 0:
            if batch is None:
                raise ValueError("batch must be provided on rank 0")

            n_atoms = batch.positions.shape[0]
            all_positions = batch.positions.clone()
            all_velocities = (
                batch.velocities.clone()
                if hasattr(batch, "velocities") and batch.velocities is not None
                else torch.zeros(n_atoms, 3, device=device)
            )
            all_atomic_numbers = batch.atomic_numbers.clone().to(torch.int64)
            all_atomic_masses = (
                batch.atomic_masses.clone()
                if hasattr(batch, "atomic_masses") and batch.atomic_masses is not None
                else torch.ones(n_atoms, device=device)
            )

            rank_assignment = self._partitioner.assign_atoms_to_ranks(all_positions)
            rank_assignment = rank_assignment.to(torch.int64)
            sorted_idx = torch.argsort(rank_assignment)
            per_rank_counts = torch.bincount(rank_assignment, minlength=world_size)
            per_rank_sizes = per_rank_counts.tolist()

            # Sort all fields by rank assignment.
            sorted_positions = all_positions[sorted_idx]
            sorted_velocities = all_velocities[sorted_idx]
            sorted_atomic_numbers = all_atomic_numbers[sorted_idx]
            sorted_atomic_masses = all_atomic_masses[sorted_idx]
            has_velocities = (
                hasattr(batch, "velocities")
                and batch.velocities is not None
                and batch.velocities.any()
            )
        else:
            # Dummy tensors for non-root ranks (scatter_v_wrapper requires a tensor).
            sorted_positions = torch.zeros(0, 3, device=device)
            sorted_velocities = torch.zeros(0, 3, device=device)
            sorted_atomic_numbers = torch.zeros(0, dtype=torch.int64, device=device)
            sorted_atomic_masses = torch.zeros(0, device=device)
            per_rank_sizes = [0] * world_size
            has_velocities = False

        # Broadcast per_rank_sizes and has_velocities so all ranks know them.
        sizes_tensor = torch.tensor(per_rank_sizes, dtype=torch.int64, device=device)
        dist.broadcast(sizes_tensor, src=0, group=group)
        per_rank_sizes = sizes_tensor.tolist()

        has_vel_tensor = torch.tensor(
            [int(has_velocities)], dtype=torch.int32, device=device
        )
        dist.broadcast(has_vel_tensor, src=0, group=group)
        has_velocities = has_vel_tensor.item() > 0

        # Scatter per-atom data from rank 0 to all ranks.
        local_positions = scatter_v_wrapper(
            tensor=sorted_positions,
            sizes=per_rank_sizes,
            dim=0,
            src=0,
            group=group,
        )
        local_velocities = scatter_v_wrapper(
            tensor=sorted_velocities,
            sizes=per_rank_sizes,
            dim=0,
            src=0,
            group=group,
        )
        # For 1-D tensors, unsqueeze before scatter_v then squeeze after.
        local_atomic_numbers = scatter_v_wrapper(
            tensor=sorted_atomic_numbers.unsqueeze(-1)
            if sorted_atomic_numbers.ndim == 1
            else sorted_atomic_numbers,
            sizes=per_rank_sizes,
            dim=0,
            src=0,
            group=group,
        ).squeeze(-1)
        local_atomic_masses = scatter_v_wrapper(
            tensor=sorted_atomic_masses.unsqueeze(-1)
            if sorted_atomic_masses.ndim == 1
            else sorted_atomic_masses,
            sizes=per_rank_sizes,
            dim=0,
            src=0,
            group=group,
        ).squeeze(-1)

        # Build local AtomicData -> Batch for this rank's atoms.
        local_data = AtomicData(
            positions=local_positions,
            atomic_numbers=local_atomic_numbers,
            atomic_masses=local_atomic_masses,
            cell=cell_matrix if cell_matrix.ndim == 3 else cell_matrix.unsqueeze(0),
            pbc=pbc if pbc.ndim == 2 else pbc.unsqueeze(0),
        )
        if has_velocities:
            local_data.add_node_property("velocities", local_velocities)

        local_batch = BatchCls.from_data_list([local_data], device=device)
        self._n_owned = local_positions.shape[0]
        return local_batch

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Execute one domain-decomposed dynamics step.

        The step flow is:

        1. Fire outer ``BEFORE_STEP`` hooks (scope-aware).
        2. Ghost exchange — pad the batch with halo atoms.
        3. Prepare the padded batch (local bounding box, ``pbc=False``).
        4. Delegate to the inner dynamics ``step()``.
        5. Unprepare — restore global coordinates and cell.
        6. Strip ghost atoms.
        7. Migrate atoms that crossed domain boundaries (if needed).
        8. Fire outer ``AFTER_STEP`` hooks (scope-aware).

        Parameters
        ----------
        batch : Batch
            The local batch (owned atoms only).

        Returns
        -------
        tuple[Batch, torch.Tensor | None]
            ``(updated_batch, converged)`` matching the ``BaseDynamics``
            return signature.
        """
        # Auto-prime forces on first step if not already done (handles
        # the case where user calls step() in a loop instead of run()).
        if not self._forces_primed:
            batch = self._prime_forces(batch)
            self._forces_primed = True

        logger.info(
            "[rank %d] step %d: BEFORE_STEP hooks", self._domain_rank, self.step_count
        )
        # 1. Outer BEFORE_STEP hooks
        self._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        # 2. Pre-update on the OWNED batch (velocity Verlet half-kick).
        #    This must happen BEFORE ghost exchange so that ghost atoms
        #    receive post-half-kick positions.  Without this, ghost atoms
        #    lag behind owned atoms by ~0.03 Å every step, creating a
        #    systematic force asymmetry that causes energy drift.
        dyn = self._dynamics
        dyn._ensure_state_initialized(batch)
        dyn._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, batch)
        dyn.pre_update(batch)
        dyn._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, batch)

        # Wrap positions before ghost exchange so ghost atoms receive
        # wrapped, post-half-kick positions.
        self._wrap_positions(batch)

        logger.info(
            "[rank %d] step %d: ghost exchange", self._domain_rank, self.step_count
        )
        # 3. Ghost exchange — now with post-half-kick positions
        if self._ghost_exchanger is not None:
            padded_batch, n_owned = self._ghost_exchanger.exchange(batch)
        else:
            padded_batch = batch
            n_owned = batch.positions.shape[0]
        self._n_owned = n_owned

        logger.info(
            "[rank %d] step %d: prepare + inner step",
            self._domain_rank,
            self.step_count,
        )
        # 4-5. AABB + compute + post_update on padded batch
        snapshot = self._save_geometry(padded_batch)
        self._ensure_output_tensors(padded_batch)
        padded_batch, converged = self._run_inner_step(padded_batch)

        # Debug hook: capture padded batch state (with NL, in AABB frame)
        # before geometry is restored and ghosts are stripped.
        if self._debug_post_compute_fn is not None:
            self._debug_post_compute_fn(padded_batch, n_owned, self.step_count)

        self._restore_geometry(padded_batch, snapshot)

        # 6. Strip ghost atoms
        if self._ghost_exchanger is not None:
            batch = self._ghost_exchanger.strip(padded_batch, n_owned)
        else:
            batch = padded_batch

        # --- Diagnostic: after strip, compare owned energy/forces ---
        if self.step_count < 50 and self.step_count % 5 == 0:
            _e = batch.energies
            _f = batch.forces
            _p = batch.positions
            _v = (
                batch.velocities
                if hasattr(batch, "velocities") and batch.velocities is not None
                else None
            )
            # Kinetic energy: 0.5 * m * v^2 (sum over all atoms)
            if (
                _v is not None
                and hasattr(batch, "atomic_masses")
                and batch.atomic_masses is not None
            ):
                ke = (0.5 * batch.atomic_masses.unsqueeze(-1) * _v**2).sum().item()
            else:
                ke = 0
            logger.info(
                "[rank %d] step %d DIAG after_strip: n_owned=%d E_pot=%.4f KE=%.4f "
                "E_total=%.4f fmax=%.6f pos_range=[%.2f,%.2f]",
                self._domain_rank,
                self.step_count,
                _p.shape[0],
                _e.sum().item() if _e is not None else 0,
                ke,
                (_e.sum().item() if _e is not None else 0) + ke,
                _f.norm(dim=-1).max().item()
                if _f is not None and _f.numel() > 0
                else 0,
                _p.min().item(),
                _p.max().item(),
            )

        # NOTE: Position wrapping now happens BEFORE ghost exchange
        # (in step 2 above), not here.  This ensures ghost atoms
        # receive wrapped, post-half-kick positions.

        logger.info(
            "[rank %d] step %d: migration check", self._domain_rank, self.step_count
        )
        # 7. Migration (if needed).
        # Migration uses collectives (all_to_all_single, batch_isend_irecv),
        # so ALL ranks must participate.  Synchronize the decision.
        if self._migrator is not None:
            needs = self._migrator.needs_migration(batch)
            if dist.is_initialized():
                flag = torch.tensor(
                    [int(needs)], dtype=torch.int32, device=batch.positions.device
                )
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                needs = flag.item() > 0
            if needs:
                logger.info(
                    "[rank %d] step %d: migrating atoms",
                    self._domain_rank,
                    self.step_count,
                )
                batch = self._migrator.migrate(batch)

        logger.info(
            "[rank %d] step %d: AFTER_STEP hooks", self._domain_rank, self.step_count
        )
        # 8. Outer AFTER_STEP hooks
        self._call_hooks(DynamicsStage.AFTER_STEP, batch)

        self.step_count += 1
        logger.info("[rank %d] step %d: complete", self._domain_rank, self.step_count)

        return batch, converged

    # ------------------------------------------------------------------
    # Prepare / unprepare padded batch
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Inner step orchestration
    # ------------------------------------------------------------------

    def _run_inner_step(self, padded_batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        """Execute compute + post_update on the padded batch.

        ``pre_update`` has already been run on the OWNED batch before
        ghost exchange (in ``step()``), so ghost atoms already have
        post-half-kick positions.  This method handles:

        1. AABB cell computation and position shift
        2. Neighbor list build + model forward (compute)
        3. Velocity Verlet finalize (post_update)
        """
        dyn = self._dynamics

        dyn._call_hooks(DynamicsStage.BEFORE_STEP, padded_batch)

        # --- AABB geometry: compute cell from current positions ---
        n_padded = padded_batch.num_nodes
        size_changed = n_padded != self._prev_n_padded
        self._apply_aabb(padded_batch, force_recompute=size_changed)
        if size_changed:
            self._invalidate_nl_ref()
            self._prev_n_padded = n_padded

        # --- Diagnostic: state AFTER AABB, BEFORE compute ---
        step = self.step_count
        if step < 50 and step % 5 == 0:
            _p = padded_batch.positions
            # Check how many neighbors the NL hook will find
            nm = getattr(padded_batch, "neighbor_matrix", None)
            nn = getattr(padded_batch, "num_neighbors", None)
            logger.info(
                "[rank %d] step %d DIAG pre_compute: pos_range=[%.2f,%.2f] "
                "has_NL=%s cell_diag=[%.2f,%.2f,%.2f]",
                self._domain_rank,
                step,
                _p.min().item(),
                _p.max().item(),
                nm is not None,
                padded_batch.cell[0, 0, 0].item(),
                padded_batch.cell[0, 1, 1].item(),
                padded_batch.cell[0, 2, 2].item(),
            )

        # --- compute (neighbor list build + model forward) ---
        dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, padded_batch)
        dyn.compute(padded_batch)
        dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, padded_batch)

        # --- Diagnostic: state AFTER compute, BEFORE post_update ---
        if step < 50 and step % 5 == 0:
            n_owned = self._n_owned
            _e = padded_batch.energies
            _f = padded_batch.forces
            nn = getattr(padded_batch, "num_neighbors", None)
            logger.info(
                "[rank %d] step %d DIAG post_compute: E_padded=%.4f "
                "fmax_owned=%.6f fmean_owned=%.6f "
                "avg_neighbors=%.1f min_neighbors=%d",
                self._domain_rank,
                step,
                _e.sum().item() if _e is not None else 0,
                _f[:n_owned].norm(dim=-1).max().item()
                if _f is not None and n_owned > 0
                else 0,
                _f[:n_owned].norm(dim=-1).mean().item()
                if _f is not None and n_owned > 0
                else 0,
                nn.float().mean().item() if nn is not None else 0,
                nn.min().item() if nn is not None else 0,
            )

        # --- post_update (velocity Verlet finalize) ---
        dyn._call_hooks(DynamicsStage.BEFORE_POST_UPDATE, padded_batch)
        dyn.post_update(padded_batch)
        dyn._call_hooks(DynamicsStage.AFTER_POST_UPDATE, padded_batch)

        dyn._call_hooks(DynamicsStage.AFTER_STEP, padded_batch)

        # Convergence check (typically a no-op for NVE).
        converged = dyn._check_convergence(padded_batch)
        dyn._last_converged = converged
        if converged is not None:
            dyn._call_hooks(DynamicsStage.ON_CONVERGE, padded_batch)

        dyn.step_count += 1
        return padded_batch, converged

    def _apply_aabb(self, batch: Batch, force_recompute: bool = False) -> None:
        """Set the cell and shift positions for the NL builder.

        For **decomposed** dimensions (Z): AABB cell computed from
        positions + skin padding, positions shifted to local origin.
        For **non-decomposed** dimensions (X, Y): original periodic cell
        preserved, no position shift (the NL builder uses PBC shift
        vectors for cross-boundary neighbors).
        """
        positions = batch.positions
        skin = self._config.skin
        decomposed = self._decomposed_dims_mask(positions.device)

        # Minimum AABB padding: must be large enough that no atom sits at
        # the very edge of the cell, which would cause the cell-list builder
        # to wrap it into cell 0 even with pbc=False (producing spurious
        # shift vectors).  1 Å is safe for typical lattice spacings.
        _MIN_PADDING = 1.0
        padding = max(skin, _MIN_PADDING)

        if self._cached_pos_min is not None and not force_recompute:
            pos_min = self._cached_pos_min
        else:
            pos_min = positions.min(dim=0).values - padding
            # Only shift in decomposed dims; non-decomposed keep pos_min=0.
            pos_min = torch.where(decomposed, pos_min, torch.zeros_like(pos_min))
            self._cached_pos_min = pos_min

        pos_max = positions.max(dim=0).values
        aabb_lengths = (pos_max - pos_min + padding).clamp(min=1e-6)

        # Build cell: AABB for decomposed dims, original cell for periodic dims.
        original_cell = (
            self._geometry_snapshot.original_cell
            if self._geometry_snapshot is not None
            else batch.cell.clone()
        )
        local_cell = (
            torch.diag(aabb_lengths)
            .unsqueeze(0)
            .to(dtype=batch.cell.dtype, device=batch.cell.device)
        )
        for d in range(3):
            if not decomposed[d]:
                local_cell[0, d, d] = original_cell[0, d, d]

        if batch.cell.shape[0] > 1:
            local_cell = local_cell.expand(batch.cell.shape[0], -1, -1)
        batch.cell = local_cell

        # Shift positions only in decomposed dims.
        batch.positions = positions - pos_min

        if self._geometry_snapshot is not None:
            self._geometry_snapshot.pos_min = pos_min

    def _add_periodic_images(self, batch: Batch) -> Batch:
        """Add copies of atoms near cell edges for non-decomposed periodic dims.

        The nvalchemiops NL builder does not correctly handle mixed PBC
        (e.g. ``[True, True, False]``).  Instead, we run with
        ``pbc=False`` everywhere and explicitly add periodic image copies
        of atoms within ``cutoff + skin`` of each periodic cell face.

        Returns a **new** ``Batch`` (the ``SegmentedLevelStorage`` used
        by ``Batch`` validates that all tensors in a group share the same
        length, so we cannot extend in-place).

        Only non-decomposed periodic dimensions get images (decomposed
        dims are handled by ghost exchange).
        """
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        original_pbc = (
            self._geometry_snapshot.original_pbc
            if self._geometry_snapshot is not None
            else getattr(batch, "pbc", None)
        )
        if original_pbc is None or not original_pbc.any():
            return batch

        original_cell = (
            self._geometry_snapshot.original_cell
            if self._geometry_snapshot is not None
            else batch.cell
        )
        decomposed = self._decomposed_dims_mask(batch.positions.device)
        pbc_flags = original_pbc.squeeze(0)  # (3,) bool
        box_diag = torch.diag(original_cell.squeeze(0))  # (3,)

        ghost_width = self._config.cutoff + self._config.skin
        positions = batch.positions  # global coords (before AABB shift)

        image_parts: list[torch.Tensor] = []

        for d in range(3):
            if not pbc_flags[d] or decomposed[d]:
                continue
            L = box_diag[d].item()
            pos_d = positions[:, d]

            lo_mask = pos_d < ghost_width
            if lo_mask.any():
                shifted = positions[lo_mask].clone()
                shifted[:, d] += L
                image_parts.append(shifted)

            hi_mask = pos_d > (L - ghost_width)
            if hi_mask.any():
                shifted = positions[hi_mask].clone()
                shifted[:, d] -= L
                image_parts.append(shifted)

        if not image_parts:
            return batch

        all_images = torch.cat(image_parts, dim=0)
        n_images = all_images.shape[0]
        device = positions.device
        dtype = positions.dtype

        # Build a new Batch with real + image atoms (Batch validates
        # tensor lengths, so we can't mutate in-place).
        all_pos = torch.cat([positions, all_images], dim=0)
        all_Z = torch.cat(
            [batch.atomic_numbers, batch.atomic_numbers[:1].expand(n_images).clone()],
            dim=0,
        )

        data = AtomicData(positions=all_pos, atomic_numbers=all_Z)

        if hasattr(batch, "atomic_masses") and batch.atomic_masses is not None:
            data.atomic_masses = torch.cat(
                [batch.atomic_masses, batch.atomic_masses[:1].expand(n_images).clone()],
                dim=0,
            )
        if hasattr(batch, "velocities") and batch.velocities is not None:
            data.add_node_property(
                "velocities",
                torch.cat(
                    [
                        batch.velocities,
                        torch.zeros(n_images, 3, dtype=dtype, device=device),
                    ],
                    dim=0,
                ),
            )
        if hasattr(batch, "forces") and batch.forces is not None:
            data.add_node_property(
                "forces",
                torch.cat(
                    [
                        batch.forces,
                        torch.zeros(n_images, 3, dtype=dtype, device=device),
                    ],
                    dim=0,
                ),
            )
        if hasattr(batch, "cell") and batch.cell is not None:
            data.cell = batch.cell.clone()
        if hasattr(batch, "pbc") and batch.pbc is not None:
            data.pbc = batch.pbc.clone()

        new_batch = BatchCls.from_data_list([data], device=device)
        if hasattr(batch, "energies") and batch.energies is not None:
            new_batch.energies = batch.energies.clone()

        return new_batch

    @staticmethod
    def _strip_periodic_images(padded_batch: Batch, n_real: int) -> Batch:
        """Remove periodic image atoms, rebuilding a clean ``Batch``.

        Returns a fresh ``Batch`` with only the first *n_real* atoms.
        """
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        if padded_batch.num_nodes <= n_real:
            return padded_batch

        device = padded_batch.positions.device
        data = AtomicData(
            positions=padded_batch.positions[:n_real],
            atomic_numbers=padded_batch.atomic_numbers[:n_real],
        )
        if (
            hasattr(padded_batch, "atomic_masses")
            and padded_batch.atomic_masses is not None
        ):
            data.atomic_masses = padded_batch.atomic_masses[:n_real]
        if hasattr(padded_batch, "velocities") and padded_batch.velocities is not None:
            data.add_node_property("velocities", padded_batch.velocities[:n_real])
        if hasattr(padded_batch, "forces") and padded_batch.forces is not None:
            data.add_node_property("forces", padded_batch.forces[:n_real])
        if hasattr(padded_batch, "cell") and padded_batch.cell is not None:
            data.cell = padded_batch.cell.clone()
        if hasattr(padded_batch, "pbc") and padded_batch.pbc is not None:
            data.pbc = padded_batch.pbc.clone()

        new_batch = BatchCls.from_data_list([data], device=device)
        if hasattr(padded_batch, "energies") and padded_batch.energies is not None:
            new_batch.energies = padded_batch.energies.clone()

        return new_batch

    def _invalidate_nl_ref(self) -> None:
        """Reset the NeighborListHook's reference positions.

        Each step creates a fresh ``Batch`` in a different local
        coordinate frame, so the skin-check displacement from a
        previous step's reference is meaningless.  Resetting forces
        a full neighbor list rebuild.
        """
        from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook

        for hook in self._dynamics.hooks:
            if isinstance(hook, NeighborListHook):
                hook._ref_positions = None
                break

    @staticmethod
    def _wrap_positions(batch: Batch) -> None:
        """Wrap atom positions back into the periodic box.

        After the inner step (which runs with ``pbc=False``), atoms may
        have drifted outside the global cell.  This wraps them back using
        fractional-coordinate modulo, which is correct for both
        orthorhombic and triclinic cells.

        Must be called BEFORE migration so that ``assign_atoms_to_ranks``
        sees wrapped positions and moves atoms to the correct domain.
        The subsequent ghost exchange (next step) then provides correct
        neighbors.
        """
        cell = batch.cell  # (1, 3, 3)
        pbc = batch.pbc  # (1, 3) bool
        if cell is None or pbc is None or not pbc.any():
            return

        cell_3x3 = cell.squeeze(0)  # (3, 3)
        inv_cell = torch.linalg.inv(cell_3x3)

        # Convert to fractional, wrap periodic dims, convert back.
        frac = batch.positions @ inv_cell.T  # (N, 3)
        pbc_mask = pbc.squeeze(0)  # (3,) bool
        frac[:, pbc_mask] = frac[:, pbc_mask] % 1.0
        batch.positions = frac @ cell_3x3

    def _decomposed_dims_mask(self, device: torch.device) -> torch.Tensor:
        """Return a ``(3,)`` bool mask: True for dimensions that are decomposed.

        A dimension is decomposed when the rank grid has > 1 rank along it.
        Decomposed dims use open boundaries (ghost atoms handle neighbors);
        non-decomposed dims keep PBC so the NL builder wraps correctly.
        """
        if self._partitioner is None:
            return torch.ones(3, dtype=torch.bool, device=device)
        grid = self._partitioner.rank_grid  # (Px, Py, Pz)
        return torch.tensor(
            [grid[0] > 1, grid[1] > 1, grid[2] > 1],
            dtype=torch.bool,
            device=device,
        )

    def _save_geometry(self, padded_batch: Batch) -> _GeometrySnapshot:
        """Save original cell and PBC, then disable PBC for decomposed dims.

        Non-decomposed dimensions keep ``pbc=True`` so the NL builder
        uses shift vectors for periodic neighbors in X/Y.  Only the
        decomposed dimension (Z) is set to ``pbc=False`` because ghost
        atoms handle that boundary explicitly.
        """
        positions = padded_batch.positions
        original_cell = padded_batch.cell.clone()
        original_pbc = (
            padded_batch.pbc.clone()
            if hasattr(padded_batch, "pbc") and padded_batch.pbc is not None
            else torch.ones(1, 3, dtype=torch.bool, device=positions.device)
        )

        # Disable PBC only for decomposed dims; keep PBC for non-decomposed.
        decomposed = self._decomposed_dims_mask(positions.device)
        n_graphs = max(1, padded_batch.cell.shape[0])
        new_pbc = original_pbc.clone().expand(n_graphs, -1).clone()
        new_pbc[:, decomposed] = False
        padded_batch.pbc = new_pbc

        snapshot = _GeometrySnapshot(
            original_cell=original_cell,
            original_pbc=original_pbc,
            pos_min=torch.zeros(3, device=positions.device),
        )
        self._geometry_snapshot = snapshot
        return snapshot

    def _restore_geometry(
        self, padded_batch: Batch, snapshot: _GeometrySnapshot
    ) -> None:
        """Restore original cell, PBC, and undo position shift."""
        if snapshot.pos_min is not None:
            padded_batch.positions = padded_batch.positions + snapshot.pos_min
        padded_batch.cell = snapshot.original_cell
        padded_batch.pbc = snapshot.original_pbc

    @staticmethod
    def _ensure_output_tensors(batch: Batch) -> None:
        """Pre-allocate forces and energies if they don't exist.

        ``BaseDynamics.compute()`` writes model outputs in-place via
        ``copy_()``, so the destination tensors must already exist on the
        batch.  Ghost exchange builds a new padded batch each step, which
        may lack these fields.
        """
        n_total = batch.num_nodes
        n_graphs = batch.num_graphs
        device = batch.positions.device
        dtype = batch.positions.dtype
        if not hasattr(batch, "forces") or batch.forces is None:
            batch.forces = torch.zeros(n_total, 3, dtype=dtype, device=device)
        elif batch.forces.shape[0] != n_total:
            # Forces exist but wrong size (e.g., from pre-ghost-exchange batch)
            batch.forces = torch.zeros(n_total, 3, dtype=dtype, device=device)
        if not hasattr(batch, "energies") or batch.energies is None:
            batch.energies = torch.zeros(
                n_graphs, 1, dtype=torch.float64, device=device
            )
        elif batch.energies.shape[0] != n_graphs:
            batch.energies = torch.zeros(
                n_graphs, 1, dtype=torch.float64, device=device
            )

    # ------------------------------------------------------------------
    # Gather
    # ------------------------------------------------------------------

    def gather(self, local_batch: Batch, dst: int = 0) -> Batch | None:
        """Gather local batches back to a full batch on rank *dst*.

        Uses ``physicsnemo.distributed.utils.gather_v_wrapper`` to
        variable-length gather per-atom tensors from all ranks.

        Parameters
        ----------
        local_batch : Batch
            This rank's owned-atom batch.
        dst : int
            Destination rank that receives the full batch.

        Returns
        -------
        Batch | None
            The full batch on rank *dst*, ``None`` on other ranks.
        """
        if not dist.is_initialized():
            return local_batch

        from physicsnemo.distributed.utils import gather_v_wrapper

        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        device = local_batch.positions.device
        group = self._config.mesh.get_group() if self._config.mesh is not None else None

        # Collect per-rank atom counts so gather_v_wrapper knows the sizes.
        n_local = torch.tensor(
            [local_batch.positions.shape[0]], dtype=torch.int64, device=device
        )
        world_size = dist.get_world_size()
        all_counts = [
            torch.zeros(1, dtype=torch.int64, device=device) for _ in range(world_size)
        ]
        dist.all_gather(all_counts, n_local, group=group)
        per_rank_sizes = [c.item() for c in all_counts]

        # Gather per-atom fields.
        positions = gather_v_wrapper(
            tensor=local_batch.positions,
            sizes=per_rank_sizes,
            dim=0,
            dst=dst,
            group=group,
        )
        atomic_numbers = gather_v_wrapper(
            tensor=local_batch.atomic_numbers.unsqueeze(-1).to(torch.int64)
            if local_batch.atomic_numbers.ndim == 1
            else local_batch.atomic_numbers.to(torch.int64),
            sizes=per_rank_sizes,
            dim=0,
            dst=dst,
            group=group,
        )
        atomic_masses = gather_v_wrapper(
            tensor=(
                local_batch.atomic_masses.unsqueeze(-1)
                if hasattr(local_batch, "atomic_masses")
                and local_batch.atomic_masses is not None
                and local_batch.atomic_masses.ndim == 1
                else local_batch.atomic_masses
                if hasattr(local_batch, "atomic_masses")
                and local_batch.atomic_masses is not None
                else torch.ones(local_batch.positions.shape[0], 1, device=device)
            ),
            sizes=per_rank_sizes,
            dim=0,
            dst=dst,
            group=group,
        )

        has_velocities = (
            hasattr(local_batch, "velocities") and local_batch.velocities is not None
        )
        velocities = gather_v_wrapper(
            tensor=local_batch.velocities
            if has_velocities
            else torch.zeros_like(local_batch.positions),
            sizes=per_rank_sizes,
            dim=0,
            dst=dst,
            group=group,
        )

        has_forces = hasattr(local_batch, "forces") and local_batch.forces is not None
        forces = gather_v_wrapper(
            tensor=local_batch.forces
            if has_forces
            else torch.zeros_like(local_batch.positions),
            sizes=per_rank_sizes,
            dim=0,
            dst=dst,
            group=group,
        )

        if self._domain_rank != dst:
            return None

        # Build the full batch on the destination rank.
        # Squeeze back 1-D fields that were unsqueezed for gather.
        if (
            atomic_numbers is not None
            and atomic_numbers.ndim == 2
            and atomic_numbers.shape[-1] == 1
        ):
            atomic_numbers = atomic_numbers.squeeze(-1)
        if (
            atomic_masses is not None
            and atomic_masses.ndim == 2
            and atomic_masses.shape[-1] == 1
        ):
            atomic_masses = atomic_masses.squeeze(-1)

        data = AtomicData(
            positions=positions,
            atomic_numbers=atomic_numbers,
            atomic_masses=atomic_masses,
            cell=local_batch.cell.clone()
            if hasattr(local_batch, "cell") and local_batch.cell is not None
            else None,
            pbc=local_batch.pbc.clone()
            if hasattr(local_batch, "pbc") and local_batch.pbc is not None
            else None,
        )
        if has_velocities and velocities is not None:
            data.add_node_property("velocities", velocities)
        if has_forces and forces is not None:
            data.add_node_property("forces", forces)

        full_batch = BatchCls.from_data_list([data], device=device)
        return full_batch

    # ------------------------------------------------------------------
    # Hook overrides
    # ------------------------------------------------------------------

    def _build_context(self, batch: Batch) -> HookContext:
        """Build a ``HookContext`` with domain-parallel fields populated.

        Parameters
        ----------
        batch : Batch
            Current batch being processed.

        Returns
        -------
        HookContext
            Context with ``n_owned``, ``domain_mesh``, ``is_domain_parallel``,
            and ``global_cell`` populated.
        """
        ctx = super()._build_context(batch)
        ctx.n_owned = self._n_owned
        ctx.domain_mesh = self._config.mesh
        ctx.is_domain_parallel = True
        ctx.global_cell = (
            self._geometry_snapshot.original_cell
            if self._geometry_snapshot is not None
            else None
        )
        return ctx

    def _call_hooks(self, stage: DynamicsStage, batch: Batch) -> None:
        """Invoke hooks respecting their ``HookScope``.

        For the POC:

        - ``LOCAL``: hook runs on every rank with the local batch.
        - ``GLOBAL``: an ``all_reduce`` on ``batch.energies`` is performed
          before running the hook.
        - ``RANK_ZERO``: deferred — logs a warning.

        Parameters
        ----------
        stage : DynamicsStage
            Current workflow stage.
        batch : Batch
            Current batch being processed.
        """
        ctx = self._build_context(batch)

        for hook in self.hooks:
            # Check stage match
            runs_on_stage = getattr(hook, "_runs_on_stage", None)
            if runs_on_stage is not None:
                if not runs_on_stage(stage):
                    continue
            elif stage != hook.stage:
                continue

            # Frequency gating
            if self.step_count % hook.frequency != 0:
                continue

            scope = getattr(hook, "scope", HookScope.LOCAL)

            if scope == HookScope.GLOBAL:
                # All-reduce system scalars before running.
                if (
                    dist.is_initialized()
                    and hasattr(batch, "energies")
                    and batch.energies is not None
                ):
                    dist.all_reduce(batch.energies, op=dist.ReduceOp.SUM)
                hook(ctx, stage)

            elif scope == HookScope.RANK_ZERO:
                full_batch = self.gather(batch, dst=0)
                if self._domain_rank == 0 and full_batch is not None:
                    ctx_full = self._build_context(full_batch)
                    hook(ctx_full, stage)

            else:
                # LOCAL (default): run on every rank.
                hook(ctx, stage)

    # ------------------------------------------------------------------
    # Force priming
    # ------------------------------------------------------------------

    def _prime_forces(self, batch: Batch) -> Batch:
        """Run one ghost-exchange + compute pass to initialize forces.

        Velocity Verlet's ``pre_update`` uses forces from the previous
        step.  On the very first step there are no forces yet, so we
        run the neighbor list hook + model forward once to populate
        ``batch.forces`` and ``batch.energies`` before the step loop.
        """
        logger.info("[rank %d] priming forces (initial compute)", self._domain_rank)

        # Ghost exchange
        if self._ghost_exchanger is not None:
            padded_batch, n_owned = self._ghost_exchanger.exchange(batch)
        else:
            padded_batch = batch
            n_owned = batch.positions.shape[0]
        self._n_owned = n_owned

        snapshot = self._save_geometry(padded_batch)
        self._ensure_output_tensors(padded_batch)

        # Compute AABB and shift positions (no pre_update needed for
        # priming — positions haven't moved).  Force fresh computation
        # to initialize the cached AABB state.
        self._apply_aabb(padded_batch, force_recompute=True)
        self._invalidate_nl_ref()
        self._prev_n_padded = padded_batch.num_nodes

        # Build neighbor list + model forward.
        self._dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, padded_batch)
        self._dynamics.compute(padded_batch)
        self._dynamics._call_hooks(DynamicsStage.AFTER_COMPUTE, padded_batch)

        self._restore_geometry(padded_batch, snapshot)

        if self._ghost_exchanger is not None:
            batch = self._ghost_exchanger.strip(padded_batch, n_owned)
        else:
            batch = padded_batch

        logger.info("[rank %d] force priming complete", self._domain_rank)
        return batch

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self, batch: Batch, n_steps: int | None = None) -> Batch:
        """Run the domain-decomposed simulation for *n_steps* steps.

        Reuses the ``BaseDynamics.run()`` loop logic but operates on the
        local subdomain batch.  The caller is responsible for calling
        ``partition()`` before ``run()`` to obtain the local batch.

        Parameters
        ----------
        batch : Batch
            The local batch (from ``partition()``).
        n_steps : int | None, optional
            Number of steps.  Falls back to ``self.n_steps``.

        Returns
        -------
        Batch
            The local batch after all steps.
        """
        resolved = n_steps if n_steps is not None else self.n_steps
        if resolved is None:
            raise ValueError(
                "No step count provided. Either pass `n_steps` to run() "
                "or set it at construction time via "
                f"`{type(self).__name__}(..., n_steps=N)`."
            )
        self._open_hooks()
        try:
            # Prime forces before the first step so that pre_update (the
            # first half-kick in velocity Verlet) has forces to work with.
            # This mirrors FusedStage.run() which calls compute() before
            # the step loop.
            if not self._forces_primed:
                batch = self._prime_forces(batch)
                self._forces_primed = True

            for _ in range(resolved):
                batch, _converged = self.step(batch)
        finally:
            self._close_hooks()
        return batch
