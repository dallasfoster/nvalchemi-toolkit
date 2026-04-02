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
import warnings
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

        # Multi-process: POC approach — broadcast the full batch from rank 0,
        # then each rank selects only its own atoms.  A production
        # implementation would use scatter_v for efficiency.
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        if self._domain_rank == 0:
            if batch is None:
                raise ValueError("batch must be provided on rank 0")
            n_atoms = batch.positions.shape[0]
            n_atoms_t = torch.tensor([n_atoms], dtype=torch.int64, device=device)
        else:
            n_atoms_t = torch.tensor([0], dtype=torch.int64, device=device)

        # Broadcast atom count so all ranks can allocate receive buffers.
        dist.broadcast(n_atoms_t, src=0)
        n_atoms = n_atoms_t.item()

        # Broadcast per-atom data: positions, velocities, atomic_numbers, atomic_masses.
        if self._domain_rank == 0:
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
        else:
            all_positions = torch.zeros(n_atoms, 3, device=device)
            all_velocities = torch.zeros(n_atoms, 3, device=device)
            all_atomic_numbers = torch.zeros(n_atoms, dtype=torch.int64, device=device)
            all_atomic_masses = torch.zeros(n_atoms, device=device)

        dist.broadcast(all_positions, src=0)
        dist.broadcast(all_velocities, src=0)
        dist.broadcast(all_atomic_numbers, src=0)
        dist.broadcast(all_atomic_masses, src=0)

        # Assign atoms to ranks and select this rank's atoms.
        rank_assignment = self._partitioner.assign_atoms_to_ranks(all_positions)
        my_mask = rank_assignment == self._domain_rank
        my_indices = torch.where(my_mask)[0]

        # Build local AtomicData → Batch for this rank's atoms.
        local_data = AtomicData(
            positions=all_positions[my_indices],
            atomic_numbers=all_atomic_numbers[my_indices],
            atomic_masses=all_atomic_masses[my_indices],
            cell=cell_matrix if cell_matrix.ndim == 3 else cell_matrix.unsqueeze(0),
            pbc=pbc if pbc.ndim == 2 else pbc.unsqueeze(0),
        )
        if all_velocities.any():
            local_data.add_node_property("velocities", all_velocities[my_indices])

        local_batch = BatchCls.from_data_list([local_data], device=device)
        self._n_owned = my_indices.shape[0]
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

        logger.info(
            "[rank %d] step %d: ghost exchange", self._domain_rank, self.step_count
        )
        # 2. Ghost exchange
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
        # 3-5. Prepare, inner step, unprepare — orchestrated directly
        #      so we can insert AABB computation between pre_update
        #      and compute (see _run_inner_step).
        snapshot = self._save_geometry(padded_batch)
        self._ensure_output_tensors(padded_batch)
        padded_batch, converged = self._run_inner_step(padded_batch)
        self._restore_geometry(padded_batch, snapshot)

        # 6. Strip ghost atoms
        if self._ghost_exchanger is not None:
            batch = self._ghost_exchanger.strip(padded_batch, n_owned)
        else:
            batch = padded_batch

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
        """Execute the inner dynamics step with AABB geometry management.

        Instead of delegating to ``self._dynamics.step()`` as a black box,
        we replicate the inner step's hook sequence so that the AABB cell
        and position shift can be inserted between ``pre_update`` (which
        moves atoms) and ``compute`` (which needs the cell-list).  This
        avoids the fragile ``_AABBPrepareHook`` and gives ``DomainParallel``
        explicit control over the geometry lifecycle.

        The sequence mirrors ``BaseDynamics.step()`` (lines 1800-1818)
        but inserts ``_apply_aabb`` + ``_invalidate_nl_ref`` after
        ``pre_update``.
        """
        dyn = self._dynamics
        dyn._ensure_state_initialized(padded_batch)

        dyn._call_hooks(DynamicsStage.BEFORE_STEP, padded_batch)

        # --- pre_update (velocity Verlet half-kick) ---
        dyn._call_hooks(DynamicsStage.BEFORE_PRE_UPDATE, padded_batch)
        dyn.pre_update(padded_batch)
        dyn._call_hooks(DynamicsStage.AFTER_PRE_UPDATE, padded_batch)

        # --- AABB geometry: compute cell from current positions ---
        # This is the key insertion point.  Positions have been moved
        # by pre_update, so the AABB now encompasses all current atoms.
        # The cell-list neighbor builder (fired next) will see positions
        # that are guaranteed to be inside the cell.
        self._apply_aabb(padded_batch)
        self._invalidate_nl_ref()

        # --- compute (neighbor list build + model forward) ---
        dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, padded_batch)
        dyn.compute(padded_batch)
        dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, padded_batch)

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

    def _apply_aabb(self, batch: Batch) -> None:
        """Compute the AABB cell from current positions and shift to origin.

        Called between ``pre_update`` and ``compute`` so the cell-list
        neighbor builder always sees positions within ``[0, box_length)``.
        Updates ``self._geometry_snapshot.pos_min`` for later restoration.
        """
        positions = batch.positions
        pos_min = positions.min(dim=0).values
        pos_max = positions.max(dim=0).values

        # Small epsilon so atoms at the exact boundary are inside.
        eps = 0.01
        pos_min = pos_min - eps
        box_lengths = (pos_max - pos_min + eps).clamp(min=1e-6)

        local_cell = torch.diag(box_lengths).unsqueeze(0)
        if batch.cell.shape[0] > 1:
            local_cell = local_cell.expand(batch.cell.shape[0], -1, -1)
        batch.cell = local_cell.to(dtype=batch.cell.dtype, device=batch.cell.device)

        # Shift positions to local origin.
        batch.positions = positions - pos_min

        # Update snapshot so _restore_geometry can undo the shift.
        if self._geometry_snapshot is not None:
            self._geometry_snapshot.pos_min = pos_min

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

    def _save_geometry(self, padded_batch: Batch) -> _GeometrySnapshot:
        """Save original cell and PBC, then set ``pbc=False``."""
        positions = padded_batch.positions
        original_cell = padded_batch.cell.clone()
        original_pbc = (
            padded_batch.pbc.clone()
            if hasattr(padded_batch, "pbc") and padded_batch.pbc is not None
            else torch.ones(1, 3, dtype=torch.bool, device=positions.device)
        )

        # Set pbc=False for open boundaries within the subdomain.
        n_graphs = max(1, padded_batch.cell.shape[0])
        padded_batch.pbc = torch.zeros(
            n_graphs, 3, dtype=torch.bool, device=positions.device
        )

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

    def gather(self) -> Batch | None:
        """All-gather local batches back to a full batch on rank 0.

        For the POC this uses ``isend``/``irecv`` — each rank sends to
        rank 0, which assembles the results.

        Returns
        -------
        Batch | None
            The full batch on rank 0, ``None`` on other ranks.
        """
        if not dist.is_initialized():
            # Single-process: nothing to gather.
            return None

        # POC: placeholder — full gather requires serializing Batch tensors
        # across ranks.  For now, log a warning and return None.
        warnings.warn(
            "DomainParallel.gather() is not yet fully implemented in the POC. "
            "Returning None.",
            stacklevel=2,
        )
        return None

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
                logger.warning(
                    "HookScope.RANK_ZERO is not yet implemented in the POC. "
                    "Hook %r will be skipped on all ranks.",
                    hook,
                )

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
        # priming — positions haven't moved).
        self._apply_aabb(padded_batch)
        self._invalidate_nl_ref()

        # Build neighbor list + model forward.
        self._dynamics._call_hooks(DynamicsStage.BEFORE_COMPUTE, padded_batch)
        self._dynamics.compute(padded_batch)
        self._dynamics._call_hooks(DynamicsStage.AFTER_COMPUTE, padded_batch)

        self._restore_geometry(padded_batch, snapshot)

        # Strip ghosts — forces on owned atoms are what we keep
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
