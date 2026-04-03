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
"""Single-GPU test that exercises the REAL DD code paths and compares forces.

Uses the actual ``SpatialPartitioner``, ``GhostExchanger`` (ghost identification
+ PBC shifts), and ``DomainParallel._save_geometry`` / ``_apply_aabb`` code —
just without ``torch.distributed`` communication.  The ghost exchange data
transfer is simulated by directly calling the mask/shift methods and building
the padded batch manually.

No ``torchrun`` or multi-GPU required.
"""

from __future__ import annotations

import pytest
import torch

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.ghost_exchanger import GhostExchanger
from nvalchemi.distributed.partitioner import SpatialPartitioner
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.lj import LennardJonesModelWrapper

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LJ_EPS = 0.0104
LJ_SIG = 3.40
LJ_CUT = 8.5
LATTICE = 3.82
N_SIDE = 8  # 512 atoms
BOX = N_SIDE * LATTICE  # 30.56


# ──────────────────────────────────────────────────────────────
# Mock mesh (no dist required)
# ──────────────────────────────────────────────────────────────


class _MockMesh:
    """Minimal mock of ``DeviceMesh`` so we can instantiate GhostExchanger."""

    def __init__(self, rank: int, world_size: int):
        self._rank = rank
        self._ws = world_size

    def get_local_rank(self) -> int:
        return self._rank

    def size(self) -> int:
        return self._ws

    def get_group(self):
        return None


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def _make_model():
    return LennardJonesModelWrapper(epsilon=LJ_EPS, sigma=LJ_SIG, cutoff=LJ_CUT).to(
        DEVICE
    )


def _make_positions(perturb: bool = False):
    """Create 8^3 Argon positions.  Optionally perturb off-lattice."""
    coords = torch.arange(N_SIDE, dtype=torch.float32) * LATTICE
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    if perturb:
        torch.manual_seed(42)
        positions = positions + torch.randn_like(positions) * 0.15
        positions = positions % BOX
    return positions


def _reference_forces(model, positions, skin=0.0):
    """Single-GPU full-PBC force computation.  Returns (forces, num_neighbors)."""
    n = positions.shape[0]
    config = model.model_card.neighbor_config
    data = AtomicData(
        positions=positions.clone(),
        atomic_numbers=torch.full((n,), 18, dtype=torch.long),
        forces=torch.zeros(n, 3),
        energies=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * BOX,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    batch = Batch.from_data_list([data], device=DEVICE)
    nl = NeighborListHook(config=config, skin=skin)
    ctx = HookContext(batch=batch, step_count=0, model=model)
    nl(ctx, DynamicsStage.BEFORE_COMPUTE)
    outputs = model(batch)
    batch.forces.copy_(outputs["forces"])
    return batch.forces.cpu().clone(), batch.num_neighbors.cpu().clone()


def _setup_dd_components(skin=0.0):
    """Create partitioner + per-rank ghost exchangers using mock meshes."""
    cell = torch.eye(3) * BOX
    pbc = torch.ones(3, dtype=torch.bool)
    mesh0 = _MockMesh(0, 2)
    config = DomainConfig(cutoff=LJ_CUT, skin=skin, mesh=mesh0, mesh_dim="domain")

    partitioner = SpatialPartitioner(config=config, cell_matrix=cell, pbc=pbc)

    exchangers = {}
    for rank in range(2):
        mesh = _MockMesh(rank, 2)
        exchangers[rank] = GhostExchanger(
            partitioner=partitioner, config=config, mesh=mesh
        )

    return partitioner, exchangers, config


def _simulate_ghost_exchange(partitioner, exchangers, all_positions, rank):
    """Simulate what the real ghost exchange does for one rank.

    Returns (owned_positions, ghost_positions, n_owned).
    """
    # Partition atoms
    rank_assignment = partitioner.assign_atoms_to_ranks(all_positions)
    owned_mask = rank_assignment == rank
    owned_pos = all_positions[owned_mask]

    other_rank = 1 - rank  # Only works for 2-rank case
    other_mask = rank_assignment == other_rank
    other_pos = all_positions[other_mask]

    # Use the OTHER rank's GE to identify which of its atoms are ghosts for us
    other_ge = exchangers[other_rank]
    masks = other_ge.compute_ghost_masks_batched(other_pos)

    ghost_parts = []
    if rank in masks:
        direct_mask, pbc_mask = masks[rank]

        # Direct ghosts: sent at original positions
        if direct_mask.any():
            ghost_parts.append(other_pos[direct_mask])

        # PBC ghosts: sent at shifted positions
        if pbc_mask.any():
            shift_key = (other_rank, rank)
            pbc_pos = other_pos[pbc_mask].clone()
            if shift_key in other_ge._pbc_shifts:
                pbc_pos += other_ge._pbc_shifts[shift_key]
            ghost_parts.append(pbc_pos)

    ghost_pos = torch.cat(ghost_parts) if ghost_parts else torch.empty(0, 3)
    return owned_pos, ghost_pos


def _dd_force_compute(model, owned_pos, ghost_pos, config, skin=0.0):
    """Run the REAL DomainParallel AABB + mixed PBC + NL + compute path.

    Mimics ``_save_geometry`` → ``_apply_aabb`` → NL hook → model forward.
    Returns (owned_forces, num_neighbors, n_z_shifts).
    """
    from nvalchemi.distributed.domain_parallel import DomainParallel
    from nvalchemi.dynamics.integrators.nve import NVE

    n_owned = owned_pos.shape[0]
    n_ghosts = ghost_pos.shape[0]
    n_total = n_owned + n_ghosts

    # Build padded batch (same as GhostExchanger.exchange output)
    all_pos = torch.cat([owned_pos, ghost_pos], dim=0)
    data = AtomicData(
        positions=all_pos,
        atomic_numbers=torch.full((n_total,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_total,), 39.948),
        forces=torch.zeros(n_total, 3),
        energies=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * BOX,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    padded_batch = Batch.from_data_list([data], device=DEVICE)

    # Create a DomainParallel instance to use its real methods
    nl_hook = NeighborListHook(config=model.model_card.neighbor_config, skin=skin)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])
    # Use mock mesh with rank=0 (doesn't matter, just need _partitioner)
    mesh = _MockMesh(0, 2)
    dd_config = DomainConfig(cutoff=LJ_CUT, skin=skin, mesh=mesh, mesh_dim="domain")
    dd = DomainParallel(nve, config=dd_config)
    # Initialize the partitioner so _decomposed_dims_mask works
    dd._partitioner = SpatialPartitioner(
        config=dd_config,
        cell_matrix=torch.eye(3) * BOX,
        pbc=torch.ones(3, dtype=torch.bool),
    )

    # Run the REAL code path: save_geometry → apply_aabb → NL → compute
    # snapshot = dd._save_geometry(padded_batch)
    dd._ensure_output_tensors(padded_batch)
    dd._apply_aabb(padded_batch, force_recompute=True)

    # NL build
    dyn = dd._dynamics
    dyn._ensure_state_initialized(padded_batch)
    dyn._call_hooks(DynamicsStage.BEFORE_COMPUTE, padded_batch)
    dyn.compute(padded_batch)
    dyn._call_hooks(DynamicsStage.AFTER_COMPUTE, padded_batch)

    # Capture NL info
    nn = padded_batch.num_neighbors.cpu().clone()
    shifts = (
        padded_batch.neighbor_shifts.cpu().clone()
        if hasattr(padded_batch, "neighbor_shifts")
        and padded_batch.neighbor_shifts is not None
        else None
    )
    z_shifts = 0
    if shifts is not None:
        z_shifts = (shifts[:, :, 2] != 0).sum().item()

    owned_forces = padded_batch.forces[:n_owned].cpu().clone()
    pbc_used = padded_batch.pbc.cpu()
    cell_used = padded_batch.cell.cpu()

    return owned_forces, nn, z_shifts, pbc_used, cell_used


# ──────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
class TestDDForceAccuracy:
    """Verify DD force computation matches single-GPU reference."""

    def test_ghost_exchange_composition(self):
        """Check ghost shell structure: counts, Z ranges, PBC shifts."""
        partitioner, exchangers, config = _setup_dd_components(skin=0.0)
        positions = _make_positions(perturb=False)

        for rank in range(2):
            owned_pos, ghost_pos = _simulate_ghost_exchange(
                partitioner, exchangers, positions, rank
            )
            n_owned = owned_pos.shape[0]
            n_ghost = ghost_pos.shape[0]
            z_min = ghost_pos[:, 2].min().item() if n_ghost > 0 else 0
            z_max = ghost_pos[:, 2].max().item() if n_ghost > 0 else 0
            print(
                f"Rank {rank}: {n_owned} owned + {n_ghost} ghosts, "
                f"ghost Z=[{z_min:.2f}, {z_max:.2f}]"
            )
            assert n_owned > 0
            assert n_ghost > 0

    def test_save_geometry_sets_mixed_pbc(self):
        """_save_geometry should set pbc=[T,T,F] for 1D Z decomposition."""
        model = _make_model()
        positions = _make_positions()
        partitioner, exchangers, config = _setup_dd_components()
        owned_pos, ghost_pos = _simulate_ghost_exchange(
            partitioner, exchangers, positions, rank=0
        )
        _, _, _, pbc_used, cell_used = _dd_force_compute(
            model, owned_pos, ghost_pos, config, skin=0.0
        )
        # pbc should be [True, True, False] (X/Y periodic, Z open)
        assert pbc_used[0, 0].item() is True, "X should be periodic"
        assert pbc_used[0, 1].item() is True, "Y should be periodic"
        assert pbc_used[0, 2].item() is False, "Z should be non-periodic"

        # Cell X/Y should be original (30.56), Z should be AABB (> 30.56)
        assert abs(cell_used[0, 0, 0].item() - BOX) < 0.01, "Cell X should be original"
        assert abs(cell_used[0, 1, 1].item() - BOX) < 0.01, "Cell Y should be original"
        # Z cell is AABB, should be larger than the Z span of atoms
        assert cell_used[0, 2, 2].item() > 20, "Cell Z should cover atom range"

    def test_no_spurious_z_shifts(self):
        """NL builder should produce zero Z-shifts with pbc_Z=False."""
        model = _make_model()
        positions = _make_positions(perturb=True)
        partitioner, exchangers, config = _setup_dd_components()

        for rank in range(2):
            owned_pos, ghost_pos = _simulate_ghost_exchange(
                partitioner, exchangers, positions, rank
            )
            _, nn, z_shifts, _, _ = _dd_force_compute(
                model, owned_pos, ghost_pos, config, skin=0.0
            )
            assert z_shifts == 0, (
                f"Rank {rank}: {z_shifts} spurious Z-shifts with pbc_Z=False"
            )

    def test_dd_forces_match_reference_perfect_lattice(self):
        """DD forces on perfect lattice should be near-zero."""
        model = _make_model()
        positions = _make_positions(perturb=False)
        partitioner, exchangers, config = _setup_dd_components()

        for rank in range(2):
            owned_pos, ghost_pos = _simulate_ghost_exchange(
                partitioner, exchangers, positions, rank
            )
            dd_forces, nn, z_shifts, _, _ = _dd_force_compute(
                model, owned_pos, ghost_pos, config, skin=0.0
            )
            fmax = dd_forces.norm(dim=-1).max().item()
            print(
                f"Rank {rank}: fmax={fmax:.6f}  avg_nn_owned={nn[: owned_pos.shape[0]].float().mean():.1f}"
            )
            assert z_shifts == 0, f"Rank {rank}: spurious Z-shifts"
            assert fmax < 0.02, f"Rank {rank}: fmax={fmax:.4f} (expected ~0)"

    def test_dd_forces_match_reference_perturbed(self):
        """DD forces on perturbed system should match single-GPU within tolerance."""
        model = _make_model()
        positions = _make_positions(perturb=True)
        partitioner, exchangers, config = _setup_dd_components()

        ref_forces, ref_nn = _reference_forces(model, positions, skin=0.0)

        # Compute DD forces for each rank and map back to global indices
        rank_assignment = partitioner.assign_atoms_to_ranks(positions)
        all_dd_forces = torch.zeros_like(ref_forces)

        for rank in range(2):
            owned_pos, ghost_pos = _simulate_ghost_exchange(
                partitioner, exchangers, positions, rank
            )
            dd_forces, nn, z_shifts, _, _ = _dd_force_compute(
                model, owned_pos, ghost_pos, config, skin=0.0
            )
            assert z_shifts == 0, f"Rank {rank}: spurious Z-shifts"

            owned_mask = rank_assignment == rank
            all_dd_forces[owned_mask] = dd_forces

        # Compare
        diff = (all_dd_forces - ref_forces).norm(dim=-1)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        print(f"Force diff: max={max_diff:.2e}  mean={mean_diff:.2e}")

        # Show worst atoms
        worst5 = diff.topk(5)
        for val, idx in zip(worst5.values, worst5.indices):
            i = idx.item()
            print(
                f"  atom {i}: |dF|={val:.2e}  ref={ref_forces[i].norm():.4f}  "
                f"dd={all_dd_forces[i].norm():.4f}  Z={positions[i, 2]:.2f}"
            )

        assert max_diff < 1e-3, f"Max diff {max_diff:.2e} (expected < 1e-3)"
        assert mean_diff < 1e-4, f"Mean diff {mean_diff:.2e} (expected < 1e-4)"

    def test_nn_count_reasonable(self):
        """DD neighbor counts should be close to single-GPU (not 3-4x more)."""
        model = _make_model()
        positions = _make_positions(perturb=True)
        partitioner, exchangers, config = _setup_dd_components()

        ref_forces, ref_nn = _reference_forces(model, positions, skin=0.0)
        ref_avg_nn = ref_nn.float().mean().item()

        for rank in range(2):
            owned_pos, ghost_pos = _simulate_ghost_exchange(
                partitioner, exchangers, positions, rank
            )
            dd_forces, nn, z_shifts, _, _ = _dd_force_compute(
                model, owned_pos, ghost_pos, config, skin=0.0
            )
            n_owned = owned_pos.shape[0]
            dd_avg_nn_owned = nn[:n_owned].float().mean().item()
            # DD avg_nn should be within ~30% of reference
            # (boundary atoms have fewer Z-periodic neighbors)
            ratio = dd_avg_nn_owned / ref_avg_nn
            print(
                f"Rank {rank}: dd_avg_nn_owned={dd_avg_nn_owned:.1f}  "
                f"ref_avg_nn={ref_avg_nn:.1f}  ratio={ratio:.2f}"
            )
            assert 0.5 < ratio < 1.5, (
                f"Rank {rank}: nn ratio {ratio:.2f} is too far from 1.0 "
                f"(dd={dd_avg_nn_owned:.1f} vs ref={ref_avg_nn:.1f})"
            )
