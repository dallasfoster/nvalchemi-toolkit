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
"""Tests for GhostExchanger (single-process, no torch.distributed needed)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.distributed import DeviceMesh  # noqa: F401 — forward ref

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.ghost_exchanger import GhostExchanger
from nvalchemi.distributed.partitioner import SpatialPartitioner

# Resolve forward reference so pydantic can validate DomainConfig.
DomainConfig.model_rebuild(_types_namespace={"DeviceMesh": DeviceMesh})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_orthorhombic_cell(lx: float, ly: float, lz: float) -> torch.Tensor:
    """Create a diagonal (orthorhombic) cell matrix."""
    return torch.diag(torch.tensor([lx, ly, lz], dtype=torch.float64))


def _make_partitioner(
    cell_matrix: torch.Tensor,
    cutoff: float,
    world_size: int = 1,
    pbc: torch.Tensor | None = None,
    grid_dims: tuple[int, int, int] | None = None,
) -> SpatialPartitioner:
    """Build a SpatialPartitioner without a real DeviceMesh."""
    if pbc is None:
        pbc = torch.tensor([True, True, True])
    config = DomainConfig(cutoff=cutoff, grid_dims=grid_dims)
    part = SpatialPartitioner.__new__(SpatialPartitioner)
    part.config = config
    part.cell_matrix = cell_matrix
    part.pbc = pbc
    part.world_size = world_size

    if config.grid_dims is not None:
        part.cells_per_dim = config.grid_dims
    else:
        part.cells_per_dim = SpatialPartitioner._compute_cells_per_dim(
            cell_matrix, config.cutoff
        )

    total_cells = part.cells_per_dim[0] * part.cells_per_dim[1] * part.cells_per_dim[2]
    if total_cells < world_size:
        part.cells_per_dim = SpatialPartitioner.refine_grid_for_ranks(
            part.cells_per_dim, world_size
        )

    part.rank_grid = SpatialPartitioner.compute_rank_grid(
        part.cells_per_dim, world_size
    )
    part._neighbor_ranks = part._compute_all_neighbor_ranks()
    return part


def _make_ghost_exchanger(
    partitioner: SpatialPartitioner,
    cutoff: float,
    skin: float = 0.0,
    rank: int = 0,
) -> GhostExchanger:
    """Build a GhostExchanger with a mock DeviceMesh."""
    config = DomainConfig(cutoff=cutoff, skin=skin)
    mock_mesh = MagicMock()
    mock_mesh.get_local_rank.return_value = rank

    exchanger = GhostExchanger(partitioner, config, mock_mesh)
    return exchanger


# ---------------------------------------------------------------------------
# Tests — PBC shift vectors
# ---------------------------------------------------------------------------


class TestPBCShiftVectorsOrthorhombic:
    """PBC shift vectors for an orthorhombic cell with a 2x1x1 grid."""

    def test_shifts_exist_for_wrapping_pair(self):
        """Rank 0 (low-x) and rank 1 (high-x) should have PBC shifts."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        assert part.rank_grid == (2, 1, 1)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        shifts = exchanger._pbc_shifts

        # Rank 1 is at high-x edge, rank 0 is at low-x edge.
        # (sender=1, receiver=0): sender at high edge, receiver at low edge
        #   => shift -= cell_matrix[0, :] = -[10, 0, 0]
        assert (1, 0) in shifts
        expected = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(1, 0)], expected)

        # (sender=0, receiver=1): sender at low edge, receiver at high edge
        #   => shift += cell_matrix[0, :] = +[10, 0, 0]
        assert (0, 1) in shifts
        expected = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(0, 1)], expected)

    def test_no_shifts_for_non_wrapping(self):
        """Adjacent ranks in interior should have no PBC shifts."""
        cell = _make_orthorhombic_cell(30.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=4, grid_dims=(4, 1, 1))
        assert part.rank_grid == (4, 1, 1)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=1)
        shifts = exchanger._pbc_shifts

        # Ranks 1 and 2 are interior neighbors — no PBC wrap.
        assert (1, 2) not in shifts
        assert (2, 1) not in shifts

    def test_shift_magnitude_equals_box_length(self):
        """Shift magnitude should equal the box length along the wrapping axis."""
        cell = _make_orthorhombic_cell(20.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=3.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=3.0, rank=0)
        shifts = exchanger._pbc_shifts

        for key, shift in shifts.items():
            assert torch.isclose(
                shift.norm(), torch.tensor(20.0, dtype=torch.float64)
            ), f"Shift for {key} has wrong magnitude: {shift.norm()}"


class TestPBCShiftVectorsTriclinic:
    """PBC shift vectors for a triclinic (sheared) cell."""

    def test_triclinic_shifts_use_full_lattice_vector(self):
        """For a sheared cell, shifts should use the full lattice vector,
        not just the diagonal element."""
        # Sheared cell: a = [10, 0, 0], b = [2, 10, 0], c = [0, 0, 10]
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=torch.float64,
        )
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        assert part.rank_grid == (2, 1, 1)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        shifts = exchanger._pbc_shifts

        # (sender=1, receiver=0): wraps along x => shift = -cell[0, :] = [-10, 0, 0]
        assert (1, 0) in shifts
        expected = torch.tensor([-10.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(1, 0)], expected)

        # (sender=0, receiver=1): wraps along x => shift = +cell[0, :] = [10, 0, 0]
        assert (0, 1) in shifts
        expected = torch.tensor([10.0, 0.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(0, 1)], expected)

    def test_triclinic_y_wrap_includes_off_diagonal(self):
        """Wrapping along y in a sheared cell should include the off-diagonal."""
        # Sheared cell: a = [10, 0, 0], b = [3, 10, 0], c = [0, 0, 10]
        cell = torch.tensor(
            [[10.0, 0.0, 0.0], [3.0, 10.0, 0.0], [0.0, 0.0, 10.0]],
            dtype=torch.float64,
        )
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(1, 2, 1))
        assert part.rank_grid == (1, 2, 1)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        shifts = exchanger._pbc_shifts

        # (sender=1, receiver=0): rank 1 at high-y, rank 0 at low-y
        #   => shift -= cell[1, :] = -[3, 10, 0]
        assert (1, 0) in shifts
        expected = torch.tensor([-3.0, -10.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(1, 0)], expected)

        # (sender=0, receiver=1): shift += cell[1, :] = [3, 10, 0]
        assert (0, 1) in shifts
        expected = torch.tensor([3.0, 10.0, 0.0], dtype=torch.float64)
        torch.testing.assert_close(shifts[(0, 1)], expected)


# ---------------------------------------------------------------------------
# Tests — Ghost identification
# ---------------------------------------------------------------------------


class TestGhostIdentification:
    """Test that atoms near domain boundaries are correctly identified as ghosts."""

    def test_atoms_near_boundary_are_ghosts(self):
        """In a 2x1x1 decomposition of a 20A box, atoms near the x=10
        boundary should be identified as ghosts for the adjacent rank."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        # rank_grid = (2, 1, 1)
        # rank 0 owns x in [0, 10), rank 1 owns x in [10, 20)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # Place atoms: one near boundary (x=9.5), one far (x=5.0)
        positions = torch.tensor(
            [[9.5, 10.0, 10.0], [5.0, 10.0, 10.0]],
            dtype=torch.float64,
        )

        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=1)
        # Atom at x=9.5 is within ghost_width=2.0 of rank 1's boundary at x=10
        assert mask[0].item() is True
        # Atom at x=5.0 is 5.0 away from x=10, beyond ghost_width=2.0
        # and also far from PBC-wrapped edge (10.0 away from x=20→0 boundary)
        assert mask[1].item() is False

    def test_ghosts_from_both_sides_of_boundary(self):
        """Both ranks should identify ghost atoms near the shared boundary."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))

        # Rank 0: atom at x=4.0 (near boundary at x=5)
        exchanger_0 = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        pos_0 = torch.tensor([[4.0, 5.0, 5.0]], dtype=torch.float64)
        mask_0 = exchanger_0.identify_ghosts_for_neighbor(pos_0, neighbor_rank=1)
        assert mask_0[0].item() is True

        # Rank 1: atom at x=6.0 (near boundary at x=5)
        exchanger_1 = _make_ghost_exchanger(part, cutoff=2.0, rank=1)
        pos_1 = torch.tensor([[6.0, 5.0, 5.0]], dtype=torch.float64)
        mask_1 = exchanger_1.identify_ghosts_for_neighbor(pos_1, neighbor_rank=0)
        assert mask_1[0].item() is True

    def test_pbc_ghost_identification(self):
        """Atoms near PBC-wrapping boundaries should be identified as ghosts."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        # rank 0 owns x in [0, 5), rank 1 owns x in [5, 10)

        # Rank 0: atom at x=0.5 — should be ghost for rank 1 via PBC wrap
        # (rank 0 at low-x, rank 1 at high-x; PBC wraps)
        exchanger_0 = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        pos = torch.tensor([[0.5, 5.0, 5.0]], dtype=torch.float64)
        mask = exchanger_0.identify_ghosts_for_neighbor(pos, neighbor_rank=1)
        # After PBC shift (+10 along x), atom appears at x=10.5
        # Rank 1 domain is [5, 10), expanded to [3, 12) with ghost_width=2
        # 10.5 is in [3, 12) but not in [5, 10), so it should be a ghost.
        assert mask[0].item() is True

    def test_multiple_atoms_vectorized(self):
        """Ghost identification should be fully vectorized over atoms."""
        # Use a larger box so that PBC-wrapped distances are large enough
        # to keep interior atoms away from both boundaries.
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=1.5, world_size=2, grid_dims=(2, 1, 1))

        exchanger = _make_ghost_exchanger(part, cutoff=1.5, rank=0)
        # rank 0 owns x in [0, 20), ghost_width=1.5
        # rank 1 domain [20, 40), expanded to [18.5, 41.5)
        # PBC-wrapped edge: rank 1 high edge at x=40 wraps to x=0,
        # so atoms within 1.5 of x=0 are also PBC ghosts.

        positions = torch.tensor(
            [
                [19.0, 20.0, 20.0],  # Near boundary at x=20 — ghost
                [
                    17.0,
                    20.0,
                    20.0,
                ],  # Just outside ghost width (2.0 > 1.5 from expanded edge at 18.5) — not ghost
                [19.5, 20.0, 20.0],  # Very near boundary — ghost
                [10.0, 20.0, 20.0],  # Far from boundary — not ghost
                [0.5, 20.0, 20.0],  # PBC ghost (within 1.5 of x=40→0 boundary)
            ],
            dtype=torch.float64,
        )

        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=1)
        # atom 0 at x=19.0: distance to x=20 boundary is 1.0 < 1.5 => ghost
        assert mask[0].item() is True
        # atom 1 at x=17.0: distance to x=20 boundary is 3.0 > 1.5 => not ghost
        assert mask[1].item() is False
        # atom 2 at x=19.5: distance to x=20 boundary is 0.5 < 1.5 => ghost
        assert mask[2].item() is True
        # atom 3 at x=10.0: distance to boundary is 10.0 > 1.5 => not ghost
        assert mask[3].item() is False
        # atom 4 at x=0.5: PBC-shifted to x=40.5, distance to rank 1
        # high edge (x=40) is 0.5 < 1.5 => PBC ghost
        assert mask[4].item() is True


class TestNoGhostsFarAtoms:
    """Atoms far from all boundaries should never be identified as ghosts."""

    def test_center_atoms_not_ghosts(self):
        """Atoms in the center of a domain should not be ghosts for any neighbor."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        # rank 0 owns [0,10) x [0,10) x [0,10), center at (5, 5, 5)

        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        positions = torch.tensor([[5.0, 5.0, 5.0]], dtype=torch.float64)

        for neighbor_rank in exchanger.neighbor_ranks:
            mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank)
            assert not mask[0].item(), (
                f"Center atom incorrectly identified as ghost for rank {neighbor_rank}"
            )

    def test_deep_interior_not_ghost(self):
        """An atom deep inside a domain should not be a ghost for any neighbor."""
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=3.0, world_size=8, grid_dims=(2, 2, 2))
        # rank 0 owns [0,20) x [0,20) x [0,20)

        exchanger = _make_ghost_exchanger(part, cutoff=3.0, rank=0)
        positions = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)

        for neighbor_rank in exchanger.neighbor_ranks:
            mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank)
            assert not mask[0].item(), (
                f"Deep interior atom incorrectly identified as ghost for rank {neighbor_rank}"
            )

    def test_many_far_atoms_all_non_ghost(self):
        """A batch of atoms far from boundaries should all be non-ghost."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=1.0, world_size=2, grid_dims=(2, 1, 1))
        # rank 0 owns x in [0, 10), ghost_width=1.0

        exchanger = _make_ghost_exchanger(part, cutoff=1.0, rank=0)
        # All atoms at x=5 — center of rank 0's domain, far from boundary
        positions = torch.tensor(
            [[5.0, y, z] for y in [3.0, 5.0, 7.0] for z in [3.0, 5.0, 7.0]],
            dtype=torch.float64,
        )

        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=1)
        assert not mask.any().item(), "No center atoms should be ghosts"


# ---------------------------------------------------------------------------
# Tests — GhostExchanger.__init__
# ---------------------------------------------------------------------------


class TestGhostExchangerInit:
    """Test GhostExchanger initialization with mock mesh."""

    def test_init_stores_config_and_partitioner(self):
        """__init__ should store partitioner, config, and ghost_width."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        assert exchanger.partitioner is part
        assert exchanger.ghost_width == 2.0
        assert exchanger.rank == 0

    def test_init_with_failing_get_local_rank(self):
        """When mesh.get_local_rank() raises, rank should default to 0."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=1, grid_dims=(1, 1, 1))
        config = DomainConfig(cutoff=2.0)

        mock_mesh = MagicMock()
        mock_mesh.get_local_rank.side_effect = RuntimeError("no mesh")

        exchanger = GhostExchanger(part, config, mock_mesh)
        assert exchanger.rank == 0

    def test_init_precomputes_neighbor_ranks(self):
        """neighbor_ranks should be populated from the partitioner."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        assert len(exchanger.neighbor_ranks) > 0
        assert all(isinstance(r, int) for r in exchanger.neighbor_ranks)
        # rank 0 should not be its own neighbor
        assert 0 not in exchanger.neighbor_ranks

    def test_init_precomputes_pbc_shifts(self):
        """_pbc_shifts should be a dict populated at init time."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        assert isinstance(exchanger._pbc_shifts, dict)
        # 2x1x1 with PBC should have shifts
        assert len(exchanger._pbc_shifts) > 0

    def test_init_with_skin(self):
        """effective_ghost_width should include the skin parameter."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, skin=1.0, rank=0)

        # effective_ghost_width = cutoff + skin = 3.0
        assert exchanger.ghost_width == 3.0


# ---------------------------------------------------------------------------
# Tests — _check_halo_region
# ---------------------------------------------------------------------------


class TestCheckHaloRegion:
    """Directly test the _check_halo_region helper."""

    def test_atom_in_halo_returns_true(self):
        """An atom in the expanded box but outside core should be flagged."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # Neighbor rank 1 owns frac [0.5, 1.0]
        frac_lo = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        frac_hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        gw_frac = exchanger._ghost_width_fractional()

        # Atom at frac x=0.45 — in expanded box but not in core
        frac_pos = torch.tensor([[0.45, 0.5, 0.5]], dtype=torch.float64)
        mask = exchanger._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)
        assert mask[0].item() is True

    def test_atom_in_core_returns_false(self):
        """An atom fully inside the core box should NOT be flagged."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        frac_lo = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        frac_hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        gw_frac = exchanger._ghost_width_fractional()

        # Atom at frac x=0.75 — inside core
        frac_pos = torch.tensor([[0.75, 0.5, 0.5]], dtype=torch.float64)
        mask = exchanger._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)
        assert mask[0].item() is False

    def test_atom_outside_expanded_box_returns_false(self):
        """An atom outside the expanded bounding box should NOT be flagged."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        frac_lo = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        frac_hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        gw_frac = exchanger._ghost_width_fractional()

        # Atom at frac x=0.1 — far outside expanded box
        frac_pos = torch.tensor([[0.1, 0.5, 0.5]], dtype=torch.float64)
        mask = exchanger._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)
        assert mask[0].item() is False

    def test_atom_exactly_on_boundary(self):
        """An atom exactly on the expanded boundary edge should be in halo."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        frac_lo = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        frac_hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        gw_frac = exchanger._ghost_width_fractional()

        # Atom exactly at the expanded lower boundary
        expanded_lo_x = 0.5 - gw_frac[0].item()
        frac_pos = torch.tensor([[expanded_lo_x, 0.5, 0.5]], dtype=torch.float64)
        mask = exchanger._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)
        # On the boundary edge — should be included (>= check)
        assert mask[0].item() is True

    def test_multiple_atoms_batch(self):
        """_check_halo_region should handle multiple atoms correctly."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        frac_lo = torch.tensor([0.5, 0.0, 0.0], dtype=torch.float64)
        frac_hi = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64)
        gw_frac = exchanger._ghost_width_fractional()

        frac_pos = torch.tensor(
            [
                [0.45, 0.5, 0.5],  # halo
                [0.75, 0.5, 0.5],  # core (excluded)
                [0.1, 0.5, 0.5],  # outside expanded box
            ],
            dtype=torch.float64,
        )
        mask = exchanger._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)
        assert mask.shape == (3,)
        assert mask[0].item() is True
        assert mask[1].item() is False
        assert mask[2].item() is False


# ---------------------------------------------------------------------------
# Tests — compute_ghost_masks_batched
# ---------------------------------------------------------------------------


class TestComputeGhostMasksBatched:
    """Test batched ghost mask computation over all neighbors."""

    def test_returns_dict_with_all_neighbors(self):
        """Should return a mask for every neighbor rank."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        positions = torch.tensor(
            [[9.5, 9.5, 9.5], [5.0, 5.0, 5.0]],
            dtype=torch.float64,
        )
        masks = exchanger.compute_ghost_masks_batched(positions)

        assert isinstance(masks, dict)
        assert set(masks.keys()) == set(exchanger.neighbor_ranks)
        for rank, mask in masks.items():
            assert mask.shape == (2,), f"Mask shape mismatch for rank {rank}"
            assert mask.dtype == torch.bool

    def test_correct_masks_for_corner_atom(self):
        """An atom near a corner should be a ghost for multiple neighbors."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        # rank 0 owns [0,10) x [0,10) x [0,10)
        # atom at (9.5, 9.5, 9.5) is near all three boundaries

        positions = torch.tensor([[9.5, 9.5, 9.5]], dtype=torch.float64)
        masks = exchanger.compute_ghost_masks_batched(positions)

        # Count how many neighbors see this as a ghost
        ghost_count = sum(1 for m in masks.values() if m[0].item())
        # Should be ghost for multiple neighbors (face, edge, corner neighbors)
        assert ghost_count >= 3, (
            f"Corner atom should be ghost for at least 3 neighbors, got {ghost_count}"
        )

    def test_center_atom_no_ghosts(self):
        """An atom in the center should not be a ghost for any neighbor."""
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        positions = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)
        masks = exchanger.compute_ghost_masks_batched(positions)

        for rank, mask in masks.items():
            assert not mask[0].item(), (
                f"Center atom incorrectly ghosted for rank {rank}"
            )

    def test_multiple_neighbors_2x2x1(self):
        """Test with a 2x2x1 grid — 4 ranks, each with multiple neighbors."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=4, grid_dims=(2, 2, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # rank 0 owns [0,10) x [0,10) x [0,20)
        # atom near x=9.5, y=9.5 — near boundary with ranks 1, 2, and 3
        positions = torch.tensor([[9.5, 9.5, 10.0]], dtype=torch.float64)
        masks = exchanger.compute_ghost_masks_batched(positions)

        assert len(masks) == len(exchanger.neighbor_ranks)
        ghost_neighbors = [r for r, m in masks.items() if m[0].item()]
        assert len(ghost_neighbors) >= 2, (
            f"Expected ghost for at least 2 neighbors, got {ghost_neighbors}"
        )


# ---------------------------------------------------------------------------
# Tests — identify_ghosts_for_neighbor edge cases
# ---------------------------------------------------------------------------


class TestIdentifyGhostsEdgeCases:
    """Edge cases for ghost identification."""

    def test_atom_exactly_on_domain_boundary(self):
        """An atom exactly on the domain boundary (frac = 0.5 in 2x1x1)."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # Atom exactly at x=10.0 (the boundary between rank 0 and rank 1)
        # In fractional coords this is 0.5, which is the frac_lo of rank 1.
        # It's in the core of rank 1, so NOT a halo atom.
        positions = torch.tensor([[10.0, 10.0, 10.0]], dtype=torch.float64)
        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=1)
        # At the boundary edge, it's in the core of rank 1, not halo
        assert mask[0].item() is False

    def test_atom_at_box_corner_with_pbc(self):
        """Atom at the corner of the box with 3D PBC should be ghost for
        diagonal neighbors via PBC wrapping."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        # rank 0 owns [0,5) x [0,5) x [0,5)

        # Atom near origin corner (0.5, 0.5, 0.5) — PBC ghost for rank 7
        # which owns [5,10) x [5,10) x [5,10)
        positions = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float64)
        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=7)
        # After PBC shift, atom should appear near rank 7's domain boundary
        assert mask[0].item() is True

    def test_pbc_wrapping_3d_all_dimensions(self):
        """Test PBC wrapping when shifts exist along all three dimensions."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # Check that PBC shifts exist for diagonal wrapping pairs
        # rank 0 coords (0,0,0), rank 7 coords (1,1,1)
        # This should have shifts in all 3 dims
        shift_key = (0, 7)
        assert shift_key in exchanger._pbc_shifts
        shift = exchanger._pbc_shifts[shift_key]
        # Shift should be +[10, 10, 10]
        expected = torch.tensor([10.0, 10.0, 10.0], dtype=torch.float64)
        torch.testing.assert_close(shift, expected)

    def test_no_pbc_dimension_skips_shift(self):
        """When PBC is disabled along one dimension, no shift along that dim."""
        cell = _make_orthorhombic_cell(10.0, 10.0, 10.0)
        pbc = torch.tensor([True, True, False])  # no PBC in z
        part = _make_partitioner(
            cell, cutoff=2.0, world_size=8, grid_dims=(2, 2, 2), pbc=pbc
        )
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # For wrapping in z (rank 0 at z=0, rank 4 at z=1), there should
        # be no z-component in the shift since PBC is off in z.
        # Check all shifts — none should have a z component
        for key, shift in exchanger._pbc_shifts.items():
            assert shift[2].item() == 0.0, (
                f"PBC shift for {key} has nonzero z-component with PBC off in z"
            )

    def test_empty_positions_returns_empty_mask(self):
        """Empty positions tensor should return an empty mask."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        positions = torch.zeros((0, 3), dtype=torch.float64)
        mask = exchanger.identify_ghosts_for_neighbor(positions, neighbor_rank=1)
        assert mask.shape == (0,)


# ---------------------------------------------------------------------------
# Tests — exchange() without distributed
# ---------------------------------------------------------------------------


class TestExchangeNoDist:
    """Test exchange() when torch.distributed is not initialized."""

    def test_returns_batch_unchanged(self):
        """Without dist initialized, exchange should return (batch, n_owned)."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        # Create a mock batch with positions
        mock_batch = MagicMock()
        mock_batch.positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )

        result_batch, n_owned = exchanger.exchange(mock_batch)
        assert result_batch is mock_batch
        assert n_owned == 2

    def test_returns_correct_n_owned_large_batch(self):
        """n_owned should match the number of atoms in positions."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        mock_batch.positions = torch.randn(100, 3, dtype=torch.float64)

        result_batch, n_owned = exchanger.exchange(mock_batch)
        assert n_owned == 100


@pytest.mark.skip(
    reason="exchange() now uses batch_isend_irecv which validates real dist functions; needs real 2-GPU test"
)
class TestExchangeWithMockedDist:
    """Test exchange() internals by mocking torch.distributed calls."""

    def test_exchange_with_no_ghosts(self):
        """When all atoms are far from boundaries, no ghosts should be sent/received."""
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        # Atoms in the center of rank 0's domain — no ghosts
        mock_batch.positions = torch.tensor([[10.0, 20.0, 20.0]], dtype=torch.float64)

        mock_work = MagicMock()
        mock_work.wait.return_value = None

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.isend", return_value=mock_work),
            patch("torch.distributed.irecv", return_value=mock_work),
        ):
            result_batch, n_owned = exchanger.exchange(mock_batch)

        assert n_owned == 1
        assert result_batch is mock_batch
        # No ghosts received, so padded_positions should equal original
        torch.testing.assert_close(exchanger._padded_positions, mock_batch.positions)

    def test_exchange_with_ghosts_sent(self):
        """When atoms are near a boundary, isend/irecv should be called."""
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        # Atom near boundary at x=19.5 (rank 0 owns [0,20))
        mock_batch.positions = torch.tensor([[19.5, 20.0, 20.0]], dtype=torch.float64)

        mock_work = MagicMock()
        mock_work.wait.return_value = None

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.isend", return_value=mock_work) as mock_isend,
            patch("torch.distributed.irecv", return_value=mock_work),
        ):
            result_batch, n_owned = exchanger.exchange(mock_batch)

        assert n_owned == 1
        # isend should have been called (count exchange + position exchange)
        assert mock_isend.call_count >= 1

    def test_exchange_with_received_ghosts(self):
        """When neighbor sends ghost atoms, padded_positions should grow."""
        cell = _make_orthorhombic_cell(40.0, 40.0, 40.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)
        n_neighbors = len(exchanger.neighbor_ranks)

        mock_batch = MagicMock()
        mock_batch.positions = torch.tensor([[10.0, 20.0, 20.0]], dtype=torch.float64)

        mock_work = MagicMock()
        mock_work.wait.return_value = None

        ghosts_per_neighbor = 2

        def fake_irecv(tensor, src):
            """Simulate receiving ghost count or positions from neighbor."""
            if tensor.shape == (1,):
                # Count exchange: simulate ghosts incoming
                tensor.fill_(ghosts_per_neighbor)
            elif len(tensor.shape) == 2 and tensor.shape[1] == 3:
                # Position exchange: fill with ghost positions
                tensor.fill_(99.0)
            return mock_work

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.isend", return_value=mock_work),
            patch("torch.distributed.irecv", side_effect=fake_irecv),
        ):
            result_batch, n_owned = exchanger.exchange(mock_batch)

        assert n_owned == 1
        expected_total = 1 + ghosts_per_neighbor * n_neighbors
        assert exchanger._padded_positions.shape[0] == expected_total
        assert exchanger._n_owned == 1


# ---------------------------------------------------------------------------
# Tests — strip()
# ---------------------------------------------------------------------------


class TestStrip:
    """Test the strip method for removing ghost atoms."""

    def test_strip_noop_when_n_owned_equals_total(self):
        """When n_owned == total atoms, strip should return the batch unchanged."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        mock_batch.positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )

        result = exchanger.strip(mock_batch, n_owned=2)
        assert result is mock_batch

    def test_strip_with_ghost_atoms_present(self):
        """When n_owned < total, strip should still return the batch
        (current POC implementation)."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        # 5 total atoms, 3 owned + 2 ghosts
        mock_batch.positions = torch.randn(5, 3, dtype=torch.float64)

        result = exchanger.strip(mock_batch, n_owned=3)
        # Current POC: returns batch as-is even when ghosts present
        assert result is mock_batch

    def test_strip_single_atom_owned(self):
        """Edge case: single owned atom, no ghosts."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=2.0, world_size=2, grid_dims=(2, 1, 1))
        exchanger = _make_ghost_exchanger(part, cutoff=2.0, rank=0)

        mock_batch = MagicMock()
        mock_batch.positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)

        result = exchanger.strip(mock_batch, n_owned=1)
        assert result is mock_batch
