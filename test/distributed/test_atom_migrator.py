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
"""Tests for AtomMigrator — indexing, assignment, and packing logic.

These tests do NOT require ``torch.distributed``; they exercise the
sort-based index construction and atom field packing/unpacking on CPU.
"""

from __future__ import annotations

import torch
from torch.distributed import DeviceMesh  # noqa: F401 — resolve forward ref

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.distributed.atom_migrator import (
    AtomMigrator,
    _build_batch_from_fields,
    _compute_field_layout,
    _field_width,
    _graph_indices_for_atoms,
    pack_atom_fields,
    packed_dim,
    unpack_atom_fields,
)
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

# Resolve the forward reference for DeviceMesh so pydantic can validate.
DomainConfig.model_rebuild(_types_namespace={"DeviceMesh": DeviceMesh})


# ---------------------------------------------------------------------------
# Helpers (reused from test_spatial_partitioner)
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


def _make_batch(positions: torch.Tensor, device: str = "cpu") -> Batch:
    """Create a single-system Batch with positions, velocities, and masses."""
    n = positions.shape[0]
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.arange(1, n + 1, dtype=torch.long),
    )
    data.add_node_property("velocities", torch.randn(n, 3, dtype=positions.dtype))
    data.add_node_property("forces", torch.randn(n, 3, dtype=positions.dtype))
    data.add_node_property("atomic_masses", torch.ones(n, dtype=positions.dtype) * 12.0)
    return Batch.from_data_list([data], device=torch.device(device))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSortBasedIndexing:
    """Verify argsort + bincount produces correct per-rank index slices."""

    def test_uniform_distribution(self):
        """Each of 4 ranks gets exactly 3 atoms."""
        # 12 atoms, pre-assigned ranks: 0,0,0,1,1,1,2,2,2,3,3,3
        new_rank = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        world_size = 4

        counts = torch.bincount(new_rank, minlength=world_size)
        sorted_idx = torch.argsort(new_rank)
        offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)])

        for r in range(world_size):
            indices = sorted_idx[offsets[r] : offsets[r + 1]]
            assert len(indices) == 3
            # All selected atoms should have new_rank == r.
            assert (new_rank[indices] == r).all()

    def test_uneven_distribution(self):
        """Some ranks get more atoms than others."""
        new_rank = torch.tensor([2, 0, 2, 2, 1, 0])
        world_size = 4

        counts = torch.bincount(new_rank, minlength=world_size)
        sorted_idx = torch.argsort(new_rank)
        offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)])

        expected_counts = {0: 2, 1: 1, 2: 3, 3: 0}
        for r in range(world_size):
            indices = sorted_idx[offsets[r] : offsets[r + 1]]
            assert len(indices) == expected_counts[r]
            if len(indices) > 0:
                assert (new_rank[indices] == r).all()

    def test_all_atoms_same_rank(self):
        """All atoms assigned to rank 1."""
        new_rank = torch.tensor([1, 1, 1, 1])
        world_size = 4

        counts = torch.bincount(new_rank, minlength=world_size)
        sorted_idx = torch.argsort(new_rank)
        offsets = torch.cat([torch.zeros(1, dtype=counts.dtype), counts.cumsum(0)])

        assert counts[0] == 0
        assert counts[1] == 4
        assert counts[2] == 0
        assert counts[3] == 0

        rank1_indices = sorted_idx[offsets[1] : offsets[2]]
        assert len(rank1_indices) == 4

    def test_sorted_idx_is_permutation(self):
        """sorted_idx should be a permutation of [0, N)."""
        new_rank = torch.tensor([3, 0, 1, 2, 0, 3])
        sorted_idx = torch.argsort(new_rank)
        assert sorted_idx.sort().values.tolist() == list(range(6))


class TestNoMigrationNeeded:
    """All atoms already in the correct domain => batch unchanged."""

    def test_all_atoms_stay(self):
        """Atoms within rank 0's domain should all be assigned to rank 0."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        # 2 ranks along x: rank 0 owns x in [0, 10), rank 1 owns x in [10, 20)
        part = _make_partitioner(cell, cutoff=5.0, world_size=2, grid_dims=(2, 1, 1))

        # All atoms in rank 0's domain (x < 10).
        positions = torch.tensor(
            [[1.0, 5.0, 5.0], [3.0, 5.0, 5.0], [9.0, 5.0, 5.0]],
            dtype=torch.float64,
        )

        new_rank = part.assign_atoms_to_ranks(positions)
        assert (new_rank == 0).all()

        # No atoms need to leave rank 0.
        counts = torch.bincount(new_rank.to(torch.int64), minlength=2)
        assert counts[0].item() == 3
        assert counts[1].item() == 0


class TestAtomsCrossingBoundary:
    """Atoms moved past domain boundary get re-assigned to correct rank."""

    def test_single_atom_crosses(self):
        """Move one atom from rank 0's domain into rank 1's domain."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=2, grid_dims=(2, 1, 1))

        # Initially: 3 atoms in rank 0's domain.
        positions = torch.tensor(
            [[1.0, 5.0, 5.0], [5.0, 5.0, 5.0], [9.0, 5.0, 5.0]],
            dtype=torch.float64,
        )
        ranks_before = part.assign_atoms_to_ranks(positions)
        assert (ranks_before == 0).all()

        # Move atom 2 past the boundary (x=9 -> x=11).
        positions[2, 0] = 11.0
        ranks_after = part.assign_atoms_to_ranks(positions)
        assert ranks_after[0].item() == 0
        assert ranks_after[1].item() == 0
        assert ranks_after[2].item() == 1  # migrated!

    def test_multiple_atoms_cross_different_ranks(self):
        """Atoms migrate to different ranks in a 4-rank decomposition."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        # 2x2x1 rank grid.
        part = _make_partitioner(cell, cutoff=5.0, world_size=4, grid_dims=(2, 2, 1))

        # Atom positions placed in specific rank domains.
        positions = torch.tensor(
            [
                [1.0, 1.0, 5.0],  # rank 0 (x<10, y<10)
                [15.0, 1.0, 5.0],  # rank 1 (x>=10, y<10)
                [1.0, 15.0, 5.0],  # rank 2 (x<10, y>=10)
                [15.0, 15.0, 5.0],  # rank 3 (x>=10, y>=10)
            ],
            dtype=torch.float64,
        )

        ranks = part.assign_atoms_to_ranks(positions)
        assert ranks[0].item() == 0
        assert ranks[1].item() == 1
        assert ranks[2].item() == 2
        assert ranks[3].item() == 3

    def test_pbc_wrap_reassignment(self):
        """Atom that wraps around PBC gets assigned to correct rank."""
        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=2, grid_dims=(2, 1, 1))

        # Atom at x=21.0 should wrap to x=1.0 via PBC => rank 0.
        positions = torch.tensor([[21.0, 5.0, 5.0]], dtype=torch.float64)
        ranks = part.assign_atoms_to_ranks(positions)
        assert ranks[0].item() == 0


class TestPackUnpack:
    """Verify round-trip packing and unpacking of atom fields."""

    def test_round_trip(self):
        """Pack and unpack should recover the original field values."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            dtype=torch.float64,
        )
        batch = _make_batch(positions)

        indices = torch.tensor([0, 2])
        layout = _compute_field_layout(batch)
        buf = pack_atom_fields(batch, indices)

        assert buf.shape[0] == 2
        assert buf.shape[1] == packed_dim(batch)

        restored = unpack_atom_fields(buf, layout)

        # Positions should match exactly.
        torch.testing.assert_close(restored["positions"], batch.positions[indices])
        # Atomic numbers (int field) should round-trip through float.
        torch.testing.assert_close(
            restored["atomic_numbers"],
            batch.atomic_numbers[indices],
        )

    def test_empty_indices(self):
        """Packing with empty indices produces a (0, D) tensor."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)

        indices = torch.tensor([], dtype=torch.long)
        buf = pack_atom_fields(batch, indices)
        assert buf.shape[0] == 0
        assert buf.shape[1] == packed_dim(batch)

    def test_packed_dim_correct(self):
        """packed_dim should sum field widths (3+3+3+1+1 = 11)."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        # positions(3) + velocities(3) + forces(3) + atomic_numbers(1) + atomic_masses(1)
        assert packed_dim(batch) == 11


# ---------------------------------------------------------------------------
# Additional coverage tests
# ---------------------------------------------------------------------------


class _FakeBatch:
    """Lightweight mock that mimics a Batch for field-access tests.

    Only exposes the attributes explicitly passed in; ``getattr`` returns
    ``None`` for anything else, matching the ``getattr(batch, name, None)``
    pattern used in the migrator helpers.
    """

    def __init__(self, **kwargs: torch.Tensor) -> None:
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_batch_no_velocities(positions: torch.Tensor) -> _FakeBatch:
    """Return a fake batch with positions, forces, atomic_numbers, atomic_masses — no velocities."""
    n = positions.shape[0]
    return _FakeBatch(
        positions=positions,
        forces=torch.randn(n, 3, dtype=positions.dtype),
        atomic_numbers=torch.arange(1, n + 1, dtype=torch.long),
        atomic_masses=torch.ones(n, dtype=positions.dtype) * 12.0,
    )


def _make_multi_graph_batch(device: str = "cpu") -> Batch:
    """Create a 3-graph batch for testing graph-level operations."""
    data_list = []
    for i in range(3):
        n = i + 2  # 2, 3, 4 atoms per graph
        positions = torch.randn(n, 3, dtype=torch.float64)
        data = AtomicData(
            positions=positions,
            atomic_numbers=torch.tensor([6] * n, dtype=torch.long),
        )
        data.add_node_property("velocities", torch.randn(n, 3, dtype=torch.float64))
        data.add_node_property("forces", torch.randn(n, 3, dtype=torch.float64))
        data.add_node_property(
            "atomic_masses", torch.ones(n, dtype=torch.float64) * 12.0
        )
        data_list.append(data)
    return Batch.from_data_list(data_list, device=torch.device(device))


class TestFieldWidth:
    """Tests for _field_width helper."""

    def test_3d_field(self):
        """A 2-D tensor (N, 3) should return width 3."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        assert _field_width(batch, "positions") == 3

    def test_1d_field(self):
        """A 1-D tensor (N,) should return width 1."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        assert _field_width(batch, "atomic_numbers") == 1

    def test_missing_field(self):
        """A missing field should return 0."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        assert _field_width(batch, "nonexistent_field") == 0


class TestComputeFieldLayoutPartial:
    """Test _compute_field_layout with missing fields."""

    def test_skips_missing_velocities(self):
        """Layout should omit velocities when the batch has none."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_fake_batch_no_velocities(positions)
        layout = _compute_field_layout(batch)
        field_names = [name for name, _, _ in layout]
        assert "velocities" not in field_names
        assert "positions" in field_names
        assert "forces" in field_names

    def test_layout_widths_match_fields(self):
        """Each width in layout should match the actual field width."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        layout = _compute_field_layout(batch)
        for name, width, dtype in layout:
            assert width == _field_width(batch, name)


class TestPackedDimPartial:
    """Test packed_dim with partial fields."""

    def test_without_velocities(self):
        """Without velocities: positions(3) + forces(3) + atomic_numbers(1) + atomic_masses(1) = 8."""
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_fake_batch_no_velocities(positions)
        assert packed_dim(batch) == 8


class TestPackMissingFields:
    """Test packing when some fields are missing."""

    def test_pack_without_velocities(self):
        """Packing should skip missing velocities and produce fewer columns."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_fake_batch_no_velocities(positions)
        indices = torch.tensor([0, 1])
        buf = pack_atom_fields(batch, indices)
        assert buf.shape == (2, 8)  # 3 + 3 + 1 + 1

    def test_round_trip_without_velocities(self):
        """Round-trip should work even without velocities."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_fake_batch_no_velocities(positions)
        indices = torch.tensor([0, 1])
        layout = _compute_field_layout(batch)
        buf = pack_atom_fields(batch, indices)
        restored = unpack_atom_fields(buf, layout)
        assert "velocities" not in restored
        torch.testing.assert_close(restored["positions"], batch.positions[indices])
        torch.testing.assert_close(restored["forces"], batch.forces[indices])


class TestUnpackSqueeze:
    """Verify squeeze behavior for width-1 columns."""

    def test_width_1_is_squeezed(self):
        """Width-1 columns should be squeezed back to 1-D."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_batch(positions)
        indices = torch.tensor([0, 1])
        layout = _compute_field_layout(batch)
        buf = pack_atom_fields(batch, indices)
        restored = unpack_atom_fields(buf, layout)
        # atomic_numbers and atomic_masses are width-1 => should be 1-D
        assert restored["atomic_numbers"].ndim == 1
        assert restored["atomic_masses"].ndim == 1

    def test_width_3_not_squeezed(self):
        """Width-3 columns should remain 2-D."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_batch(positions)
        indices = torch.tensor([0, 1])
        layout = _compute_field_layout(batch)
        buf = pack_atom_fields(batch, indices)
        restored = unpack_atom_fields(buf, layout)
        assert restored["positions"].ndim == 2
        assert restored["positions"].shape[1] == 3


class TestGraphIndicesForAtoms:
    """Test _graph_indices_for_atoms with multi-graph batches."""

    def test_single_graph(self):
        """All atoms in one graph => unique graph index is [0]."""
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_batch(positions)
        atom_indices = torch.tensor([0, 1])
        graph_ids = _graph_indices_for_atoms(batch, atom_indices)
        assert graph_ids.tolist() == [0]

    def test_multi_graph_single_atom(self):
        """Selecting one atom from graph 1 in a 3-graph batch."""
        batch = _make_multi_graph_batch()
        # Graph 0 has 2 atoms (indices 0,1), graph 1 has 3 atoms (indices 2,3,4)
        atom_indices = torch.tensor([3])  # belongs to graph 1
        graph_ids = _graph_indices_for_atoms(batch, atom_indices)
        assert graph_ids.tolist() == [1]

    def test_multi_graph_mixed(self):
        """Selecting atoms from graphs 0 and 2 => graph indices [0, 2]."""
        batch = _make_multi_graph_batch()
        # Graph 0: atoms 0,1; graph 1: atoms 2,3,4; graph 2: atoms 5,6,7,8
        atom_indices = torch.tensor([0, 5, 7])  # graphs 0 and 2
        graph_ids = _graph_indices_for_atoms(batch, atom_indices)
        assert sorted(graph_ids.tolist()) == [0, 2]


class TestBuildBatchFromFields:
    """Test _build_batch_from_fields."""

    def test_all_fields(self):
        """Build a batch with all fields present."""
        n = 4
        fields = {
            "positions": torch.randn(n, 3),
            "velocities": torch.randn(n, 3),
            "forces": torch.randn(n, 3),
            "atomic_numbers": torch.tensor([6, 7, 8, 1], dtype=torch.long),
            "atomic_masses": torch.tensor([12.0, 14.0, 16.0, 1.0]),
        }
        batch = _build_batch_from_fields(fields, torch.device("cpu"))
        assert batch.num_graphs == 1
        assert batch.positions.shape == (n, 3)
        torch.testing.assert_close(batch.positions, fields["positions"])
        torch.testing.assert_close(batch.atomic_numbers, fields["atomic_numbers"])

    def test_minimal_fields(self):
        """Build a batch with only positions and atomic_numbers."""
        n = 2
        fields = {
            "positions": torch.randn(n, 3),
            "atomic_numbers": torch.tensor([6, 8], dtype=torch.long),
        }
        batch = _build_batch_from_fields(fields, torch.device("cpu"))
        assert batch.num_graphs == 1
        assert batch.positions.shape == (n, 3)

    def test_missing_positions(self):
        """When positions missing, zeros are used."""
        n = 3
        fields = {
            "atomic_numbers": torch.tensor([6, 7, 8], dtype=torch.long),
            "forces": torch.randn(n, 3),
        }
        batch = _build_batch_from_fields(fields, torch.device("cpu"))
        assert batch.positions.shape == (n, 3)
        assert (batch.positions == 0).all()

    def test_missing_atomic_numbers(self):
        """When atomic_numbers missing, zeros are used."""
        n = 2
        fields = {
            "positions": torch.randn(n, 3),
        }
        batch = _build_batch_from_fields(fields, torch.device("cpu"))
        assert batch.atomic_numbers.shape == (n,)
        assert (batch.atomic_numbers == 0).all()

    def test_optional_fields_attached(self):
        """Velocities, forces, and atomic_masses should be attached."""
        n = 3
        fields = {
            "positions": torch.randn(n, 3),
            "atomic_numbers": torch.tensor([6, 7, 8], dtype=torch.long),
            "velocities": torch.randn(n, 3),
            "forces": torch.randn(n, 3),
            "atomic_masses": torch.tensor([12.0, 14.0, 16.0]),
        }
        batch = _build_batch_from_fields(fields, torch.device("cpu"))
        torch.testing.assert_close(batch.velocities, fields["velocities"])
        torch.testing.assert_close(batch.forces, fields["forces"])
        torch.testing.assert_close(batch.atomic_masses, fields["atomic_masses"])


class TestNeedsMigration:
    """Test AtomMigrator.needs_migration (returns False in POC — disabled)."""

    def test_returns_false_for_poc(self):
        """needs_migration returns False (migration disabled until Verlet skin wired up)."""
        migrator = AtomMigrator.__new__(AtomMigrator)
        positions = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float64)
        batch = _make_batch(positions)
        assert migrator.needs_migration(batch) is False


class TestAtomMigratorInit:
    """Test AtomMigrator.__init__ with a mock mesh."""

    def test_init_stores_attributes(self):
        """__init__ should store partitioner, config, rank, and world_size."""
        from unittest.mock import MagicMock

        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=4, grid_dims=(2, 2, 1))
        config = DomainConfig(cutoff=5.0)

        mock_mesh = MagicMock()
        mock_mesh.get_local_rank.return_value = 2
        mock_mesh.size.return_value = 4

        migrator = AtomMigrator(partitioner=part, config=config, mesh=mock_mesh)

        assert migrator.partitioner is part
        assert migrator.config is config
        assert migrator.mesh is mock_mesh
        assert migrator.rank == 2
        assert migrator.world_size == 4
        mock_mesh.get_local_rank.assert_called_once()
        mock_mesh.size.assert_called_once()

    def test_init_rank_zero(self):
        """When mesh reports rank 0, the migrator should store rank 0."""
        from unittest.mock import MagicMock

        cell = _make_orthorhombic_cell(20.0, 20.0, 20.0)
        part = _make_partitioner(cell, cutoff=5.0, world_size=2, grid_dims=(2, 1, 1))
        config = DomainConfig(cutoff=5.0)

        mock_mesh = MagicMock()
        mock_mesh.get_local_rank.return_value = 0
        mock_mesh.size.return_value = 2

        migrator = AtomMigrator(partitioner=part, config=config, mesh=mock_mesh)

        assert migrator.rank == 0
        assert migrator.world_size == 2


class TestMigrateNoDistributed:
    """Test AtomMigrator.migrate when dist is NOT initialized."""

    def test_returns_batch_unchanged(self):
        """Without dist initialized, migrate should return the same batch."""
        migrator = AtomMigrator.__new__(AtomMigrator)
        positions = torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64
        )
        batch = _make_batch(positions)
        result = migrator.migrate(batch)
        assert result is batch
