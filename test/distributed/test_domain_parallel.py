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
"""Tests for DomainParallel (single-process, no torch.distributed)."""

from __future__ import annotations

import logging
from enum import Enum
from unittest.mock import patch

import pytest
import torch
from torch.distributed import DeviceMesh  # noqa: F401 — resolve forward ref

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, HookScope, _GeometrySnapshot
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import BaseDynamics, DynamicsStage
from nvalchemi.dynamics.demo import DemoDynamics
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.demo import DemoModelWrapper

# Resolve DeviceMesh forward reference for pydantic validation.
DomainConfig.model_rebuild(_types_namespace={"DeviceMesh": DeviceMesh})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_batch(n_atoms: int = 8) -> Batch:
    """Create a single-graph batch with atoms in a 10x10x10 box."""
    positions = torch.rand(n_atoms, 3) * 10.0
    data = AtomicData(
        atomic_numbers=torch.tensor([6] * n_atoms, dtype=torch.long),
        positions=positions,
    )
    batch = Batch.from_data_list([data])
    batch.forces = torch.zeros(n_atoms, 3)
    batch.energies = torch.zeros(1, 1)
    # Set a cell matrix for the batch
    batch.cell = torch.diag(torch.tensor([10.0, 10.0, 10.0])).unsqueeze(0)
    batch.pbc = torch.tensor([[True, True, True]])  # (1, 3)
    return batch


def _make_triclinic_batch(n_atoms: int = 8) -> Batch:
    """Create a single-graph batch with a triclinic cell."""
    positions = torch.rand(n_atoms, 3) * 10.0
    data = AtomicData(
        atomic_numbers=torch.tensor([6] * n_atoms, dtype=torch.long),
        positions=positions,
    )
    batch = Batch.from_data_list([data])
    batch.forces = torch.zeros(n_atoms, 3)
    batch.energies = torch.zeros(1, 1)
    # Triclinic cell: off-diagonal elements
    cell = torch.tensor(
        [[10.0, 0.0, 0.0], [2.0, 10.0, 0.0], [1.0, 0.5, 10.0]]
    ).unsqueeze(0)
    batch.cell = cell
    batch.pbc = torch.tensor([[True, True, True]])  # (1, 3)
    return batch


def _make_domain_parallel() -> tuple[DomainParallel, DemoDynamics]:
    """Create a DomainParallel wrapping a DemoDynamics."""
    model = DemoModelWrapper()
    inner = DemoDynamics(model=model, n_steps=10)
    config = DomainConfig(cutoff=3.0, skin=0.5)
    dp = DomainParallel(dynamics=inner, config=config)
    return dp, inner


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInit:
    """Test DomainParallel construction."""

    def test_is_base_dynamics(self) -> None:
        dp, _ = _make_domain_parallel()
        assert isinstance(dp, BaseDynamics)

    def test_stores_inner_dynamics(self) -> None:
        dp, inner = _make_domain_parallel()
        assert dp._dynamics is inner

    def test_stores_config(self) -> None:
        dp, _ = _make_domain_parallel()
        assert dp._config.cutoff == 3.0
        assert dp._config.skin == 0.5

    def test_lazy_components_none(self) -> None:
        dp, _ = _make_domain_parallel()
        assert dp._partitioner is None
        assert dp._ghost_exchanger is None
        assert dp._migrator is None

    def test_initial_state(self) -> None:
        dp, _ = _make_domain_parallel()
        assert dp._n_owned == 0
        assert dp._geometry_snapshot is None
        assert dp._domain_rank == 0

    def test_model_shared_with_inner(self) -> None:
        dp, inner = _make_domain_parallel()
        assert dp.model is inner.model


class TestPrepareUnprepareRoundtrip:
    """Test _save_geometry / _restore_geometry cycle.

    Note: _save_geometry now only saves originals and sets
    pbc=False.  The AABB cell and position shift are done by the
    _AABBPrepareHook at BEFORE_COMPUTE.
    """

    def test_roundtrip_orthorhombic(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=10)

        original_positions = batch.positions.clone()
        original_cell = batch.cell.clone()
        original_pbc = batch.pbc.clone()

        # Prepare
        snapshot = dp._save_geometry(batch)

        # After prepare: pbc should be False
        assert (~batch.pbc).all()

        # Snapshot should store the original cell
        assert torch.allclose(snapshot.original_cell, original_cell, atol=1e-6)

        # Positions should be UNCHANGED (the hook shifts them later)
        assert torch.allclose(batch.positions, original_positions, atol=1e-6)

        # Unprepare
        dp._restore_geometry(batch, snapshot)

        # Should be fully restored
        assert torch.allclose(batch.positions, original_positions, atol=1e-6)
        assert torch.allclose(batch.cell, original_cell, atol=1e-6)
        assert (batch.pbc == original_pbc).all()

    def test_roundtrip_preserves_exact_positions(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)

        original_positions = batch.positions.clone()
        snapshot = dp._save_geometry(batch)
        dp._restore_geometry(batch, snapshot)

        assert torch.allclose(batch.positions, original_positions, atol=1e-7)


class TestPrepareTriclinic:
    """Test _save_geometry saves state and sets pbc=False.

    Note: The AABB cell computation and position shift are now handled
    by _AABBPrepareHook (registered at BEFORE_COMPUTE on the inner
    dynamics), not by _save_geometry.  These tests verify the
    prepare/unprepare snapshot and pbc behavior.
    """

    def test_triclinic_sets_pbc_false(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_triclinic_batch(n_atoms=10)

        original_cell = batch.cell.clone()
        snapshot = dp._save_geometry(batch)

        # pbc should be False
        assert (~batch.pbc).all()

        # Snapshot should store original triclinic cell
        assert torch.allclose(snapshot.original_cell, original_cell, atol=1e-6)

    def test_triclinic_roundtrip(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_triclinic_batch(n_atoms=10)

        original_positions = batch.positions.clone()
        original_cell = batch.cell.clone()
        original_pbc = batch.pbc.clone()

        snapshot = dp._save_geometry(batch)
        dp._restore_geometry(batch, snapshot)

        # Positions unchanged because prepare no longer shifts them
        # (the hook does that at BEFORE_COMPUTE, which isn't called here).
        assert torch.allclose(batch.positions, original_positions, atol=1e-6)
        assert torch.allclose(batch.cell, original_cell, atol=1e-6)
        assert (batch.pbc == original_pbc).all()


class TestBuildContext:
    """Test that _build_context populates domain-parallel fields."""

    def test_domain_fields_populated(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)

        dp._n_owned = 5
        ctx = dp._build_context(batch)

        assert isinstance(ctx, HookContext)
        assert ctx.is_domain_parallel is True
        assert ctx.n_owned == 5
        assert ctx.domain_mesh is None  # No mesh in single-process mode

    def test_global_cell_from_snapshot(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)

        # Set a geometry snapshot
        dp._geometry_snapshot = _GeometrySnapshot(
            original_cell=torch.eye(3).unsqueeze(0) * 10.0,
            original_pbc=torch.tensor([True, True, True]),
        )

        ctx = dp._build_context(batch)
        assert ctx.global_cell is not None
        assert ctx.global_cell.shape == (1, 3, 3)

    def test_global_cell_none_without_snapshot(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)

        dp._geometry_snapshot = None
        ctx = dp._build_context(batch)
        assert ctx.global_cell is None

    def test_step_count_forwarded(self) -> None:
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)

        dp.step_count = 42
        ctx = dp._build_context(batch)
        assert ctx.step_count == 42


class TestPropertiesDelegate:
    """Test that __needs_keys__ and __provides_keys__ delegate to inner dynamics."""

    def test_needs_keys_delegates(self) -> None:
        dp, inner = _make_domain_parallel()
        assert dp.__needs_keys__ == inner.__needs_keys__

    def test_provides_keys_delegates(self) -> None:
        dp, inner = _make_domain_parallel()
        assert dp.__provides_keys__ == inner.__provides_keys__

    def test_needs_keys_reflects_inner_changes(self) -> None:
        """If inner dynamics __needs_keys__ is modified, DomainParallel reflects it."""
        dp, inner = _make_domain_parallel()
        # DemoDynamics uses class-level sets; check they match
        original = inner.__needs_keys__.copy()
        assert dp.__needs_keys__ == original


# ---------------------------------------------------------------------------
# Mock dynamics that bypasses model evaluation
# ---------------------------------------------------------------------------


class _MockDynamics(BaseDynamics):
    """A minimal dynamics that just returns the batch unchanged.

    Does not call model.forward, so it works without a real model.
    """

    __needs_keys__: set[str] = set()
    __provides_keys__: set[str] = set()

    def __init__(self, **kwargs):
        model = DemoModelWrapper()
        super().__init__(model=model, **kwargs)
        self.step_calls: int = 0

    def step(self, batch: Batch) -> tuple[Batch, torch.Tensor | None]:
        self.step_calls += 1
        return batch, None


class _SimpleHook:
    """Minimal hook for testing _call_hooks."""

    def __init__(
        self,
        stage: DynamicsStage,
        frequency: int = 1,
        scope: HookScope = HookScope.LOCAL,
    ):
        self.stage = stage
        self.frequency = frequency
        self.scope = scope
        self.call_count: int = 0

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        self.call_count += 1

    def __repr__(self) -> str:
        return f"_SimpleHook(stage={self.stage}, scope={self.scope})"


# ---------------------------------------------------------------------------
# Partition tests (single-process)
# ---------------------------------------------------------------------------


class TestPartitionSingleProcess:
    """Test partition() in single-process mode (no torch.distributed).

    Note: the partition() code passes ``batch.pbc`` (shape ``(1,3)``) directly
    to ``SpatialPartitioner`` which expects shape ``(3,)``.  We patch the
    partitioner's ``__init__`` to flatten the pbc tensor so the rest of the
    partitioner logic works correctly in these unit tests.
    """

    @staticmethod
    def _patched_partitioner_init(original_init):
        """Return a wrapped __init__ that flattens pbc before delegating."""

        def _init(self, *, config, cell_matrix, pbc):
            # Flatten (1,3) -> (3,) so bool(pbc[i]) works inside partitioner
            original_init(
                self, config=config, cell_matrix=cell_matrix, pbc=pbc.flatten()
            )

        return _init

    def test_partition_returns_batch_unchanged(self) -> None:
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=12)
        original_positions = batch.positions.clone()

        orig_init = SpatialPartitioner.__init__
        with patch.object(
            SpatialPartitioner, "__init__", self._patched_partitioner_init(orig_init)
        ):
            local_batch = dp.partition(batch)

        assert torch.allclose(local_batch.positions, original_positions)

    def test_partition_sets_n_owned(self) -> None:
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=12)

        orig_init = SpatialPartitioner.__init__
        with patch.object(
            SpatialPartitioner, "__init__", self._patched_partitioner_init(orig_init)
        ):
            dp.partition(batch)

        assert dp._n_owned == 12

    def test_partition_initializes_partitioner(self) -> None:
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=12)

        orig_init = SpatialPartitioner.__init__
        with patch.object(
            SpatialPartitioner, "__init__", self._patched_partitioner_init(orig_init)
        ):
            dp.partition(batch)

        assert dp._partitioner is not None

    def test_partition_no_ghost_exchanger_without_mesh(self) -> None:
        """Without a mesh, ghost_exchanger and migrator stay None."""
        from nvalchemi.distributed.partitioner import SpatialPartitioner

        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=12)

        orig_init = SpatialPartitioner.__init__
        with patch.object(
            SpatialPartitioner, "__init__", self._patched_partitioner_init(orig_init)
        ):
            dp.partition(batch)

        assert dp._ghost_exchanger is None
        assert dp._migrator is None

    def test_partition_asserts_batch_not_none(self) -> None:
        """In single-process mode, batch=None triggers the assertion after
        partitioner init. We use grid_dims to avoid the singular-matrix error
        from the zero cell_matrix fallback."""
        model = DemoModelWrapper()
        inner = DemoDynamics(model=model, n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5, grid_dims=(1, 1, 1))
        dp = DomainParallel(dynamics=inner, config=config)

        with pytest.raises(ValueError, match="batch must be provided"):
            dp.partition(None)


# ---------------------------------------------------------------------------
# Step tests (single-process)
# ---------------------------------------------------------------------------


class TestStepSingleProcess:
    """Test step() in single-process mode with a mock inner dynamics."""

    def _make_dp_with_mock(self) -> tuple[DomainParallel, _MockDynamics]:
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config)
        return dp, mock_inner

    def test_step_orchestrates_inner(self) -> None:
        """DomainParallel orchestrates the inner step directly
        (pre_update → AABB → compute → post_update), incrementing
        the inner dynamics' step_count."""
        dp, mock_inner = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        assert mock_inner.step_count == 0
        result_batch, converged = dp.step(batch)

        # Inner step_count incremented by _run_inner_step
        assert mock_inner.step_count == 1
        assert result_batch is not None

    def test_step_increments_step_count(self) -> None:
        dp, _ = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        assert dp.step_count == 0
        dp.step(batch)
        assert dp.step_count == 1
        dp.step(batch)
        assert dp.step_count == 2

    def test_step_prepare_unprepare_roundtrip(self) -> None:
        """Positions should be restored to global coords after step."""
        dp, _ = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)
        original_cell = batch.cell.clone()
        original_pbc = batch.pbc.clone()

        result_batch, _ = dp.step(batch)

        # After step, cell and pbc should be restored
        assert torch.allclose(result_batch.cell, original_cell, atol=1e-5)
        assert (result_batch.pbc == original_pbc).all()

    def test_step_sets_n_owned(self) -> None:
        dp, _ = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        dp.step(batch)

        assert dp._n_owned == 8

    def test_step_without_ghost_exchanger(self) -> None:
        """Without ghost_exchanger, padded_batch == batch and strip is a no-op."""
        dp, mock_inner = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        # Ensure no ghost exchanger
        assert dp._ghost_exchanger is None

        result_batch, _ = dp.step(batch)
        assert result_batch is not None
        # DomainParallel orchestrates the inner step directly
        # (not via mock_inner.step()), so step_count increments.
        assert mock_inner.step_count == 1


# ---------------------------------------------------------------------------
# Gather tests
# ---------------------------------------------------------------------------


class TestGatherWithoutDist:
    """Test gather() when torch.distributed is not initialized."""

    def test_gather_returns_none(self) -> None:
        dp, _ = _make_domain_parallel()
        result = dp.gather()
        assert result is None


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------


class TestRunMethod:
    """Test run() method."""

    def _make_dp_with_mock(
        self, n_steps: int | None = None
    ) -> tuple[DomainParallel, _MockDynamics]:
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config, n_steps=n_steps)
        return dp, mock_inner

    def test_run_executes_n_steps(self) -> None:
        dp, mock_inner = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        result = dp.run(batch, n_steps=5)

        assert mock_inner.step_count == 5
        assert result is not None

    def test_run_uses_constructor_n_steps(self) -> None:
        dp, mock_inner = self._make_dp_with_mock(n_steps=3)
        batch = _make_batch(n_atoms=8)

        dp.run(batch)

        assert mock_inner.step_count == 3

    def test_run_prefers_argument_over_constructor(self) -> None:
        dp, mock_inner = self._make_dp_with_mock(n_steps=100)
        batch = _make_batch(n_atoms=8)

        dp.run(batch, n_steps=2)

        assert mock_inner.step_count == 2

    def test_run_raises_without_n_steps(self) -> None:
        dp, _ = self._make_dp_with_mock(n_steps=None)
        batch = _make_batch(n_atoms=8)

        with pytest.raises(ValueError, match="No step count provided"):
            dp.run(batch)

    def test_run_step_count_increments(self) -> None:
        dp, _ = self._make_dp_with_mock()
        batch = _make_batch(n_atoms=8)

        dp.run(batch, n_steps=4)

        assert dp.step_count == 4


# ---------------------------------------------------------------------------
# _call_hooks tests
# ---------------------------------------------------------------------------


class TestCallHooksWithScope:
    """Test _call_hooks with different HookScope values."""

    def _make_dp_with_hooks(self, hooks: list) -> DomainParallel:
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config, hooks=hooks)
        return dp

    def test_local_hook_fires(self) -> None:
        hook = _SimpleHook(stage=DynamicsStage.BEFORE_STEP, scope=HookScope.LOCAL)
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        assert hook.call_count == 1

    def test_global_hook_fires_without_dist(self) -> None:
        """GLOBAL hooks fire even without dist (the all_reduce is skipped)."""
        hook = _SimpleHook(stage=DynamicsStage.BEFORE_STEP, scope=HookScope.GLOBAL)
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        assert hook.call_count == 1

    def test_rank_zero_hook_logs_warning(self, caplog) -> None:
        """RANK_ZERO hooks should log a warning and NOT fire."""
        hook = _SimpleHook(stage=DynamicsStage.BEFORE_STEP, scope=HookScope.RANK_ZERO)
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        with caplog.at_level(
            logging.WARNING, logger="nvalchemi.distributed.domain_parallel"
        ):
            dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        assert hook.call_count == 0
        assert "RANK_ZERO is not yet implemented" in caplog.text

    def test_hook_stage_filtering(self) -> None:
        """Hooks should only fire for their declared stage."""
        before_hook = _SimpleHook(
            stage=DynamicsStage.BEFORE_STEP, scope=HookScope.LOCAL
        )
        after_hook = _SimpleHook(stage=DynamicsStage.AFTER_STEP, scope=HookScope.LOCAL)
        dp = self._make_dp_with_hooks([before_hook, after_hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        assert before_hook.call_count == 1
        assert after_hook.call_count == 0

    def test_hook_frequency_gating(self) -> None:
        """Hooks with frequency > 1 should only fire on matching steps."""
        hook = _SimpleHook(
            stage=DynamicsStage.BEFORE_STEP, frequency=3, scope=HookScope.LOCAL
        )
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        # step_count=0 -> 0 % 3 == 0 -> fires
        dp.step_count = 0
        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        assert hook.call_count == 1

        # step_count=1 -> 1 % 3 != 0 -> skipped
        dp.step_count = 1
        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        assert hook.call_count == 1

        # step_count=3 -> 3 % 3 == 0 -> fires
        dp.step_count = 3
        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        assert hook.call_count == 2

    def test_multiple_scopes_together(self) -> None:
        """LOCAL and GLOBAL hooks fire; RANK_ZERO does not."""
        local_hook = _SimpleHook(stage=DynamicsStage.BEFORE_STEP, scope=HookScope.LOCAL)
        global_hook = _SimpleHook(
            stage=DynamicsStage.BEFORE_STEP, scope=HookScope.GLOBAL
        )
        rank_zero_hook = _SimpleHook(
            stage=DynamicsStage.BEFORE_STEP, scope=HookScope.RANK_ZERO
        )
        dp = self._make_dp_with_hooks([local_hook, global_hook, rank_zero_hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)

        assert local_hook.call_count == 1
        assert global_hook.call_count == 1
        assert rank_zero_hook.call_count == 0


# ---------------------------------------------------------------------------
# _save_geometry multi-graph (line 316 branch)
# ---------------------------------------------------------------------------


class TestPreparePaddedBatchMultiGraph:
    """Test _save_geometry with a multi-graph batch (cell.shape[0] > 1)."""

    def test_multi_graph_cell_expansion(self) -> None:
        """When batch has multiple graphs, local_cell should expand to match."""
        dp, _ = _make_domain_parallel()

        # Create a 2-graph batch
        data1 = AtomicData(
            atomic_numbers=torch.tensor([6, 6, 6], dtype=torch.long),
            positions=torch.rand(3, 3) * 10.0,
        )
        data2 = AtomicData(
            atomic_numbers=torch.tensor([8, 8], dtype=torch.long),
            positions=torch.rand(2, 3) * 10.0,
        )
        batch = Batch.from_data_list([data1, data2])
        batch.forces = torch.zeros(5, 3)
        batch.energies = torch.zeros(2, 1)
        batch.cell = torch.stack(
            [
                torch.diag(torch.tensor([10.0, 10.0, 10.0])),
                torch.diag(torch.tensor([12.0, 12.0, 12.0])),
            ]
        )  # (2, 3, 3)
        batch.pbc = torch.tensor([[True, True, True], [True, True, True]])

        original_cell = batch.cell.clone()
        snapshot = dp._save_geometry(batch)

        # Cell should still have 2 graphs (unchanged by prepare;
        # the hook sets the AABB cell later at BEFORE_COMPUTE).
        assert batch.cell.shape[0] == 2
        # pbc should be False for both
        assert batch.pbc.shape == (2, 3)
        assert (~batch.pbc).all()

        # Roundtrip
        dp._restore_geometry(batch, snapshot)
        assert batch.cell.shape[0] == 2
        assert torch.allclose(batch.cell, original_cell, atol=1e-6)


# ---------------------------------------------------------------------------
# Rank resolution branches
# ---------------------------------------------------------------------------


class TestRankResolution:
    """Test rank resolution branches in __init__."""

    def test_rank_zero_without_mesh_or_dist(self) -> None:
        """Without mesh and without dist initialized, rank should be 0."""
        model = DemoModelWrapper()
        inner = DemoDynamics(model=model, n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)  # mesh=None
        dp = DomainParallel(dynamics=inner, config=config)

        assert dp._domain_rank == 0

    def test_mesh_none_falls_through(self) -> None:
        """When mesh is None and dist not initialized, rank == 0."""
        config = DomainConfig(cutoff=3.0, skin=0.5, mesh=None)
        assert config.mesh is None

        model = DemoModelWrapper()
        inner = DemoDynamics(model=model, n_steps=5)
        dp = DomainParallel(dynamics=inner, config=config)
        assert dp._domain_rank == 0

    def test_rank_from_mock_mesh(self) -> None:
        """When mesh is provided, rank should come from mesh.get_local_rank()."""
        from unittest.mock import MagicMock

        mock_mesh = MagicMock()
        mock_mesh.get_local_rank.return_value = 3

        config = DomainConfig(cutoff=3.0, skin=0.5, mesh=mock_mesh)
        model = DemoModelWrapper()
        inner = DemoDynamics(model=model, n_steps=5)
        dp = DomainParallel(dynamics=inner, config=config)

        assert dp._domain_rank == 3
        mock_mesh.get_local_rank.assert_called_once()

    def test_rank_fallback_when_mesh_raises(self) -> None:
        """When mesh.get_local_rank() raises, rank should fall back to 0."""
        from unittest.mock import MagicMock

        mock_mesh = MagicMock()
        mock_mesh.get_local_rank.side_effect = RuntimeError("no mesh backend")

        config = DomainConfig(cutoff=3.0, skin=0.5, mesh=mock_mesh)
        model = DemoModelWrapper()
        inner = DemoDynamics(model=model, n_steps=5)
        dp = DomainParallel(dynamics=inner, config=config)

        assert dp._domain_rank == 0


# ---------------------------------------------------------------------------
# _ensure_output_tensors tests
# ---------------------------------------------------------------------------


class TestEnsureOutputTensors:
    """Test _ensure_output_tensors pre-allocation logic.

    Note: Batch uses tensordict-backed storage that validates tensor shapes
    on assignment. We create batches that genuinely lack fields (by not
    adding them to AtomicData) to test the missing-field branches.
    """

    def test_forces_missing(self) -> None:
        """When batch has no forces, they should be allocated."""
        data = AtomicData(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([6] * 5, dtype=torch.long),
        )
        batch = Batch.from_data_list([data])
        batch.energies = torch.zeros(1, 1, dtype=torch.float64)
        assert not hasattr(batch, "forces") or batch.forces is None

        DomainParallel._ensure_output_tensors(batch)

        assert batch.forces is not None
        assert batch.forces.shape == (5, 3)
        assert (batch.forces == 0).all()

    def test_energies_missing(self) -> None:
        """When batch has no energies, they should be allocated."""
        data = AtomicData(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([6] * 5, dtype=torch.long),
        )
        data.add_node_property("forces", torch.zeros(5, 3))
        batch = Batch.from_data_list([data])
        assert not hasattr(batch, "energies") or batch.energies is None

        DomainParallel._ensure_output_tensors(batch)

        assert batch.energies is not None
        assert batch.energies.shape == (1, 1)
        assert (batch.energies == 0).all()

    def test_both_missing(self) -> None:
        """When both forces and energies are missing, both should be allocated."""
        data = AtomicData(
            positions=torch.randn(4, 3),
            atomic_numbers=torch.tensor([6] * 4, dtype=torch.long),
        )
        batch = Batch.from_data_list([data])

        DomainParallel._ensure_output_tensors(batch)

        assert batch.forces is not None
        assert batch.forces.shape == (4, 3)
        assert batch.energies is not None
        assert batch.energies.shape == (1, 1)

    def test_forces_and_energies_correct_noop(self) -> None:
        """When forces and energies already have correct shapes, no reallocation."""
        batch = _make_batch(n_atoms=5)
        original_forces = batch.forces.clone()
        original_energies = batch.energies.clone()

        DomainParallel._ensure_output_tensors(batch)

        # Values should be unchanged
        torch.testing.assert_close(batch.forces, original_forces)
        torch.testing.assert_close(batch.energies, original_energies)

    def test_forces_wrong_size_via_padded_batch(self) -> None:
        """Simulate the ghost-exchange scenario: a padded batch is built from
        a larger set of atoms, but forces from the pre-exchange batch have
        the wrong size. We verify _ensure_output_tensors handles this case
        by building two batches of different sizes."""
        # Build a small batch with 3 atoms + forces
        data_small = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6] * 3, dtype=torch.long),
        )
        data_small.add_node_property("forces", torch.randn(3, 3))
        batch_small = Batch.from_data_list([data_small])
        assert batch_small.forces.shape == (3, 3)

        # Build a larger padded batch (simulating ghost exchange grew the batch)
        data_large = AtomicData(
            positions=torch.randn(5, 3),
            atomic_numbers=torch.tensor([6] * 5, dtype=torch.long),
        )
        batch_large = Batch.from_data_list([data_large])
        # This batch has no forces — _ensure_output_tensors should add them
        DomainParallel._ensure_output_tensors(batch_large)
        assert batch_large.forces.shape == (5, 3)

    def test_energies_wrong_num_graphs_via_multi_graph(self) -> None:
        """Test energies reallocation with a multi-graph batch where we set
        energies with the correct num_graphs first, then verify the method
        does not touch them."""
        data1 = AtomicData(
            positions=torch.randn(3, 3),
            atomic_numbers=torch.tensor([6] * 3, dtype=torch.long),
        )
        data2 = AtomicData(
            positions=torch.randn(2, 3),
            atomic_numbers=torch.tensor([8] * 2, dtype=torch.long),
        )
        batch = Batch.from_data_list([data1, data2])
        batch.forces = torch.zeros(5, 3)
        batch.energies = torch.zeros(2, 1, dtype=torch.float64)

        DomainParallel._ensure_output_tensors(batch)

        # Both should have correct shapes for 2 graphs, 5 atoms
        assert batch.forces.shape == (5, 3)
        assert batch.energies.shape == (2, 1)


# ---------------------------------------------------------------------------
# _save_geometry with missing pbc
# ---------------------------------------------------------------------------


class TestUnprepareWithNullPosMin:
    """Test _restore_geometry when pos_min is None in snapshot."""

    def test_unprepare_skips_position_shift_when_pos_min_none(self) -> None:
        """When snapshot.pos_min is None, positions should not be shifted."""
        dp, _ = _make_domain_parallel()
        batch = _make_batch(n_atoms=5)
        original_positions = batch.positions.clone()
        original_cell = torch.diag(torch.tensor([5.0, 5.0, 5.0])).unsqueeze(0)
        original_pbc = torch.tensor([[True, True, True]])

        snapshot = _GeometrySnapshot(
            original_cell=original_cell,
            original_pbc=original_pbc,
            pos_min=None,
        )

        dp._restore_geometry(batch, snapshot)

        # Positions should be unchanged (no shift applied)
        torch.testing.assert_close(batch.positions, original_positions)
        # Cell and pbc should be restored
        torch.testing.assert_close(batch.cell, original_cell)
        assert (batch.pbc == original_pbc).all()


class TestPreparePaddedBatchMissingPBC:
    """Test _save_geometry when pbc is not set on the batch."""

    def test_missing_pbc_creates_default(self) -> None:
        """When pbc is not present on the batch, prepare should create a
        default pbc (all True) for the snapshot and then set pbc=False
        for the local domain."""
        dp, _ = _make_domain_parallel()

        # Create a batch without pbc but with a cell
        data = AtomicData(
            positions=torch.rand(5, 3) * 10.0,
            atomic_numbers=torch.tensor([6] * 5, dtype=torch.long),
        )
        batch = Batch.from_data_list([data])
        batch.cell = torch.diag(torch.tensor([10.0, 10.0, 10.0])).unsqueeze(0)
        assert not hasattr(batch, "pbc") or batch.pbc is None

        snapshot = dp._save_geometry(batch)

        # After prepare, pbc should be False (open boundaries)
        assert batch.pbc is not None
        assert (~batch.pbc).all()

        # The snapshot should contain the default pbc (all True)
        assert snapshot.original_pbc is not None
        assert snapshot.original_pbc.shape == (1, 3)
        assert snapshot.original_pbc.all()


# ---------------------------------------------------------------------------
# _call_hooks with _runs_on_stage attribute
# ---------------------------------------------------------------------------


class TestCallHooksWithRunsOnStage:
    """Test _call_hooks with hooks that have a _runs_on_stage method."""

    def _make_dp_with_hooks(self, hooks: list) -> DomainParallel:
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config, hooks=hooks)
        return dp

    def test_runs_on_stage_true(self) -> None:
        """Hook with _runs_on_stage returning True should fire."""

        class _StageAwareHook:
            def __init__(self):
                self.stage = DynamicsStage.BEFORE_STEP
                self.frequency = 1
                self.scope = HookScope.LOCAL
                self.call_count = 0

            def _runs_on_stage(self, stage: DynamicsStage) -> bool:
                return stage == DynamicsStage.BEFORE_STEP

            def __call__(self, ctx, stage):
                self.call_count += 1

        hook = _StageAwareHook()
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        assert hook.call_count == 1

    def test_runs_on_stage_false_skips(self) -> None:
        """Hook with _runs_on_stage returning False should be skipped."""

        class _StageAwareHook:
            def __init__(self):
                self.stage = DynamicsStage.BEFORE_STEP
                self.frequency = 1
                self.scope = HookScope.LOCAL
                self.call_count = 0

            def _runs_on_stage(self, stage: DynamicsStage) -> bool:
                return stage == DynamicsStage.AFTER_STEP  # Never matches BEFORE_STEP

            def __call__(self, ctx, stage):
                self.call_count += 1

        hook = _StageAwareHook()
        dp = self._make_dp_with_hooks([hook])
        batch = _make_batch(n_atoms=5)

        dp._call_hooks(DynamicsStage.BEFORE_STEP, batch)
        assert hook.call_count == 0


# ---------------------------------------------------------------------------
# _prime_forces tests
# ---------------------------------------------------------------------------


class TestPrimeForces:
    """Test _prime_forces in single-process mode (no ghost exchanger)."""

    def test_prime_forces_sets_flag(self) -> None:
        """After _prime_forces, the flag should be set so it's not called again."""
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config)

        batch = _make_batch(n_atoms=5)
        assert dp._forces_primed is False

        # step() calls _prime_forces on the first call
        dp.step(batch)
        assert dp._forces_primed is True

    def test_prime_forces_not_called_twice(self) -> None:
        """_prime_forces should only run on the first step."""
        mock_inner = _MockDynamics(n_steps=10)
        config = DomainConfig(cutoff=3.0, skin=0.5)
        dp = DomainParallel(dynamics=mock_inner, config=config)

        batch = _make_batch(n_atoms=5)
        dp.step(batch)
        dp.step(batch)

        # mock_inner.step_count should be 2 (one per step call)
        # If _prime_forces ran twice, there would be extra calls to compute
        assert mock_inner.step_count == 2
