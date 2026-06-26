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
"""Tests for UMAWrapper (fairchem-core predict-unit wrapper).

Structural tests use a ``_MockPredictUnit`` — they exercise
``adapt_input`` / ``adapt_output`` / forward composition, task-name
validation, and model-config correctness without needing a real
fairchem checkpoint.

Phase 2.5(a) forward-equivalence tests live in
``test_uma_equivalence.py`` and require a downloaded checkpoint.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

pytest.importorskip(
    "fairchem.core", reason="fairchem-core not installed; skipping UMA tests"
)

from fairchem.core.datasets.atomic_data import AtomicData as FCAtomicData  # noqa: E402

from nvalchemi.data import AtomicData, Batch  # noqa: E402
from nvalchemi.models.base import NeighborListFormat  # noqa: E402
from nvalchemi.models.uma import _UMA_TASKS, UMAWrapper  # noqa: E402

# ---------------------------------------------------------------------------
# Mocks
# ---------------------------------------------------------------------------


class _MockInferenceSettings:
    base_precision_dtype = torch.float32
    external_graph_gen = False


class _MockBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.r_max = torch.tensor(6.0)
        self.sph_feature_size = 16  # (lmax+1)² with lmax=3
        self.sphere_channels = 128

    def forward(self, data: FCAtomicData) -> dict:
        n = data.pos.shape[0]
        return {
            "embedding": torch.zeros(
                n,
                self.sph_feature_size,
                self.sphere_channels,
                dtype=data.pos.dtype,
                device=data.pos.device,
            ),
            "batch": data.batch,
        }


class _MockInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _MockBackbone()


class _MockModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.module = _MockInnerModel()


class _MockPredictUnit:
    """Minimal stand-in for ``fairchem.core.units.mlip_unit.MLIPPredictUnit``.

    Records the ``FCAtomicData`` handed to ``predict`` so tests can
    inspect the tensor-native conversion. Returns a synthetic
    energy/forces dict matching the prediction schema.
    """

    def __init__(self, supported_tasks: list[str] | None = None) -> None:
        if supported_tasks is None:
            supported_tasks = list(_UMA_TASKS)
        self.dataset_to_tasks = {t: [] for t in supported_tasks}
        self.inference_settings = _MockInferenceSettings()
        self.model = _MockModelWrapper()
        self.last_data: FCAtomicData | None = None

    def predict(self, data: FCAtomicData, undo_element_references: bool = True) -> dict:
        self.last_data = data
        n_graphs = data.num_graphs
        n_atoms = data.pos.shape[0]
        # Differentiable energy = sum of per-atom position L2 norms,
        # grouped by graph. Lets autograd tests through (forces != 0).
        norms = data.pos.pow(2).sum(dim=-1).clamp(min=1e-8).sqrt()
        energy = torch.zeros(n_graphs, dtype=data.pos.dtype, device=data.pos.device)
        energy.scatter_add_(0, data.batch, norms)
        # MLIPPredictUnit returns keyed by task; we hand back the flat
        # form that adapt_output consumes.
        return {
            "energy": energy,
            "forces": torch.zeros(
                n_atoms, 3, dtype=data.pos.dtype, device=data.pos.device
            ),
        }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_pu() -> _MockPredictUnit:
    return _MockPredictUnit()


@pytest.fixture
def wrapper_omol(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omol")


@pytest.fixture
def wrapper_omat(mock_pu) -> UMAWrapper:
    return UMAWrapper(mock_pu, task_name="omat")


def _make_propane() -> AtomicData:
    """Propane C3H8 — 11 atoms, molecular (no PBC)."""
    positions = torch.tensor(
        [
            [0.0000, 0.0000, 0.0000],
            [1.5260, 0.0000, 0.0000],
            [2.0330, 1.4360, 0.0000],
            [-0.5093, 1.0222, 0.0000],
            [-0.5093, -0.5111, 0.8853],
            [-0.5093, -0.5111, -0.8853],
            [2.0319, -0.5111, 0.8853],
            [2.0319, -0.5111, -0.8853],
            [3.1193, 1.4360, 0.0000],
            [1.6763, 1.9471, 0.8853],
            [1.6763, 1.9471, -0.8853],
        ],
        dtype=torch.float32,
    )
    numbers = torch.tensor([6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1], dtype=torch.long)
    return AtomicData(positions=positions, atomic_numbers=numbers)


def _make_periodic_cu() -> AtomicData:
    """Cubic Cu cell (1 atom per cell, a=3.615 Å) — minimal periodic system."""
    a = 3.615
    positions = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    numbers = torch.tensor([29], dtype=torch.long)
    cell = (torch.eye(3, dtype=torch.float32) * a).unsqueeze(0)
    pbc = torch.tensor([[True, True, True]])
    return AtomicData(positions=positions, atomic_numbers=numbers, cell=cell, pbc=pbc)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_invalid_task_name_raises(self, mock_pu):
        with pytest.raises(ValueError, match="task_name"):
            UMAWrapper(mock_pu, task_name="not_a_task")

    def test_unsupported_checkpoint_task_raises(self):
        # A predict unit that only ships the omol head.
        pu = _MockPredictUnit(supported_tasks=["omol"])
        with pytest.raises(ValueError, match="does not ship"):
            UMAWrapper(pu, task_name="omat")

    def test_task_stored(self, wrapper_omol):
        assert wrapper_omol.task_name == "omol"

    def test_cutoff_from_backbone(self, wrapper_omol):
        assert math.isclose(wrapper_omol.cutoff, 6.0, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


class TestModelConfig:
    def test_omol_active_outputs(self, wrapper_omol):
        active = wrapper_omol.model_config.active_outputs
        assert "energy" in active and "forces" in active
        assert "stress" not in active  # molecular — no stress

    def test_omat_active_outputs(self, wrapper_omat):
        active = wrapper_omat.model_config.active_outputs
        assert active >= {"energy", "forces", "stress"}

    def test_needs_pbc_by_task(self, wrapper_omol, wrapper_omat):
        assert wrapper_omol.model_config.needs_pbc is False
        assert wrapper_omat.model_config.needs_pbc is True

    def test_supports_pbc_always(self, wrapper_omol, wrapper_omat):
        assert wrapper_omol.model_config.supports_pbc is True
        assert wrapper_omat.model_config.supports_pbc is True

    def test_neighbor_config(self, wrapper_omol):
        nc = wrapper_omol.model_config.neighbor_config
        assert nc.cutoff == wrapper_omol.cutoff
        assert nc.format is NeighborListFormat.COO
        assert nc.half_list is False

    def test_autograd_forces(self, wrapper_omol):
        assert "forces" in wrapper_omol.model_config.autograd_outputs
        assert "positions" in wrapper_omol.model_config.autograd_inputs

    def test_autograd_stress_for_periodic(self, wrapper_omat):
        assert "stress" in wrapper_omat.model_config.autograd_outputs


# ---------------------------------------------------------------------------
# adapt_input — tensor-native, GPU-residence, no ASE
# ---------------------------------------------------------------------------


class TestAdaptInput:
    def test_molecular_single_system(self, wrapper_omol):
        batch = Batch.from_data_list([_make_propane()])
        fc = wrapper_omol.adapt_input(batch)

        assert isinstance(fc, FCAtomicData)
        assert fc.pos.shape == (11, 3)
        assert fc.atomic_numbers.shape == (11,)
        assert fc.natoms.tolist() == [11]
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.shape == (1, 3)
        assert fc.pbc.any().item() is False  # omol — no PBC
        assert fc.edge_index.shape == (2, 0)
        assert fc.nedges.tolist() == [0]
        assert fc.charge.tolist() == [0]  # default for omol
        assert fc.spin.tolist() == [0]
        assert fc.dataset == ["omol"]

    def test_periodic_single_system(self, wrapper_omat):
        batch = Batch.from_data_list([_make_periodic_cu()])
        fc = wrapper_omat.adapt_input(batch)

        assert fc.pos.shape == (1, 3)
        assert fc.cell.shape == (1, 3, 3)
        assert fc.pbc.all().item() is True
        assert fc.dataset == ["omat"]

    def test_multi_system_batch(self, wrapper_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        fc = wrapper_omol.adapt_input(batch)

        assert fc.pos.shape == (22, 3)
        assert fc.natoms.tolist() == [11, 11]
        assert fc.batch.tolist() == [0] * 11 + [1] * 11
        assert len(fc.dataset) == 2
        assert fc.dataset == ["omol", "omol"]

    def test_accepts_atomicdata_directly(self, wrapper_omol):
        fc = wrapper_omol.adapt_input(_make_propane())
        assert fc.num_graphs == 1
        assert fc.pos.shape == (11, 3)

    def test_preserves_device(self, mock_pu):
        """adapt_input must keep tensors on data.positions.device."""
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # CPU-only test; device preservation is the invariant.
        fc = w.adapt_input(data)
        assert fc.pos.device == data.positions.device
        assert fc.atomic_numbers.device == data.positions.device
        assert fc.batch.device == data.positions.device

    def test_preserves_gradient_flow(self, wrapper_omol):
        """positions with requires_grad should flow into the FC AtomicData."""
        data = _make_propane()
        data.positions.requires_grad_(True)
        fc = wrapper_omol.adapt_input(data)
        # Tensor-native: pos is the same storage (dtype conversion is
        # identity for matching dtypes), so requires_grad carries.
        assert fc.pos.requires_grad

    def test_target_dtype_cast(self, mock_pu):
        """When predict_unit declares fp32, input fp64 positions cast down."""
        mock_pu.inference_settings.base_precision_dtype = torch.float32
        w = UMAWrapper(mock_pu, task_name="omol")
        data = AtomicData(
            positions=torch.randn(5, 3, dtype=torch.float64),
            atomic_numbers=torch.tensor([6, 1, 1, 1, 1], dtype=torch.long),
        )
        fc = w.adapt_input(data)
        assert fc.pos.dtype == torch.float32
        assert fc.cell.dtype == torch.float32

    def test_passes_explicit_charge_spin(self, mock_pu):
        w = UMAWrapper(mock_pu, task_name="omol")
        data = _make_propane()
        # Batch-level charge/spin should flow through.
        batch = Batch.from_data_list([data])
        batch.charge = torch.tensor([-1], dtype=torch.long)
        batch.spin = torch.tensor([2], dtype=torch.long)
        fc = w.adapt_input(batch)
        assert fc.charge.tolist() == [-1]
        assert fc.spin.tolist() == [2]


# ---------------------------------------------------------------------------
# adapt_output
# ---------------------------------------------------------------------------


class TestAdaptOutput:
    def test_molecular_energy_forces(self, wrapper_omol):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(5, 3),
        }
        out = wrapper_omol.adapt_output(raw)
        assert "energy" in out and "forces" in out
        # Ensure per-system 2D shape.
        assert out["energy"].shape == (1, 1)
        assert "stress" not in out

    def test_energy_already_2d(self, wrapper_omol):
        raw = {
            "energy": torch.tensor([[1.5]]),
            "forces": torch.zeros(5, 3),
        }
        out = wrapper_omol.adapt_output(raw)
        assert out["energy"].shape == (1, 1)

    def test_periodic_stress_shape(self, wrapper_omat):
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 3, 3),
        }
        out = wrapper_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)

    def test_stress_flat_reshape(self, wrapper_omat):
        """fairchem sometimes returns stress flattened to (B, 9)."""
        raw = {
            "energy": torch.tensor([1.5]),
            "forces": torch.zeros(1, 3),
            "stress": torch.zeros(1, 9),
        }
        out = wrapper_omat.adapt_output(raw)
        assert out["stress"].shape == (1, 3, 3)


# ---------------------------------------------------------------------------
# forward — adapt_input → predict → adapt_output composition
# ---------------------------------------------------------------------------


class TestForward:
    def test_composes(self, wrapper_omol):
        batch = Batch.from_data_list([_make_propane()])
        out = wrapper_omol(batch)
        assert "energy" in out and "forces" in out
        assert out["energy"].shape == (1, 1)
        assert out["forces"].shape == (11, 3)

    def test_records_input_at_predict(self, wrapper_omol, mock_pu):
        batch = Batch.from_data_list([_make_propane()])
        wrapper_omol(batch)
        assert mock_pu.last_data is not None
        assert isinstance(mock_pu.last_data, FCAtomicData)
        assert mock_pu.last_data.pos.shape == (11, 3)

    def test_batched(self, wrapper_omol):
        batch = Batch.from_data_list([_make_propane(), _make_propane()])
        out = wrapper_omol(batch)
        assert out["energy"].shape == (2, 1)
        assert out["forces"].shape == (22, 3)


# ---------------------------------------------------------------------------
# distribution spec
# ---------------------------------------------------------------------------


class TestMLIPSpec:
    def test_inherits_uma_storage_modes(self, wrapper_omol):
        """Spec carries the halo storage policy (default modes).

        The per-layer edge→node halo correction is handled by the
        :class:`ScatterOutputs` ``OpAdapter`` on the fused Triton kernel
        (see :meth:`test_edge_to_node_ops_scatter_corrected`), NOT by a
        ``scatter_mode`` override — so the policy keeps the preset's default
        ``halo_correction`` / ``halo_read`` modes. (The old ``scatter="local"``
        override belonged to the retired ``gp_utils``/replicated design.)
        """
        from nvalchemi.distributed._core.storage_policy import HaloStoragePolicy

        spec = wrapper_omol.distribution_spec
        policy = spec.distribution.policy
        assert isinstance(policy, HaloStoragePolicy)
        assert policy.gather_mode == "halo_read"
        assert spec.system_reductions is True

    def test_registers_five_triton_ops(self, wrapper_omol):
        """All five ``torch.ops.fairchem._kernel_*`` ops appear in custom_ops."""
        spec = wrapper_omol.distribution_spec
        assert len(spec.distribution.custom_ops) == 5
        names = {str(op.op).split(".")[-1] for op in spec.distribution.custom_ops}
        expected = {
            "default",  # all 5 are .default overloads — so just check count
        }
        assert expected.issubset(names)

    def test_edge_to_node_ops_scatter_corrected(self, wrapper_omol):
        """The two node-shaped edge→node kernels (forward + its node-shaped
        adjoint) carry ``ScatterOutputs`` for per-layer halo correction; the
        other three (node→edge + edge/weight-shaped adjoints) are pass-through.

        None gather inputs: the ``x`` input arrives padded (owned + halo), so
        a ``gather_inputs`` would double-pad it (the bug the halo-dispatch fix
        removed). Correction happens on the OUTPUT via ``ScatterOutputs``.
        """
        spec = wrapper_omol.distribution_spec
        scatter_corrected = [
            op for op in spec.distribution.custom_ops if op.scatter_outputs == (0,)
        ]
        passthrough = [
            op for op in spec.distribution.custom_ops if op.scatter_outputs == ()
        ]
        assert len(scatter_corrected) == 2, (
            f"expected 2 ScatterOutputs ops; got "
            f"{[str(op.op) for op in scatter_corrected]}"
        )
        assert len(passthrough) == 3
        # The forward edge→node kernel is the certain ScatterOutputs case.
        assert any(
            "permute_wigner_inv_edge_to_node.default" in str(op.op)
            for op in scatter_corrected
        )
        # No op gathers inputs (padded x in, output-side correction only).
        for op_spec in spec.distribution.custom_ops:
            assert op_spec.gather_inputs == (), (
                f"unexpected gather_inputs on {op_spec.op}: {op_spec.gather_inputs}"
            )
