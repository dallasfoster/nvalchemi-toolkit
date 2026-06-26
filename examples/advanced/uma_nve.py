#!/usr/bin/env python
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
"""UMA (fairchem-core) NVE / NVT example with energy-drift tracking.

Runs a short molecular-dynamics trajectory driven by the fairchem UMA
foundation model through nvalchemi's velocity-Verlet (NVE) or BAOAB
Langevin (NVT) integrator, and logs kinetic / potential / total energy
plus instantaneous temperature and total-energy drift each
``--log-every`` steps.

This is the single-GPU companion to ``test/models/test_uma_nve_stability.py``
(which asserts < 1 meV/atom drift on bcc Fe 2x2x2 at 300 K over 1000 steps
at 0.5 fs via ``UMAWrapper`` + ``NVE``). The equivalence test
``test/models/test_uma_equivalence.py`` verifies 1e-4 match against
fairchem's ``FAIRChemCalculator`` on OMol propane and OMat bcc Fe.

Phase 3 distributed dispatch is a separate example; this file is
deliberately single-process.

Usage
-----
::

    # HF gated repo — set token once per shell
    export HF_TOKEN=hf_xxx

    # Default: bcc Fe 2x2x2, OMat head, NVE at 300 K for 1000 steps @ 0.5 fs
    python examples/advanced/uma_nve.py

    # Propane molecular NVE (forces OMol head)
    python examples/advanced/uma_nve.py --system propane --task omol --ensemble nve

    # NVT Langevin at 500 K
    python examples/advanced/uma_nve.py --ensemble nvt --temperature-k 500
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch
from ase import Atoms
from ase.build import bulk
from ase.data import atomic_masses as ASE_ATOMIC_MASSES

from nvalchemi.data import AtomicData, Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks._utils import KB_EV, kinetic_energy_per_graph
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.models.uma import UMAWrapper

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PROPANE_POSITIONS = np.array(
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
    ]
)
_PROPANE_NUMBERS = [6, 6, 6, 1, 1, 1, 1, 1, 1, 1, 1]

_DRIFT_GATE_EV_PER_ATOM = 1e-3  # 1 meV/atom — the NVE stability target


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run UMA under nvalchemi NVE / NVT with drift tracking.",
    )
    parser.add_argument("--checkpoint", default="uma-s-1p1")
    parser.add_argument(
        "--task",
        default="omat",
        choices=["omol", "omat", "oc20", "odac", "omc"],
    )
    parser.add_argument("--ensemble", default="nve", choices=["nve", "nvt"])
    parser.add_argument("--n-steps", type=int, default=1000)
    parser.add_argument("--dt-fs", type=float, default=0.5)
    parser.add_argument("--temperature-k", type=float, default=300.0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--system",
        default="bcc-fe",
        choices=["bcc-fe", "propane", "diamond"],
    )
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--friction",
        type=float,
        default=0.01,
        help="NVT Langevin friction in 1/fs (ignored for --ensemble nve).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Structure builders
# ---------------------------------------------------------------------------


def build_ase_atoms(system: str) -> Atoms:
    """Return an ASE ``Atoms`` object for the requested benchmark system."""
    if system == "bcc-fe":
        return bulk("Fe", "bcc", a=2.87, cubic=True) * (2, 2, 2)
    if system == "diamond":
        return bulk("C", "diamond", a=3.567, cubic=True) * (2, 2, 2)
    if system == "propane":
        atoms = Atoms(
            numbers=_PROPANE_NUMBERS,
            positions=_PROPANE_POSITIONS,
            pbc=False,
        )
        atoms.info["charge"] = 0
        atoms.info["spin"] = 1
        return atoms
    raise ValueError(f"Unknown system {system!r}")


def atoms_to_batch(atoms: Atoms, temperature_k: float, device: str, seed: int) -> Batch:
    """Convert ASE ``Atoms`` to an ``nvalchemi.Batch`` with MB velocities.

    Positions/masses/forces stay in float32; atomic numbers are int64.
    Periodic systems include ``cell`` and ``pbc``. Velocities are sampled
    from a Maxwell-Boltzmann distribution at ``temperature_k`` and net
    momentum is removed.
    """
    n = len(atoms)
    pos = torch.as_tensor(
        np.asarray(atoms.positions), dtype=torch.float32, device=device
    )
    numbers_np = np.asarray(atoms.get_atomic_numbers())
    numbers = torch.as_tensor(numbers_np, dtype=torch.long, device=device)
    masses_np = ASE_ATOMIC_MASSES[numbers_np]
    masses = torch.as_tensor(masses_np, dtype=torch.float32, device=device)

    # Maxwell-Boltzmann velocities, zero net momentum.
    g = torch.Generator(device="cpu").manual_seed(seed)
    unit_vel = torch.randn(n, 3, generator=g)
    # Per-atom sigma = sqrt(kB T / m) in Å/√(amu·eV) units consistent with
    # nvalchemi's internal kinetic-energy convention (matches the NVE test).
    sigma = torch.sqrt(torch.as_tensor(KB_EV * temperature_k) / masses.cpu())
    vel = (unit_vel * sigma.unsqueeze(-1)).to(device)
    vel -= vel.mean(dim=0, keepdim=True)

    kwargs: dict[str, Any] = {
        "positions": pos,
        "atomic_numbers": numbers,
        "atomic_masses": masses,
        "velocities": vel,
        "forces": torch.zeros_like(pos),
        "energy": torch.zeros(1, 1, device=device, dtype=torch.float32),
    }
    if bool(np.any(atoms.pbc)):
        kwargs["cell"] = torch.as_tensor(
            np.asarray(atoms.cell.array), dtype=torch.float32, device=device
        ).unsqueeze(0)
        kwargs["pbc"] = torch.as_tensor(
            np.asarray(atoms.pbc), dtype=torch.bool, device=device
        ).reshape(1, 3)

    data = AtomicData(**kwargs)
    return Batch.from_data_list([data])


# ---------------------------------------------------------------------------
# Logging hook
# ---------------------------------------------------------------------------


def make_energy_hook(
    n_atoms: int, log_every: int, ensemble: str, state: dict[str, Any]
):
    """Return an AFTER_STEP hook that prints energies + drift every N steps.

    ``state`` is a mutable dict used to carry ``e0`` (the reference total
    energy captured at the first logged step) across invocations.
    """
    header = f"{'step':>6} {'PE(eV)':>14} {'KE(eV)':>12} {'E_total(eV)':>14} {'T(K)':>8} {'drift(meV/atom)':>18}"
    print(header)
    print("-" * len(header))

    def _hook(ctx, stage):  # noqa: ARG001 — stage unused, required by API
        step = ctx.step_count
        if step % log_every != 0 and step != 1:
            return
        batch = ctx.batch
        pe = float(batch.energy.squeeze(-1).sum().item())
        ke_tensor = kinetic_energy_per_graph(
            batch.velocities,
            batch.atomic_masses,
            batch.batch_idx,
            batch.num_graphs,
        )
        ke = float(ke_tensor.squeeze(-1).sum().item())
        total = pe + ke
        t_inst = (2.0 * ke) / (3.0 * n_atoms * KB_EV)

        if state.get("e0") is None:
            state["e0"] = total
            drift = 0.0
        else:
            drift = (total - state["e0"]) * 1e3 / n_atoms  # meV/atom

        state["last_total"] = total
        state["last_temp"] = t_inst
        print(
            f"{step:6d} {pe:14.6f} {ke:12.6f} {total:14.6f} {t_inst:8.2f} {drift:18.4f}"
        )

    _hook.stage = DynamicsStage.AFTER_STEP
    _hook.frequency = 1  # we filter internally so we capture step 1 too
    return _hook


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    args = parse_args()

    # Molecular systems require the OMol head — override silently if the
    # user picked a periodic task with a molecular system.
    if args.system == "propane" and args.task != "omol":
        print(f"[info] forcing --task omol for system={args.system!r}")
        args.task = "omol"

    device = torch.device(args.device)
    print(
        f"UMA {args.ensemble.upper()} | system={args.system} | task={args.task} | "
        f"checkpoint={args.checkpoint} | device={device} | "
        f"n_steps={args.n_steps} | dt={args.dt_fs} fs | T={args.temperature_k} K"
    )

    # ── Build system ──────────────────────────────────────────────────────
    atoms = build_ase_atoms(args.system)
    batch = atoms_to_batch(atoms, args.temperature_k, str(device), args.seed)
    n_atoms = int(batch.num_nodes)
    print(
        f"System: {n_atoms} atoms, pbc={bool(torch.any(batch.pbc)) if batch.pbc is not None else False}"
    )

    # ── Load UMA ──────────────────────────────────────────────────────────
    load_t0 = time.perf_counter()
    model = UMAWrapper.from_checkpoint(
        args.checkpoint, task_name=args.task, device=str(device)
    )
    load_s = time.perf_counter() - load_t0
    print(f"Loaded UMAWrapper in {load_s:.1f}s (cutoff={model.cutoff:.2f} Å)")

    # ── Integrator ────────────────────────────────────────────────────────
    if args.ensemble == "nve":
        integrator = NVE(model=model, dt=args.dt_fs)
    else:
        integrator = NVTLangevin(
            model=model,
            dt=args.dt_fs,
            temperature=args.temperature_k,
            friction=args.friction,
            random_seed=args.seed,
        )

    # ── Hook ──────────────────────────────────────────────────────────────
    state: dict[str, Any] = {"e0": None, "last_total": None, "last_temp": None}
    integrator.register_hook(
        make_energy_hook(n_atoms, args.log_every, args.ensemble, state)
    )

    # ── Run ───────────────────────────────────────────────────────────────
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    batch = integrator.run(batch, n_steps=args.n_steps)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall_s = time.perf_counter() - t0

    # ── Summary ───────────────────────────────────────────────────────────
    mean_step_ms = wall_s * 1e3 / max(1, args.n_steps)
    final_drift_mev = (
        (state["last_total"] - state["e0"]) * 1e3 / n_atoms
        if state["e0"] is not None and state["last_total"] is not None
        else float("nan")
    )
    print()
    print("-" * 72)
    print(f"wall time     : {wall_s:.2f} s ({mean_step_ms:.2f} ms/step)")
    print(
        f"final T       : {state['last_temp']:.2f} K"
        if state["last_temp"]
        else "final T       : n/a"
    )
    print(f"final drift   : {final_drift_mev:.4f} meV/atom")
    if args.ensemble == "nve":
        passed = abs(final_drift_mev) < _DRIFT_GATE_EV_PER_ATOM * 1e3
        status = "PASS" if passed else "FAIL"
        print(
            f"NVE drift gate: |drift| < 1 meV/atom  →  {status} "
            f"({abs(final_drift_mev):.4f} meV/atom)"
        )
    else:
        print("drift gate    : not applicable for NVT (thermostat absorbs drift)")


if __name__ == "__main__":
    main()
