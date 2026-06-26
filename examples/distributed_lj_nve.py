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
"""Distributed domain-decomposed NVE with Lennard-Jones potential.

Launch with::

    torchrun --nproc_per_node=2 examples/distributed_lj_nve.py

Uses ``DomainParallel.run()`` with ``LoggingHook`` and ``NaNDetectorHook``
registered as outer hooks to exercise the full hook infrastructure.

The ``LoggingHook`` logs per-step energy, temperature, and fmax to a
rank-specific CSV file.  The hook fires at ``AFTER_STEP`` with ``LOCAL``
scope, so each rank logs its own subdomain state — no cross-rank
communication beyond what ``DomainParallel.step()`` already does.
"""

from __future__ import annotations

import torch
import torch.distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DomainConfig, DomainParallel, HookScope
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.logging import LoggingHook
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.hooks.safety import NaNDetectorHook
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.models.lj import LennardJonesModelWrapper

N_SIDE = 8  # 512 atoms
N_STEPS = 80


# ──────────────────────────────────────────────────────────────
# System setup
# ──────────────────────────────────────────────────────────────


def create_argon_system(n_atoms_per_side: int = 8, lattice_constant: float = 3.82):
    """Create a simple-cubic Argon system with Maxwell-Boltzmann velocities at 50 K."""
    coords = torch.arange(n_atoms_per_side, dtype=torch.float32) * lattice_constant
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n_atoms = positions.shape[0]

    kB = 8.617333262e-5  # eV/K
    T = 50.0
    M_AR = 39.948
    torch.manual_seed(42)
    sigma_v = (kB * T / M_AR) ** 0.5
    velocities = torch.randn(n_atoms, 3) * sigma_v
    velocities -= velocities.mean(dim=0)

    box_length = n_atoms_per_side * lattice_constant
    cell = torch.eye(3, dtype=torch.float32).unsqueeze(0) * box_length
    pbc = torch.ones(1, 3, dtype=torch.bool)

    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.int64),
        atomic_masses=torch.full((n_atoms,), M_AR, dtype=torch.float32),
        cell=cell,
        pbc=pbc,
    )
    data.add_node_property("velocities", velocities)
    return data


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point for the distributed LJ NVE example."""
    import logging

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # ── 1. Distributed init ──
    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    rank = dm.rank
    world_size = dm.world_size
    device = dm.device

    print(f"[rank {rank}] world_size={world_size} device={device}", flush=True)

    # ── 2. DeviceMesh ──
    mesh = dist.device_mesh.init_device_mesh(
        "cuda", [world_size], mesh_dim_names=("domain",)
    )

    # ── 3. Model ──
    model = LennardJonesModelWrapper(
        epsilon=0.0104,
        sigma=3.40,
        cutoff=8.5,
    ).to(device)
    neighbor_config = model.model_card.neighbor_config

    # ── 4. Inner dynamics (NVE) with NL hook ──
    # skin=0 forces NL rebuild every step.
    nl_hook = NeighborListHook(config=neighbor_config, skin=0.0)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])

    # ── 5. Outer hooks for DomainParallel ──
    # LoggingHook: logs energy, temperature, fmax to CSV (LOCAL scope).
    log_path = f"dd_log_rank{rank}.csv"
    logging_hook = LoggingHook(
        backend="csv",
        log_path=log_path,
        frequency=1,
        stage=DynamicsStage.AFTER_STEP,
    )
    logging_hook.scope = HookScope.LOCAL

    # Print hook: logs to stdout so we can see progress.
    def _print_hook(ctx, stage):
        b = ctx.batch
        pe = (
            b.energies.sum().item()
            if hasattr(b, "energies") and b.energies is not None
            else 0
        )
        fmax = (
            b.forces.norm(dim=-1).max().item()
            if hasattr(b, "forces") and b.forces is not None
            else 0
        )
        v = getattr(b, "velocities", None)
        m = getattr(b, "atomic_masses", None)
        ke = (
            (0.5 * m.unsqueeze(-1) * v**2).sum().item()
            if v is not None and m is not None
            else 0
        )
        print(
            f"[rank {rank}] step {ctx.step_count:3d} | n={b.num_nodes:4d} | "
            f"PE={pe:12.4f} KE={ke:8.4f} E={pe + ke:12.4f} | fmax={fmax:.6f}",
            flush=True,
        )

    _print_hook.stage = DynamicsStage.AFTER_STEP
    _print_hook.frequency = 5  # print every 5 steps
    _print_hook.scope = HookScope.LOCAL

    # NaN detector: bail out if forces blow up.
    nan_hook = NaNDetectorHook(stage=DynamicsStage.AFTER_STEP)
    nan_hook.scope = HookScope.LOCAL

    # ── 6. Domain-parallel wrapper ──
    config = DomainConfig(
        cutoff=neighbor_config.cutoff,
        skin=0.0,
        mesh=mesh,
        mesh_dim="domain",
    )
    dd = DomainParallel(
        nve,
        config=config,
        n_steps=N_STEPS,
        hooks=[logging_hook, _print_hook, nan_hook],
    )

    # ── 7. Create system on rank 0 ──
    if rank == 0:
        data = create_argon_system(n_atoms_per_side=N_SIDE)
        batch = Batch.from_data_list([data], device=device)
        print(
            f"[rank {rank}] Created {batch.num_nodes} atoms, box={N_SIDE * 3.82:.2f} A",
            flush=True,
        )
    else:
        batch = None

    dist.barrier()

    # ── 8. Partition + run ──
    local_batch = dd.partition(batch)
    print(f"[rank {rank}] Partition: {local_batch.num_nodes} owned atoms", flush=True)

    # DomainParallel.run() handles force priming, step loop, and hook
    # open/close lifecycle (calls __enter__/__exit__ on hooks automatically).
    final_batch = dd.run(local_batch, n_steps=N_STEPS)

    print(
        f"[rank {rank}] Done. {final_batch.num_nodes} atoms, "
        f"logged {N_STEPS} steps to {log_path}",
        flush=True,
    )

    # ── 9. Optional: gather full batch on rank 0 for validation ──
    full_batch = dd.gather(final_batch, dst=0)
    if rank == 0 and full_batch is not None:
        total_pe = (
            full_batch.energies.sum().item()
            if hasattr(full_batch, "energies") and full_batch.energies is not None
            else 0
        )
        print(
            f"[rank 0] Gathered {full_batch.num_nodes} atoms, total PE={total_pe:.4f}",
            flush=True,
        )

    dist.barrier()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
