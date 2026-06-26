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
"""Distributed domain-decomposed NVE with a MACE-like GNN model.

Launch with::

    torchrun --nproc_per_node=2 examples/distributed_mace_nve.py

Uses ``DomainParallel.run()`` with ``LoggingHook`` and ``NaNDetectorHook``
to exercise the hook infrastructure with a GNN architecture.

By default uses a ``MockMACEModel`` (no checkpoint required).  Set
``MACE_MODEL_PATH=/path/to/model.pt`` to use a real MACE checkpoint.

Key differences from the LJ example that stress-test domain decomposition:
- COO edge_index format (not MATRIX neighbor_matrix)
- Forces via autograd (requires positions.requires_grad=True)
- unit_shifts + physical shifts for PBC edges
- node_attrs one-hot encoding must handle ghost atom atomic numbers
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed import DomainConfig, DomainParallel, HookScope
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.logging import LoggingHook
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.hooks.safety import NaNDetectorHook
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.models.mace import MACEWrapper

N_SIDE = 5  # 125 atoms — smaller than LJ for faster iteration
N_STEPS = 20

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def create_argon_system(n_side: int = 5, lattice: float = 3.82):
    """Create a simple-cubic Argon system with Maxwell-Boltzmann velocities at 50 K."""
    coords = torch.arange(n_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n = positions.shape[0]

    kB, T, M = 8.617333262e-5, 50.0, 39.948
    torch.manual_seed(42)
    vel = torch.randn(n, 3) * (kB * T / M) ** 0.5
    vel -= vel.mean(0)

    box = n_side * lattice
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n,), 18, dtype=torch.int64),
        atomic_masses=torch.full((n,), M),
        cell=torch.eye(3).unsqueeze(0) * box,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data.add_node_property("velocities", vel)
    return data


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main() -> None:
    """Main entry point."""
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

    # ── 3. Model — MACE uses COO format and autograd forces ──
    MACE_MODEL_PATH = os.environ.get("MACE_MODEL_PATH", "")
    if not MACE_MODEL_PATH:
        raise ValueError("MACE_MODEL_PATH is not set")
    model = MACEWrapper.from_checkpoint(checkpoint_path=MACE_MODEL_PATH, device=device)
    neighbor_config = model.model_card.neighbor_config
    print(
        f"[rank {rank}] MACE neighbor config: cutoff={neighbor_config.cutoff} "
        f"format={neighbor_config.format}",
        flush=True,
    )

    # ── 4. Inner dynamics (NVE) with COO NL hook ──
    nl_hook = NeighborListHook(config=neighbor_config, skin=0.0)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])

    # ── 5. Outer hooks for DomainParallel ──
    log_path = f"dd_mace_log_rank{rank}.csv"
    logging_hook = LoggingHook(
        backend="csv",
        log_path=log_path,
        frequency=1,
        stage=DynamicsStage.AFTER_STEP,
    )
    logging_hook.scope = HookScope.LOCAL

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
        hooks=[logging_hook, nan_hook],
    )

    # ── 7. Create system on rank 0 ──
    if rank == 0:
        data = create_argon_system(n_side=N_SIDE)
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

    # run() handles hook open/close lifecycle automatically.
    final_batch = dd.run(local_batch, n_steps=N_STEPS)

    print(
        f"[rank {rank}] Done. {final_batch.num_nodes} atoms, "
        f"logged {N_STEPS} steps to {log_path}",
        flush=True,
    )

    # ── 9. Gather for validation ──
    full_batch = dd.gather(final_batch, dst=0)
    if rank == 0 and full_batch is not None:
        print(
            f"[rank 0] Gathered {full_batch.num_nodes} atoms, "
            f"total PE={full_batch.energies.sum().item():.4f}"
            if hasattr(full_batch, "energies") and full_batch.energies is not None
            else "total PE=N/A",
            flush=True,
        )

    dist.barrier()
    DistributedManager.cleanup()


if __name__ == "__main__":
    main()
