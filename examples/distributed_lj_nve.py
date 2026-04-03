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

Saves per-step snapshots (positions, forces, velocities, neighbor matrices)
to ``dd_debug_rank{rank}.pt`` for post-hoc analysis. Load with::

    data = torch.load("dd_debug_rank0.pt")
    # data["steps"] is a list of dicts, one per step
    # data["steps"][i]["positions"], ["forces"], ["velocities"], etc.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.models.lj import LennardJonesModelWrapper

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def log(rank: int, msg: str, *args: object) -> None:
    """Log a message with timestamp and rank."""
    ts = time.strftime("%H:%M:%S")
    formatted = msg.format(*args) if args else msg
    print(f"[{ts}][rank {rank}] {formatted}", flush=True)


def compute_ke(batch: Batch) -> float:
    """Compute kinetic energy from velocities: ``0.5 * m * v^2``, summed over atoms."""
    v = getattr(batch, "velocities", None)
    m = getattr(batch, "atomic_masses", None)
    if v is None or m is None:
        return 0.0
    return (0.5 * m.unsqueeze(-1) * v**2).sum().item()


def snapshot(batch: Batch, step: int, extra: dict | None = None) -> dict:
    """Capture a CPU copy of the batch state for debugging.  Returns a dict."""
    d = {
        "step": step,
        "n_atoms": batch.num_nodes,
        "positions": batch.positions.detach().cpu().clone(),
        "forces": batch.forces.detach().cpu().clone()
        if hasattr(batch, "forces") and batch.forces is not None
        else None,
        "velocities": batch.velocities.detach().cpu().clone()
        if hasattr(batch, "velocities") and batch.velocities is not None
        else None,
        "energies": batch.energies.detach().cpu().clone()
        if hasattr(batch, "energies") and batch.energies is not None
        else None,
        "cell": batch.cell.detach().cpu().clone()
        if hasattr(batch, "cell") and batch.cell is not None
        else None,
        "pbc": batch.pbc.detach().cpu().clone()
        if hasattr(batch, "pbc") and batch.pbc is not None
        else None,
        "neighbor_matrix": batch.neighbor_matrix.detach().cpu().clone()
        if hasattr(batch, "neighbor_matrix") and batch.neighbor_matrix is not None
        else None,
        "num_neighbors": batch.num_neighbors.detach().cpu().clone()
        if hasattr(batch, "num_neighbors") and batch.num_neighbors is not None
        else None,
    }
    if extra:
        d.update(extra)
    return d


def create_argon_system(n_atoms_per_side: int = 5, lattice_constant: float = 3.82):
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

    log(rank, "world_size={} device={}", world_size, device)

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

    # ── 4. Dynamics ──
    nl_hook = NeighborListHook(config=neighbor_config, skin=4.25)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])

    # ── 5. Domain config ──
    config = DomainConfig(
        cutoff=neighbor_config.cutoff,
        skin=4.25,
        mesh=mesh,
        mesh_dim="domain",
    )
    dd = DomainParallel(nve, config=config)

    # ── 6. Create system on rank 0 ──
    N_SIDE = 5  # 125 atoms — small for debugging
    if rank == 0:
        data = create_argon_system(n_atoms_per_side=N_SIDE)
        batch = Batch.from_data_list([data], device=device)
        log(rank, "Created {} atoms, box={:.2f} A", batch.num_nodes, N_SIDE * 3.82)
    else:
        batch = None

    dist.barrier()

    # ── 7. Partition ──
    local_batch = dd.partition(batch)
    log(rank, "Partition: {} owned atoms", local_batch.num_nodes)

    # ── 8. Run with per-step snapshots ──
    N_STEPS = 80
    debug_log: list[dict] = []

    # Snapshot after partition (before any steps)
    debug_log.append(snapshot(local_batch, step=-1, extra={"label": "after_partition"}))

    # Prime forces
    local_batch = dd._prime_forces(local_batch)
    dd._forces_primed = True

    pe = local_batch.energies.sum().item() if local_batch.energies is not None else 0
    ke = compute_ke(local_batch)
    log(
        rank,
        "After prime: PE={:.4f} KE={:.4f} E={:.4f} fmax={:.6f} n={}",
        pe,
        ke,
        pe + ke,
        local_batch.forces.norm(dim=-1).max().item()
        if local_batch.forces is not None
        else 0,
        local_batch.num_nodes,
    )
    debug_log.append(snapshot(local_batch, step=0, extra={"label": "after_prime"}))

    for step in range(1, N_STEPS + 1):
        try:
            local_batch, converged = dd.step(local_batch)
        except Exception as e:
            log(rank, "EXCEPTION at step {}: {}", step, e)
            import traceback

            traceback.print_exc()
            debug_log.append(
                snapshot(
                    local_batch,
                    step=step,
                    extra={
                        "label": "exception",
                        "error": str(e),
                    },
                )
            )
            break

        pe = (
            local_batch.energies.sum().item() if local_batch.energies is not None else 0
        )
        ke = compute_ke(local_batch)
        fmax = (
            local_batch.forces.norm(dim=-1).max().item()
            if local_batch.forces is not None
            else 0
        )
        pos_min = local_batch.positions.min(dim=0).values.tolist()
        pos_max = local_batch.positions.max(dim=0).values.tolist()

        log(
            rank,
            "step {:3d} | n={:4d} | PE={:12.4f} KE={:8.4f} E={:12.4f} | "
            "fmax={:10.4f} | Z=[{:.1f},{:.1f}]",
            step,
            local_batch.num_nodes,
            pe,
            ke,
            pe + ke,
            fmax,
            pos_min[2],
            pos_max[2],
        )

        # Save snapshot every step (small system, affordable)
        debug_log.append(
            snapshot(
                local_batch,
                step=step,
                extra={
                    "pe": pe,
                    "ke": ke,
                    "fmax": fmax,
                },
            )
        )

        # Bail out early if energy is clearly exploding
        if abs(pe) > 1e4:
            log(rank, "Energy exploded (PE={:.1f}), stopping.", pe)
            break

    # ── 9. Save debug data ──
    out_path = f"dd_debug_rank{rank}.pt"
    torch.save({"rank": rank, "world_size": world_size, "steps": debug_log}, out_path)
    log(rank, "Saved {} snapshots to {}", len(debug_log), out_path)

    dist.barrier()
    DistributedManager.cleanup()
    log(rank, "Done.")


if __name__ == "__main__":
    main()
