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

Saves two kinds of per-step snapshots to ``dd_debug_rank{rank}.pt``:

- **owned snapshots**: positions/forces/velocities after strip (owned atoms only)
- **padded snapshots**: full padded batch (owned + ghosts) in AABB frame,
  including neighbor_matrix and num_neighbors — captured via a debug callback
  inside DomainParallel, right after compute and before strip.

Analyze with::

    python examples/analyze_dd_debug.py dd_debug_rank0.pt dd_debug_rank1.pt

The neighbor_matrix fill value is ``num_atoms`` (not -1), so invalid neighbor
entries equal the total atom count of the padded batch.
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

N_SIDE = 8  # 512 atoms — large enough to show instabilities
N_STEPS = 80


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


def _cpu(t):
    """Detach, move to CPU, clone — or None."""
    if t is None:
        return None
    return t.detach().cpu().clone()


def snapshot_owned(batch: Batch, step: int, extra: dict | None = None) -> dict:
    """Capture the owned-atom batch state (after strip)."""
    d = {
        "step": step,
        "kind": "owned",
        "n_atoms": batch.num_nodes,
        "positions": _cpu(batch.positions),
        "forces": _cpu(getattr(batch, "forces", None)),
        "velocities": _cpu(getattr(batch, "velocities", None)),
        "energies": _cpu(getattr(batch, "energies", None)),
        "cell": _cpu(getattr(batch, "cell", None)),
        "pbc": _cpu(getattr(batch, "pbc", None)),
    }
    if extra:
        d.update(extra)
    return d


def snapshot_padded(batch: Batch, n_owned: int, step: int) -> dict:
    """Capture the padded batch state (owned + ghosts, in AABB frame, with NL).

    The neighbor_matrix fill value is ``batch.num_nodes`` (= total padded count).
    Valid neighbor entries are in ``[0, num_atoms)``.
    """
    return {
        "step": step,
        "kind": "padded",
        "n_atoms": batch.num_nodes,
        "n_owned": n_owned,
        "n_ghosts": batch.num_nodes - n_owned,
        "positions": _cpu(batch.positions),
        "forces": _cpu(getattr(batch, "forces", None)),
        "energies": _cpu(getattr(batch, "energies", None)),
        "cell": _cpu(getattr(batch, "cell", None)),
        "pbc": _cpu(getattr(batch, "pbc", None)),
        "neighbor_matrix": _cpu(getattr(batch, "neighbor_matrix", None)),
        "num_neighbors": _cpu(getattr(batch, "num_neighbors", None)),
    }


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

    # ── 8. Wire up debug callback to capture padded batch snapshots ──
    debug_owned: list[dict] = []
    debug_padded: list[dict] = []

    def _on_post_compute(padded_batch, n_owned, step):
        debug_padded.append(snapshot_padded(padded_batch, n_owned, step))

    dd._debug_post_compute_fn = _on_post_compute

    # Snapshot after partition (before any steps)
    debug_owned.append(
        snapshot_owned(local_batch, step=-1, extra={"label": "after_partition"})
    )

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
    debug_owned.append(
        snapshot_owned(local_batch, step=0, extra={"label": "after_prime"})
    )

    for step in range(1, N_STEPS + 1):
        try:
            local_batch, converged = dd.step(local_batch)
        except Exception as e:
            log(rank, "EXCEPTION at step {}: {}", step, e)
            import traceback

            traceback.print_exc()
            debug_owned.append(
                snapshot_owned(
                    local_batch,
                    step=step,
                    extra={"label": "exception", "error": str(e)},
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

        debug_owned.append(
            snapshot_owned(
                local_batch, step=step, extra={"pe": pe, "ke": ke, "fmax": fmax}
            )
        )

        # Synchronized bail-out: all ranks must agree to stop, otherwise
        # the next dd.step() will hang on a collective.
        should_stop = torch.tensor(
            [1 if abs(pe) > 1e4 else 0], dtype=torch.int32, device=device
        )
        dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)
        if should_stop.item() > 0:
            log(rank, "Stopping (PE={:.1f}, any-rank explosion).", pe)
            break

    # ── 9. Save debug data ──
    out_path = f"dd_debug_rank{rank}.pt"
    torch.save(
        {
            "rank": rank,
            "world_size": world_size,
            "n_side": N_SIDE,
            "owned": debug_owned,
            "padded": debug_padded,
        },
        out_path,
    )
    log(
        rank,
        "Saved {} owned + {} padded snapshots to {}",
        len(debug_owned),
        len(debug_padded),
        out_path,
    )

    dist.barrier()
    DistributedManager.cleanup()
    log(rank, "Done.")


if __name__ == "__main__":
    main()
