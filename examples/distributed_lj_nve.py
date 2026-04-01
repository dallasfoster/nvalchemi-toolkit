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

    torchrun --nproc_per_node=N examples/distributed_lj_nve.py

Demonstrates:
- DomainParallel wrapping an NVE integrator
- LennardJonesModelWrapper as the force model
- Domain decomposition of a bulk Argon system across N GPUs
- Distributed hook system (LoggingHook via HookScope.GLOBAL)
- Detailed per-rank logging for debugging

All log output is prefixed with [rank X] for easy filtering::

    torchrun --nproc_per_node=2 examples/distributed_lj_nve.py 2>&1 | grep "\\[rank 0\\]"
"""

from __future__ import annotations

import time
from enum import Enum

import torch
import torch.distributed as dist
from torch.distributed import DeviceMesh

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig, HookScope
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.lj import LennardJonesModelWrapper

# ──────────────────────────────────────────────────────────────
# Logging helper
# ──────────────────────────────────────────────────────────────


def log(rank: int, msg: str, *args: object, level: str = "INFO") -> None:
    """Print a log line prefixed with rank, timestamp, and level."""
    ts = time.strftime("%H:%M:%S")
    formatted = msg.format(*args) if args else msg
    print(f"[{ts}][rank {rank}][{level}] {formatted}", flush=True)


def log_tensor_stats(rank: int, name: str, t: torch.Tensor | None) -> None:
    """Log shape, dtype, device, and basic statistics of a tensor."""
    if t is None:
        log(rank, "  {}: None", name)
        return
    if t.numel() == 0:
        log(rank, "  {}: empty (shape={})", name, tuple(t.shape))
        return
    if t.is_floating_point():
        log(
            rank,
            "  {}: shape={} dtype={} device={} min={:.6f} max={:.6f} mean={:.6f}",
            name,
            tuple(t.shape),
            t.dtype,
            t.device,
            t.min().item(),
            t.max().item(),
            t.mean().item(),
        )
    else:
        log(
            rank,
            "  {}: shape={} dtype={} device={} min={} max={}",
            name,
            tuple(t.shape),
            t.dtype,
            t.device,
            t.min().item(),
            t.max().item(),
        )


def log_batch(rank: int, label: str, batch: Batch) -> None:
    """Log all relevant fields of a Batch."""
    log(rank, "--- {} ---", label)
    log(rank, "  num_nodes={} num_graphs={}", batch.num_nodes, batch.num_graphs)
    log_tensor_stats(rank, "positions", batch.positions)
    log_tensor_stats(rank, "velocities", getattr(batch, "velocities", None))
    log_tensor_stats(rank, "forces", getattr(batch, "forces", None))
    log_tensor_stats(rank, "energies", getattr(batch, "energies", None))
    log_tensor_stats(rank, "cell", getattr(batch, "cell", None))
    log_tensor_stats(rank, "pbc", getattr(batch, "pbc", None))
    log_tensor_stats(rank, "atomic_numbers", getattr(batch, "atomic_numbers", None))
    log_tensor_stats(rank, "atomic_masses", getattr(batch, "atomic_masses", None))


# ──────────────────────────────────────────────────────────────
# Custom diagnostic hook — demonstrates HookContext fields
# ──────────────────────────────────────────────────────────────


class DiagnosticHook:
    """Observer hook that logs domain-decomposition diagnostics each step.

    Registered on the DomainParallel wrapper (outer hook) so it fires
    at AFTER_STEP with the owned-only batch.  Demonstrates reading
    HookContext domain fields: is_domain_parallel, n_owned, global_cell.
    """

    stage = DynamicsStage.AFTER_STEP
    frequency: int = 1
    scope = HookScope.LOCAL  # runs per-rank, no communication

    def __init__(self, frequency: int = 1) -> None:
        self.frequency = frequency
        self._initial_n_owned: int | None = None

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        rank = ctx.global_rank
        step = ctx.step_count
        batch = ctx.batch
        n_atoms = batch.positions.shape[0]

        # Track initial atom count for drift detection
        if self._initial_n_owned is None:
            self._initial_n_owned = n_atoms

        # Read domain-parallel context fields
        dd_info = ""
        if ctx.is_domain_parallel:
            dd_info = (
                f" | DD: n_owned={ctx.n_owned}"
                f" global_cell={'set' if ctx.global_cell is not None else 'None'}"
                f" mesh={'set' if ctx.domain_mesh is not None else 'None'}"
            )

        # Energy (may be partial — LOCAL scope means no all-reduce)
        E_local = (
            batch.energies.sum().item()
            if hasattr(batch, "energies") and batch.energies is not None
            else float("nan")
        )

        # Force stats
        if hasattr(batch, "forces") and batch.forces is not None:
            fmax = batch.forces.norm(dim=-1).max().item()
            fmean = batch.forces.norm(dim=-1).mean().item()
        else:
            fmax = fmean = float("nan")

        # Position stats
        pos_min = batch.positions.min(dim=0).values.tolist()
        pos_max = batch.positions.max(dim=0).values.tolist()

        atom_drift = n_atoms - self._initial_n_owned

        log(
            rank,
            "STEP {:4d} | atoms={:6d} (drift={:+d}) | E_local={:12.6f} eV"
            " | fmax={:10.6f} fmean={:10.6f}"
            " | pos_range=[{:.1f},{:.1f},{:.1f}]→[{:.1f},{:.1f},{:.1f}]{}",
            step,
            n_atoms,
            atom_drift,
            E_local,
            fmax,
            fmean,
            *pos_min,
            *pos_max,
            dd_info,
        )


class EnergyAllReduceHook:
    """Global hook that all-reduces energy and reports total system energy.

    Registered with scope=GLOBAL so DomainParallel all-reduces
    batch.energies before this hook fires.  Only rank 0 prints.
    """

    stage = DynamicsStage.AFTER_STEP
    frequency: int = 10
    scope = HookScope.GLOBAL  # triggers all-reduce on energies

    def __init__(self, frequency: int = 10) -> None:
        self.frequency = frequency
        self._E0: float | None = None

    def __call__(self, ctx: HookContext, stage: Enum) -> None:
        if ctx.global_rank != 0:
            return
        batch = ctx.batch
        if not hasattr(batch, "energies") or batch.energies is None:
            return
        E_total = batch.energies.sum().item()
        if self._E0 is None:
            self._E0 = E_total
        drift = abs(E_total - self._E0)
        log(
            ctx.global_rank,
            "  >>> GLOBAL E_total={:.6f} eV | drift from E0={:.2e} eV",
            E_total,
            drift,
        )


# ──────────────────────────────────────────────────────────────
# System creation
# ──────────────────────────────────────────────────────────────


def create_argon_system(
    n_atoms_per_side: int = 10, lattice_constant: float = 3.4
) -> AtomicData:
    """Create a simple cubic Argon system with thermal velocities at 300 K."""
    positions = [
        [ix * lattice_constant, iy * lattice_constant, iz * lattice_constant]
        for ix in range(n_atoms_per_side)
        for iy in range(n_atoms_per_side)
        for iz in range(n_atoms_per_side)
    ]
    positions = torch.tensor(positions, dtype=torch.float32)
    n_atoms = positions.shape[0]

    atomic_numbers = torch.full((n_atoms,), 18, dtype=torch.int32)
    atomic_masses = torch.full((n_atoms,), 39.948, dtype=torch.float32)

    # Maxwell-Boltzmann velocities at 300 K
    kB = 8.617e-5  # eV/K
    T = 300.0
    sigma_v = (kB * T / 39.948) ** 0.5
    velocities = torch.randn(n_atoms, 3) * sigma_v
    velocities -= velocities.mean(dim=0)  # zero COM velocity

    box_length = n_atoms_per_side * lattice_constant
    cell = torch.eye(3, dtype=torch.float32) * box_length
    pbc = torch.ones(3, dtype=torch.bool)

    data = AtomicData(
        positions=positions,
        atomic_numbers=atomic_numbers,
        atomic_masses=atomic_masses,
        cell=cell,
        pbc=pbc,
    )
    data.add_node_property("velocities", velocities)

    return data


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main() -> None:
    # ── 1. Distributed init ──────────────────────────────────
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    log(rank, "=== Distributed LJ NVE Example ===")
    log(rank, "world_size={} local_rank={} device={}", world_size, local_rank, device)
    log(
        rank,
        "CUDA: {} ({})",
        torch.cuda.get_device_name(local_rank),
        f"{torch.cuda.get_device_properties(local_rank).total_mem / 1e9:.1f} GB",
    )
    log(rank, "PyTorch: {} CUDA: {}", torch.__version__, torch.version.cuda)

    # ── 2. DeviceMesh ────────────────────────────────────────
    mesh = DeviceMesh("cuda", list(range(world_size)), mesh_dim_names=("domain",))
    log(rank, "DeviceMesh created: dims={} size={}", mesh.mesh_dim_names, mesh.size())

    # ── 3. Model ─────────────────────────────────────────────
    model = LennardJonesModelWrapper(
        epsilon=0.0104,
        sigma=3.40,
        cutoff=8.5,
    ).to(device)

    neighbor_config = model.model_card.neighbor_config
    log(
        rank,
        "Model: LJ epsilon={} sigma={} cutoff={}",
        0.0104,
        3.40,
        neighbor_config.cutoff,
    )
    log(
        rank,
        "NeighborConfig: format={} half_list={} max_neighbors={}",
        neighbor_config.format,
        neighbor_config.half_list,
        neighbor_config.max_neighbors,
    )

    # ── 4. Dynamics ──────────────────────────────────────────
    nl_hook = NeighborListHook(config=neighbor_config, skin=1.0)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])
    log(rank, "NVE integrator: dt=1.0 fs, skin=1.0 A")

    # ── 5. Domain config ─────────────────────────────────────
    config = DomainConfig(
        cutoff=neighbor_config.cutoff,
        skin=1.0,
        mesh=mesh,
        mesh_dim="domain",
    )
    log(
        rank,
        "DomainConfig: cutoff={} skin={} ghost_width={} mesh_dim={}",
        config.cutoff,
        config.skin,
        config.effective_ghost_width(),
        config.mesh_dim,
    )

    # ── 6. DomainParallel wrapper ────────────────────────────
    dd = DomainParallel(nve, config=config)
    log(
        rank,
        "DomainParallel wrapper created (isinstance BaseDynamics={})",
        isinstance(dd, type(nve).__mro__[1]),
    )

    # ── 7. Register diagnostic hooks on the wrapper ──────────
    diag_hook = DiagnosticHook(frequency=1)
    energy_hook = EnergyAllReduceHook(frequency=10)
    dd.register_hook(diag_hook)
    dd.register_hook(energy_hook)
    log(
        rank,
        "Registered {} outer hooks: DiagnosticHook(LOCAL, freq=1), EnergyAllReduceHook(GLOBAL, freq=10)",
        len(dd.hooks),
    )

    # ── 8. Create system on rank 0 ──────────────────────────
    if rank == 0:
        data = create_argon_system(n_atoms_per_side=10)
        batch = Batch.from_data_list([data], device=device)
        log(rank, "Created Argon system:")
        log_batch(rank, "Full system (rank 0 only)", batch)
    else:
        batch = None
        log(rank, "Waiting for partition from rank 0...")

    dist.barrier()

    # ── 9. Partition ─────────────────────────────────────────
    log(rank, ">>> Partitioning...")
    t0 = time.perf_counter()
    local_batch = dd.partition(batch)
    t_partition = time.perf_counter() - t0

    log(rank, "Partition complete in {:.3f}s", t_partition)
    log_batch(rank, "Local batch after partition", local_batch)

    # Log partitioner details
    p = dd._partitioner
    if p is not None:
        log(
            rank,
            "Partitioner: cells_per_dim={} rank_grid={} world_size={}",
            tuple(p.cells_per_dim.tolist()),
            p.rank_grid,
            p.world_size,
        )
        lo, hi = p.rank_to_cell_bounds(rank)
        log(rank, "  My cell bounds: lo={} hi={}", lo, hi)
        log(rank, "  My neighbor ranks: {}", p.get_neighbor_ranks(rank))

    # Log ghost exchanger details
    ge = dd._ghost_exchanger
    if ge is not None:
        log(
            rank,
            "GhostExchanger: ghost_width={:.2f} A, {} neighbor ranks, {} PBC shifts",
            ge.ghost_width,
            len(ge.neighbor_ranks),
            len(ge._pbc_shifts),
        )
        for (s, r), shift in ge._pbc_shifts.items():
            log(
                rank,
                "  PBC shift ({} → {}): [{:.1f}, {:.1f}, {:.1f}]",
                s,
                r,
                *shift.tolist(),
            )

    # Synchronize atom counts across ranks for validation
    n_local = torch.tensor([local_batch.num_nodes], device=device)
    all_counts = torch.zeros(world_size, dtype=n_local.dtype, device=device)
    dist.all_gather_into_tensor(all_counts, n_local)
    if rank == 0:
        log(
            rank,
            "Atom distribution: {} (total={})",
            all_counts.tolist(),
            all_counts.sum().item(),
        )

    dist.barrier()

    # ── 10. Run NVE ──────────────────────────────────────────
    n_steps = 50
    log(rank, ">>> Running {} NVE steps...", n_steps)
    log(rank, "=" * 80)

    t_start = time.perf_counter()
    step_times = []

    for step in range(n_steps):
        t_step_start = time.perf_counter()

        try:
            local_batch, converged = dd.step(local_batch)
        except Exception as e:
            log(
                rank,
                "EXCEPTION at step {}: {} {}",
                step,
                type(e).__name__,
                e,
                level="ERROR",
            )
            import traceback

            traceback.print_exc()
            break

        t_step = time.perf_counter() - t_step_start
        step_times.append(t_step)

        # Log step timing every 10 steps
        if step % 10 == 0:
            log(rank, "  step {} wall time: {:.4f}s", step, t_step)

    t_total = time.perf_counter() - t_start
    log(rank, "=" * 80)

    # ── 11. Summary ──────────────────────────────────────────
    log(rank, ">>> Simulation complete!")
    log(rank, "  Total wall time: {:.3f}s", t_total)
    if step_times:
        import statistics

        log(
            rank,
            "  Step time: mean={:.4f}s median={:.4f}s min={:.4f}s max={:.4f}s",
            statistics.mean(step_times),
            statistics.median(step_times),
            min(step_times),
            max(step_times),
        )
        # Exclude first step (includes compilation/warmup)
        if len(step_times) > 1:
            log(
                rank,
                "  Step time (excl. first): mean={:.4f}s",
                statistics.mean(step_times[1:]),
            )

    log_batch(rank, "Final local batch", local_batch)

    # Final atom count validation
    n_final = torch.tensor([local_batch.num_nodes], device=device)
    all_final = torch.zeros(world_size, dtype=n_final.dtype, device=device)
    dist.all_gather_into_tensor(all_final, n_final)
    if rank == 0:
        log(
            rank,
            "Final atom distribution: {} (total={})",
            all_final.tolist(),
            all_final.sum().item(),
        )
        initial_total = all_counts.sum().item()
        final_total = all_final.sum().item()
        if initial_total != final_total:
            log(
                rank,
                "WARNING: Atom count changed! {} → {}",
                initial_total,
                final_total,
                level="WARN",
            )
        else:
            log(rank, "Atom count conserved: {}", final_total)

    # ── 12. Cleanup ──────────────────────────────────────────
    dist.barrier()
    dist.destroy_process_group()
    log(rank, "Process group destroyed. Exiting.")


if __name__ == "__main__":
    main()
