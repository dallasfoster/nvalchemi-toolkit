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
"""Validate domain decomposition energy conservation.

Three experiments in one script, launched with torchrun on 2+ GPUs:

    torchrun --nproc_per_node=2 examples/validate_dd_energy.py

**Experiment 1 — Single-GPU Baseline**
    Rank 0 runs 100 steps of single-GPU NVE on the same 1000-atom system
    and reports energy conservation.  This establishes the baseline drift.

**Experiment 2 — Force Comparison After Initial Compute**
    After DD force priming, gathers DD forces to rank 0 and compares them
    against single-GPU forces on the identical configuration.  Any mismatch
    here means the ghost exchange or NL build is producing wrong forces.

**Experiment 3 — DD Energy Tracking via Gather**
    Runs N steps of DD.  Every K steps, gathers the full system to rank 0,
    runs a single-GPU force evaluation, and reports the *true* total energy
    (avoiding the ghost-double-counting problem in per-rank energy sums).
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────


def log(rank: int, msg: str, *args: object) -> None:
    """Log a message with timestamp and rank."""
    ts = time.strftime("%H:%M:%S")
    formatted = msg.format(*args) if args else msg
    print(f"[{ts}][rank {rank}] {formatted}", flush=True)


def create_argon_system(n_side: int = 10, lattice: float = 3.82, seed: int = 42):
    """Create an ``n_side^3`` Argon system at 50 K with deterministic velocities."""
    from nvalchemi.data import AtomicData

    coords = torch.arange(n_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n_atoms = positions.shape[0]

    kB = 8.617333262e-5  # eV/K
    T = 50.0
    M_AR = 39.948
    torch.manual_seed(seed)
    v_scale = (kB * T / M_AR) ** 0.5
    velocities = torch.randn(n_atoms, 3) * v_scale
    velocities -= velocities.mean(dim=0, keepdim=True)

    box_length = n_side * lattice
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.long),
        atomic_masses=torch.full((n_atoms,), M_AR, dtype=torch.float32),
        forces=torch.zeros(n_atoms, 3),
        energies=torch.zeros(1, 1),
        cell=torch.eye(3).unsqueeze(0) * box_length,
        pbc=torch.tensor([[True, True, True]]),
    )
    data.add_node_property("velocities", velocities)
    return data


def make_model(device):
    """Create a Lennard-Jones model."""
    from nvalchemi.models.lj import LennardJonesModelWrapper

    return LennardJonesModelWrapper(
        epsilon=0.0104, sigma=3.40, cutoff=8.5, max_neighbors=64
    ).to(device)


def compute_ke(batch) -> float:
    """Kinetic energy from velocities: ``0.5 * m * v^2``, summed over atoms."""
    v = getattr(batch, "velocities", None)
    m = getattr(batch, "atomic_masses", None)
    if v is None or m is None:
        return 0.0
    return (0.5 * m.unsqueeze(-1) * v**2).sum().item()


def single_gpu_forces(model, batch, device):
    """Run NL build + model forward on a single-GPU batch.  Returns ``(PE, forces)``."""
    # Clone batch so we don't mutate the caller's data.
    from nvalchemi.data import AtomicData, Batch
    from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook

    data = AtomicData(
        positions=batch.positions.clone(),
        atomic_numbers=batch.atomic_numbers.clone(),
        atomic_masses=batch.atomic_masses.clone(),
        forces=torch.zeros_like(batch.positions),
        energies=torch.zeros(1, 1, dtype=torch.float64, device=device),
        cell=batch.cell.clone(),
        pbc=batch.pbc.clone(),
    )
    if hasattr(batch, "velocities") and batch.velocities is not None:
        data.add_node_property("velocities", batch.velocities.clone())
    ref_batch = Batch.from_data_list([data], device=device)

    # Build NL + compute forces
    config = model.model_card.neighbor_config
    nl_hook = NeighborListHook(config=config, skin=0.0)
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.hooks._context import HookContext

    ctx = HookContext(batch=ref_batch, step_count=0, model=model)
    nl_hook(ctx, DynamicsStage.BEFORE_COMPUTE)

    outputs = model(ref_batch)
    if outputs.get("forces") is not None:
        ref_batch.forces.copy_(outputs["forces"])
    if outputs.get("energies") is not None:
        ref_batch.energies.copy_(outputs["energies"].view(ref_batch.energies.shape))

    pe = ref_batch.energies.sum().item()
    forces = ref_batch.forces.clone()
    return pe, forces


# ──────────────────────────────────────────────────────────────
# Experiment 1 — Single-GPU Baseline
# ──────────────────────────────────────────────────────────────


def experiment_1_single_gpu(rank, device):
    """Run single-GPU NVE and report energy conservation (``rank 0`` only)."""
    if rank != 0:
        return

    from nvalchemi.data import Batch
    from nvalchemi.dynamics import NVE
    from nvalchemi.dynamics.hooks import NeighborListHook, WrapPeriodicHook

    log(rank, "=" * 70)
    log(rank, "EXPERIMENT 1: Single-GPU NVE Baseline (1000 atoms)")
    log(rank, "=" * 70)

    model = make_model(device)
    data = create_argon_system(n_side=10)
    batch = Batch.from_data_list([data], device=device)

    config = model.model_card.neighbor_config
    nve = NVE(model=model, dt=1.0, n_steps=100)
    nve.register_hook(NeighborListHook(config, skin=0.5))
    nve.register_hook(WrapPeriodicHook())

    n_steps = 100
    energies = []

    # Prime forces BEFORE the first step (critical for VV energy conservation).
    # BaseDynamics.step() calls pre_update which uses current forces for the
    # first half-kick.  Without priming, forces are zero → wrong first kick.
    nve._ensure_state_initialized(batch)
    nve._call_hooks(DynamicsStage.BEFORE_COMPUTE, batch)
    nve.compute(batch)
    nve._call_hooks(DynamicsStage.AFTER_COMPUTE, batch)

    # Initial energy: use the just-computed PE and current KE.
    pe = batch.energies.sum().item()
    ke = compute_ke(batch)
    energies.append((0, pe, ke, pe + ke))
    log(
        rank,
        "Step {:4d}: PE={:12.6f}  KE={:12.6f}  E_total={:12.6f}",
        0,
        pe,
        ke,
        pe + ke,
    )

    for step in range(1, n_steps + 1):
        batch, _ = nve.step(batch)
        if step % 10 == 0 or step == 1:
            # Read PE directly from the batch (set by compute inside step)
            pe = batch.energies.sum().item()
            ke = compute_ke(batch)
            energies.append((step, pe, ke, pe + ke))
            log(
                rank,
                "Step {:4d}: PE={:12.6f}  KE={:12.6f}  E_total={:12.6f}",
                step,
                pe,
                ke,
                pe + ke,
            )

    E0 = energies[0][3]
    max_drift = max(abs(e[3] - E0) for e in energies)
    drift_per_atom_per_step = max_drift / (1000 * n_steps)
    log(rank, "")
    log(
        rank,
        "RESULT: E0={:.6f}  max |dE|={:.6e} eV  drift/atom/step={:.2e} eV",
        E0,
        max_drift,
        drift_per_atom_per_step,
    )
    log(rank, "")

    return energies


# ──────────────────────────────────────────────────────────────
# Experiment 2 — Force Comparison
# ──────────────────────────────────────────────────────────────


def experiment_2_force_comparison(rank, device, world_size):
    """Compare DD forces after initial compute with single-GPU reference (``rank 0`` only)."""
    from nvalchemi.data import Batch
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.domain_parallel import DomainParallel
    from nvalchemi.dynamics.integrators.nve import NVE

    log(rank, "=" * 70)
    log(rank, "EXPERIMENT 2: Force Comparison (DD vs single-GPU)")
    log(rank, "=" * 70)

    model = make_model(device)
    mesh = dist.device_mesh.init_device_mesh(
        "cuda", [world_size], mesh_dim_names=("domain",)
    )
    config = DomainConfig(cutoff=8.5, skin=4.25, mesh=mesh, mesh_dim="domain")

    nl_hook = NeighborListHook(model.model_card.neighbor_config, skin=4.25)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])
    dd = DomainParallel(nve, config=config)

    # Create system on rank 0
    if rank == 0:
        data = create_argon_system(n_side=10)
        batch = Batch.from_data_list([data], device=device)
    else:
        batch = None

    dist.barrier()
    local_batch = dd.partition(batch)
    log(rank, "Partitioned: {} owned atoms", local_batch.num_nodes)

    # Prime forces (ghost exchange + compute)
    local_batch = dd._prime_forces(local_batch)
    dd._forces_primed = True
    log(
        rank,
        "Forces primed. Local force stats: fmax={:.6f} fmean={:.6f}",
        local_batch.forces.norm(dim=-1).max().item(),
        local_batch.forces.norm(dim=-1).mean().item(),
    )

    # Gather full system to rank 0
    full_batch = dd.gather(local_batch, dst=0)
    dist.barrier()

    if rank == 0 and full_batch is not None:
        log(rank, "Gathered {} atoms to rank 0", full_batch.num_nodes)

        # Run single-GPU force evaluation on the gathered configuration
        ref_pe, ref_forces = single_gpu_forces(model, full_batch, device)

        # DD forces (gathered) — need to sort to match
        dd_forces = full_batch.forces

        # Compare
        force_diff = (dd_forces - ref_forces).norm(dim=-1)
        max_diff = force_diff.max().item()
        mean_diff = force_diff.mean().item()
        rms_diff = force_diff.pow(2).mean().sqrt().item()

        # Also compare magnitudes
        ref_fmag = ref_forces.norm(dim=-1)
        dd_fmag = dd_forces.norm(dim=-1)
        relative_err = force_diff / ref_fmag.clamp(min=1e-10)
        max_rel = relative_err.max().item()
        mean_rel = relative_err.mean().item()

        log(rank, "")
        log(rank, "Force comparison (DD gathered vs single-GPU reference):")
        log(rank, "  Max absolute diff:  {:.6e} eV/A", max_diff)
        log(rank, "  Mean absolute diff: {:.6e} eV/A", mean_diff)
        log(rank, "  RMS diff:           {:.6e} eV/A", rms_diff)
        log(rank, "  Max relative diff:  {:.6e}", max_rel)
        log(rank, "  Mean relative diff: {:.6e}", mean_rel)
        log(rank, "  Ref PE:  {:.6f} eV", ref_pe)
        log(
            rank,
            "  Ref fmax: {:.6f}  fmean: {:.6f}",
            ref_fmag.max().item(),
            ref_fmag.mean().item(),
        )
        log(
            rank,
            "  DD  fmax: {:.6f}  fmean: {:.6f}",
            dd_fmag.max().item(),
            dd_fmag.mean().item(),
        )

        if max_diff < 1e-4:
            log(rank, "  RESULT: PASS — forces match to < 1e-4 eV/A")
        elif max_diff < 1e-2:
            log(
                rank,
                "  RESULT: MARGINAL — forces differ by {:.2e} eV/A (check ghost NL)",
                max_diff,
            )
        else:
            log(rank, "  RESULT: FAIL — forces differ by {:.2e} eV/A", max_diff)

        # Print worst atoms
        worst_idx = force_diff.topk(min(5, force_diff.shape[0])).indices
        log(rank, "  Worst atoms:")
        for i in worst_idx:
            pos = full_batch.positions[i]
            log(
                rank,
                "    atom {:4d}: pos=[{:.2f},{:.2f},{:.2f}] diff={:.6e} ref_f=[{:.4f},{:.4f},{:.4f}] dd_f=[{:.4f},{:.4f},{:.4f}]",
                i.item(),
                pos[0].item(),
                pos[1].item(),
                pos[2].item(),
                force_diff[i].item(),
                ref_forces[i, 0].item(),
                ref_forces[i, 1].item(),
                ref_forces[i, 2].item(),
                dd_forces[i, 0].item(),
                dd_forces[i, 1].item(),
                dd_forces[i, 2].item(),
            )
        log(rank, "")

    dist.barrier()
    return dd, local_batch


# ──────────────────────────────────────────────────────────────
# Experiment 3 — DD Energy Tracking via Gather
# ──────────────────────────────────────────────────────────────


def experiment_3_dd_energy_tracking(rank, device, world_size):
    """Run DD NVE steps, gather every K steps, compute true energy on ``rank 0``."""
    from nvalchemi.data import Batch
    from nvalchemi.distributed.config import DomainConfig
    from nvalchemi.distributed.domain_parallel import DomainParallel
    from nvalchemi.dynamics.integrators.nve import NVE

    log(rank, "=" * 70)
    log(rank, "EXPERIMENT 3: DD Energy Tracking (gather + single-GPU recompute)")
    log(rank, "=" * 70)

    model = make_model(device)
    mesh = dist.device_mesh.init_device_mesh(
        "cuda", [world_size], mesh_dim_names=("domain",)
    )
    config = DomainConfig(cutoff=8.5, skin=4.25, mesh=mesh, mesh_dim="domain")

    nl_hook = NeighborListHook(model.model_card.neighbor_config, skin=4.25)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])
    dd = DomainParallel(nve, config=config)

    # Create system on rank 0
    if rank == 0:
        data = create_argon_system(n_side=10)
        batch = Batch.from_data_list([data], device=device)
    else:
        batch = None

    dist.barrier()
    local_batch = dd.partition(batch)

    # Prime forces
    local_batch = dd._prime_forces(local_batch)
    dd._forces_primed = True

    n_steps = 50
    check_every = 5
    energies = []

    # Initial energy via gather
    full = dd.gather(local_batch, dst=0)
    if rank == 0 and full is not None:
        pe, _ = single_gpu_forces(model, full, device)
        ke = compute_ke(full)
        energies.append((0, pe, ke, pe + ke))
        log(
            rank,
            "Step {:4d}: PE={:12.6f}  KE={:12.6f}  E_total={:12.6f}  (via gather)",
            0,
            pe,
            ke,
            pe + ke,
        )
    dist.barrier()

    for step in range(1, n_steps + 1):
        try:
            local_batch, _ = dd.step(local_batch)
        except Exception as e:
            log(rank, "EXCEPTION at step {}: {}", step, e)
            import traceback

            traceback.print_exc()
            break

        if step % check_every == 0 or step == 1:
            # Gather to rank 0 for true energy
            full = dd.gather(local_batch, dst=0)
            if rank == 0 and full is not None:
                pe, _ = single_gpu_forces(model, full, device)
                ke = compute_ke(full)
                energies.append((step, pe, ke, pe + ke))
                log(
                    rank,
                    "Step {:4d}: PE={:12.6f}  KE={:12.6f}  E_total={:12.6f}  (via gather)",
                    step,
                    pe,
                    ke,
                    pe + ke,
                )

                # Also report per-rank energy (the misleading metric)
                local_pe = (
                    local_batch.energies.sum().item()
                    if hasattr(local_batch, "energies")
                    and local_batch.energies is not None
                    else 0
                )
                local_ke = compute_ke(local_batch)
                log(
                    rank,
                    "          rank0 local: PE={:12.6f}  KE={:12.6f}  sum={:12.6f}  (includes ghost pairs)",
                    local_pe,
                    local_ke,
                    local_pe + local_ke,
                )
            dist.barrier()

    if rank == 0 and energies:
        log(rank, "")
        log(rank, "=" * 70)
        log(rank, "SUMMARY — True Energy Conservation (via gather)")
        log(rank, "=" * 70)
        E0 = energies[0][3]
        log(
            rank,
            "{:>6s}  {:>12s}  {:>12s}  {:>14s}  {:>12s}",
            "step",
            "PE (eV)",
            "KE (eV)",
            "E_total (eV)",
            "dE (eV)",
        )
        log(rank, "-" * 62)
        for step, pe, ke, etot in energies:
            log(
                rank,
                "{:6d}  {:12.6f}  {:12.6f}  {:14.6f}  {:12.6e}",
                step,
                pe,
                ke,
                etot,
                etot - E0,
            )

        max_drift = max(abs(e[3] - E0) for e in energies)
        last_drift = abs(energies[-1][3] - E0)
        n_measured = energies[-1][0]
        drift_per_step = last_drift / max(n_measured, 1)
        log(rank, "")
        log(rank, "E0 = {:.6f} eV", E0)
        log(rank, "Max |dE| = {:.6e} eV", max_drift)
        log(rank, "Final |dE| = {:.6e} eV (over {} steps)", last_drift, n_measured)
        log(
            rank,
            "Drift rate = {:.6e} eV/step = {:.6e} eV/atom/step",
            drift_per_step,
            drift_per_step / 1000,
        )

        if max_drift < 0.01:
            log(
                rank,
                "RESULT: PASS — energy conserved to < 0.01 eV over {} steps",
                n_measured,
            )
        elif max_drift < 1.0:
            log(
                rank,
                "RESULT: MARGINAL — {:.4f} eV drift over {} steps",
                max_drift,
                n_measured,
            )
        else:
            log(
                rank,
                "RESULT: FAIL — {:.2f} eV drift over {} steps",
                max_drift,
                n_measured,
            )


# ──────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────


def main():
    """Main entry point for the validation script."""
    import logging

    logging.basicConfig(
        level=logging.WARNING,  # Suppress verbose DD logging
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )

    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    rank = dm.rank
    world_size = dm.world_size
    device = dm.device

    log(rank, "Validation script started: world_size={} device={}", world_size, device)

    # Experiment 1: single-GPU baseline (rank 0 only)
    experiment_1_single_gpu(rank, device)
    dist.barrier()

    # Experiment 2: force comparison
    experiment_2_force_comparison(rank, device, world_size)
    dist.barrier()

    # Experiment 3: DD energy tracking
    experiment_3_dd_energy_tracking(rank, device, world_size)
    dist.barrier()

    DistributedManager.cleanup()
    log(rank, "Done.")


if __name__ == "__main__":
    main()
