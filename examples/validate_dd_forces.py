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
"""Compare DD forces against single-GPU reference on the same configuration.

Launch with::

    torchrun --nproc_per_node=2 examples/validate_dd_forces.py

Flow:
1. Create 512-atom Argon system on rank 0.
2. Partition across ranks via DomainParallel.
3. Prime forces (ghost exchange + NL build + model forward).
4. Gather the DD result (positions + forces) to rank 0.
5. On rank 0, run a single-GPU NL build + model forward on the
   gathered positions with the original periodic cell.
6. Compare DD forces vs single-GPU forces atom-by-atom.
"""

from __future__ import annotations

import time

import torch
import torch.distributed as dist

from nvalchemi.data import AtomicData, Batch
from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.domain_parallel import DomainParallel
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.hooks.neighbor_list import NeighborListHook
from nvalchemi.dynamics.integrators.nve import NVE
from nvalchemi.hooks._context import HookContext
from nvalchemi.models.lj import LennardJonesModelWrapper


def log(rank: int, msg: str, *args: object) -> None:
    """Log a message with timestamp and rank."""
    ts = time.strftime("%H:%M:%S")
    formatted = msg.format(*args) if args else msg
    print(f"[{ts}][rank {rank}] {formatted}", flush=True)


def create_argon_system(n_side: int = 8, lattice: float = 3.82):
    """Create a simple-cubic Argon system with Maxwell-Boltzmann velocities at 50 K."""
    coords = torch.arange(n_side, dtype=torch.float32) * lattice
    gx, gy, gz = torch.meshgrid(coords, coords, coords, indexing="ij")
    positions = torch.stack([gx.flatten(), gy.flatten(), gz.flatten()], dim=-1)
    n_atoms = positions.shape[0]
    kB = 8.617333262e-5
    T = 50.0
    M_AR = 39.948
    torch.manual_seed(42)
    velocities = torch.randn(n_atoms, 3) * (kB * T / M_AR) ** 0.5
    velocities -= velocities.mean(dim=0)
    box_length = n_side * lattice
    data = AtomicData(
        positions=positions,
        atomic_numbers=torch.full((n_atoms,), 18, dtype=torch.int64),
        atomic_masses=torch.full((n_atoms,), M_AR, dtype=torch.float32),
        cell=torch.eye(3).unsqueeze(0) * box_length,
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data.add_node_property("velocities", velocities)
    return data


def main() -> None:
    """Main entry point for the validation script."""
    import logging

    logging.basicConfig(level=logging.WARNING)

    from physicsnemo.distributed import DistributedManager

    DistributedManager.initialize()
    dm = DistributedManager()
    rank = dm.rank
    world_size = dm.world_size
    device = dm.device

    log(rank, "world_size={} device={}", world_size, device)

    # ── Setup ──
    mesh = dist.device_mesh.init_device_mesh(
        "cuda", [world_size], mesh_dim_names=("domain",)
    )
    model = LennardJonesModelWrapper(epsilon=0.0104, sigma=3.40, cutoff=8.5).to(device)
    neighbor_config = model.model_card.neighbor_config

    nl_hook = NeighborListHook(config=neighbor_config, skin=4.25)
    nve = NVE(model=model, dt=1.0, hooks=[nl_hook])
    config = DomainConfig(
        cutoff=neighbor_config.cutoff, skin=4.25, mesh=mesh, mesh_dim="domain"
    )
    dd = DomainParallel(nve, config=config)

    # ── Create system and run a few single-GPU steps to perturb positions ──
    # (A perfect lattice has F=0 everywhere, making force comparison useless.)
    if rank == 0:
        from nvalchemi.dynamics import NVE as NVE_ref
        from nvalchemi.dynamics.hooks import WrapPeriodicHook

        data = create_argon_system(n_side=8)
        n = data.positions.shape[0]
        data.add_node_property("forces", torch.zeros(n, 3))
        data.energies = torch.zeros(1, 1)
        batch = Batch.from_data_list([data], device=device)

        ref_nve = NVE_ref(model=model, dt=1.0, n_steps=10)
        ref_nve.register_hook(NeighborListHook(config=neighbor_config, skin=0.5))
        ref_nve.register_hook(WrapPeriodicHook())
        batch = ref_nve.run(batch)

        log(
            rank,
            "Created {} atoms, ran 10 single-GPU NVE steps to perturb. fmax={:.6f}",
            batch.num_nodes,
            batch.forces.norm(dim=-1).max().item(),
        )
    else:
        batch = None

    dist.barrier()
    local_batch = dd.partition(batch)
    log(rank, "Partition: {} owned atoms", local_batch.num_nodes)

    # ── Prime forces (ghost exchange + compute) ──
    local_batch = dd._prime_forces(local_batch)
    dd._forces_primed = True
    log(
        rank,
        "Forces primed: fmax={:.6f} fmean={:.6f}",
        local_batch.forces.norm(dim=-1).max().item(),
        local_batch.forces.norm(dim=-1).mean().item(),
    )

    # ── Gather to rank 0 ──
    full_batch = dd.gather(local_batch, dst=0)
    dist.barrier()

    # ── Compare on rank 0 ──
    if rank == 0 and full_batch is not None:
        n_atoms = full_batch.num_nodes
        log(rank, "Gathered {} atoms to rank 0", n_atoms)

        dd_positions = full_batch.positions.clone()
        dd_forces = full_batch.forces.clone()

        # Build a clean single-GPU batch with the SAME positions
        # but the ORIGINAL periodic cell and pbc=True.
        ref_data = AtomicData(
            positions=dd_positions.clone(),
            atomic_numbers=full_batch.atomic_numbers.clone(),
            atomic_masses=full_batch.atomic_masses.clone(),
            forces=torch.zeros_like(dd_positions),
            energies=torch.zeros(1, 1, dtype=torch.float64, device=device),
            cell=batch.cell.clone(),  # original periodic cell
            pbc=batch.pbc.clone(),  # [True, True, True]
        )
        ref_batch = Batch.from_data_list([ref_data], device=device)

        # Single-GPU NL build + force compute
        ref_nl = NeighborListHook(config=neighbor_config, skin=0.0)
        ctx = HookContext(batch=ref_batch, step_count=0, model=model)
        ref_nl(ctx, DynamicsStage.BEFORE_COMPUTE)

        outputs = model(ref_batch)
        ref_batch.forces.copy_(outputs["forces"])
        ref_batch.energies.copy_(outputs["energies"].view(ref_batch.energies.shape))

        ref_forces = ref_batch.forces
        ref_pe = ref_batch.energies.sum().item()

        # ── Comparison ──
        force_diff = (dd_forces - ref_forces).norm(dim=-1)
        ref_fmag = ref_forces.norm(dim=-1)
        dd_fmag = dd_forces.norm(dim=-1)

        log(rank, "")
        log(rank, "=" * 70)
        log(rank, "FORCE COMPARISON: DD (gathered) vs Single-GPU (same positions)")
        log(rank, "=" * 70)
        log(rank, "  Atoms:              {}", n_atoms)
        log(rank, "  Ref PE (single-GPU): {:.6f} eV", ref_pe)
        log(rank, "  Ref fmax:            {:.6f} eV/A", ref_fmag.max().item())
        log(rank, "  Ref fmean:           {:.6f} eV/A", ref_fmag.mean().item())
        log(rank, "  DD  fmax:            {:.6f} eV/A", dd_fmag.max().item())
        log(rank, "  DD  fmean:           {:.6f} eV/A", dd_fmag.mean().item())
        log(rank, "")
        log(rank, "  Max |F_dd - F_ref|:  {:.6e} eV/A", force_diff.max().item())
        log(rank, "  Mean |F_dd - F_ref|: {:.6e} eV/A", force_diff.mean().item())
        log(
            rank,
            "  RMS |F_dd - F_ref|:  {:.6e} eV/A",
            force_diff.pow(2).mean().sqrt().item(),
        )

        rel_err = force_diff / ref_fmag.clamp(min=1e-10)
        log(rank, "  Max relative err:    {:.6e}", rel_err.max().item())
        log(rank, "  Mean relative err:   {:.6e}", rel_err.mean().item())
        log(rank, "")

        # ── Top 10 worst atoms ──
        worst10 = force_diff.topk(min(10, n_atoms))
        log(rank, "  Top-10 worst atoms:")
        for val, idx in zip(worst10.values, worst10.indices):
            i = idx.item()
            p = dd_positions[i]
            log(
                rank,
                "    atom {:4d}: |dF|={:.6e}  ref_f={:.6f}  dd_f={:.6f}  "
                "pos=[{:.2f},{:.2f},{:.2f}]",
                i,
                val.item(),
                ref_fmag[i].item(),
                dd_fmag[i].item(),
                p[0].item(),
                p[1].item(),
                p[2].item(),
            )

        # ── Neighbor count comparison ──
        ref_nn = ref_batch.num_neighbors
        log(rank, "")
        log(
            rank,
            "  Ref NL: avg_nn={:.1f}  min_nn={}",
            ref_nn.float().mean().item(),
            ref_nn.min().item(),
        )

        if force_diff.max().item() < 1e-5:
            log(rank, "\n  RESULT: PASS — forces match to < 1e-5 eV/A")
        elif force_diff.max().item() < 1e-3:
            log(rank, "\n  RESULT: GOOD — forces match to < 1e-3 eV/A")
        elif force_diff.max().item() < 1e-1:
            log(
                rank,
                "\n  RESULT: MARGINAL — max diff {:.2e} eV/A",
                force_diff.max().item(),
            )
        else:
            log(
                rank, "\n  RESULT: FAIL — max diff {:.2e} eV/A", force_diff.max().item()
            )

    dist.barrier()
    DistributedManager.cleanup()
    log(rank, "Done.")


if __name__ == "__main__":
    main()
