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
"""Analyze dd_debug_rank*.pt files saved by distributed_lj_nve.py.

Usage::

    python examples/analyze_dd_debug.py dd_debug_rank0.pt dd_debug_rank1.pt
"""

from __future__ import annotations

import sys

import torch


def print_owned_table(owned_steps: list[dict], rank: int) -> None:
    """Print a per-step summary table from the owned (after-strip) snapshots."""
    print(f"\n{'=' * 80}")
    print(f"RANK {rank} — Owned atoms (after strip)")
    print(f"{'=' * 80}")
    print(
        f"{'step':>5} {'n':>5} {'PE':>12} {'KE':>10} {'E_total':>12} "
        f"{'fmax':>10} {'fmean':>10} {'Z_min':>7} {'Z_max':>7}"
    )
    print("-" * 95)

    prev_n = None
    for snap in owned_steps:
        step = snap["step"]
        n = snap["n_atoms"]
        pos = snap["positions"]
        forces = snap["forces"]
        vel = snap["velocities"]
        energies = snap["energies"]

        pe = energies.sum().item() if energies is not None else 0
        ke = snap.get("ke", 0)
        if ke == 0 and vel is not None:
            masses = torch.full((n,), 39.948)
            ke = (0.5 * masses.unsqueeze(-1) * vel**2).sum().item()

        fmax = forces.norm(dim=-1).max().item() if forces is not None else 0
        fmean = forces.norm(dim=-1).mean().item() if forces is not None else 0
        z_min = pos[:, 2].min().item()
        z_max = pos[:, 2].max().item()

        flag = ""
        if prev_n is not None and n != prev_n:
            flag += f" [dn={n - prev_n:+d}]"
        if fmax > 1.0:
            flag += " [HIGH_F]"
        if pe > 0:
            flag += " [PE+]"

        print(
            f"{step:5d} {n:5d} {pe:12.4f} {ke:10.4f} {pe + ke:12.4f} "
            f"{fmax:10.4f} {fmean:10.4f} {z_min:7.2f} {z_max:7.2f}{flag}"
        )
        prev_n = n


def print_padded_table(padded_steps: list[dict], rank: int) -> None:
    """Print a per-step summary of the padded (owned+ghost) batch with NL stats."""
    print(f"\n{'=' * 80}")
    print(f"RANK {rank} — Padded batch (owned + ghosts, AABB frame, with NL)")
    print(f"{'=' * 80}")
    print(
        f"{'step':>5} {'n_own':>6} {'n_gho':>6} {'n_tot':>6} {'PE_pad':>12} "
        f"{'fmax_own':>10} {'fmax_gho':>10} "
        f"{'avg_nn':>7} {'min_nn':>7} {'min_nn_own':>10} "
        f"{'Z_min':>7} {'Z_max':>7}"
    )
    print("-" * 120)

    for snap in padded_steps:
        step = snap["step"]
        n_owned = snap["n_owned"]
        n_ghosts = snap["n_ghosts"]
        n_total = snap["n_atoms"]
        pos = snap["positions"]
        forces = snap["forces"]
        energies = snap["energies"]
        nn = snap.get("num_neighbors")

        pe = energies.sum().item() if energies is not None else 0
        fmax_own = (
            forces[:n_owned].norm(dim=-1).max().item()
            if forces is not None and n_owned > 0
            else 0
        )
        fmax_gho = (
            forces[n_owned:].norm(dim=-1).max().item()
            if forces is not None and n_ghosts > 0
            else 0
        )

        if nn is not None:
            avg_nn = nn.float().mean().item()
            min_nn_all = nn.min().item()
            min_nn_own = nn[:n_owned].min().item() if n_owned > 0 else -1
        else:
            avg_nn = min_nn_all = min_nn_own = -1

        z_min = pos[:, 2].min().item()
        z_max = pos[:, 2].max().item()

        flag = ""
        if min_nn_own == 0 and n_owned > 0:
            flag += " [OWNED_NO_NBRS]"
        if fmax_own > 1.0:
            flag += " [HIGH_F]"
        if pe > 0:
            flag += " [PE+]"

        print(
            f"{step:5d} {n_owned:6d} {n_ghosts:6d} {n_total:6d} {pe:12.4f} "
            f"{fmax_own:10.4f} {fmax_gho:10.4f} "
            f"{avg_nn:7.1f} {min_nn_all:7.0f} {min_nn_own:10.0f} "
            f"{z_min:7.2f} {z_max:7.2f}{flag}"
        )


def print_padded_details(padded_steps: list[dict], rank: int) -> None:
    """Print per-atom details for key steps from the padded batch."""
    if not padded_steps:
        return

    # Find worst step (highest PE)
    worst_idx = max(
        range(len(padded_steps)),
        key=lambda i: (
            padded_steps[i]["energies"].sum().item()
            if padded_steps[i]["energies"] is not None
            else 0
        ),
    )
    # Find first step where an owned atom has 0 neighbors
    zero_nn_idx = None
    for i, snap in enumerate(padded_steps):
        nn = snap.get("num_neighbors")
        n_owned = snap["n_owned"]
        if nn is not None and n_owned > 0 and nn[:n_owned].min().item() == 0:
            zero_nn_idx = i
            break

    indices = [("first", 0), ("worst", worst_idx), ("last", len(padded_steps) - 1)]
    if zero_nn_idx is not None:
        indices.append(("first_zero_nn", zero_nn_idx))

    print(f"\n--- Rank {rank}: Per-atom details from padded batch ---")

    for label, idx in indices:
        snap = padded_steps[idx]
        pos = snap["positions"]
        forces = snap["forces"]
        nn = snap.get("num_neighbors")
        nm = snap.get("neighbor_matrix")
        n_owned = snap["n_owned"]
        n_total = snap["n_atoms"]

        if forces is None:
            continue

        fmag = forces.norm(dim=-1)
        print(f"\n  [{label}] step={snap['step']} n_owned={n_owned} n_total={n_total}")

        # Top 5 highest-force OWNED atoms
        if n_owned > 0:
            own_fmag = fmag[:n_owned]
            top5 = own_fmag.topk(min(5, own_fmag.shape[0]))
            print("    Top-5 force OWNED atoms:")
            for val, ai in zip(top5.values, top5.indices):
                p = pos[ai]
                n_nbr = nn[ai].item() if nn is not None else -1
                tag = " <<<" if n_nbr == 0 else ""
                print(
                    f"      atom {ai.item():4d}: f={val:.6f} pos=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] "
                    f"nn={n_nbr}{tag}"
                )

        # Owned atoms with fewest neighbors
        if nn is not None and n_owned > 0:
            own_nn = nn[:n_owned]
            bot5 = own_nn.topk(min(5, own_nn.shape[0]), largest=False)
            print("    Bottom-5 neighbor-count OWNED atoms:")
            for val, ai in zip(bot5.values, bot5.indices):
                p = pos[ai]
                f = fmag[ai].item()
                print(
                    f"      atom {ai.item():4d}: nn={val.item()} "
                    f"f={f:.6f} pos=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]"
                )

        # Ghost atoms with fewest neighbors (might reveal NL issues)
        n_ghosts = n_total - n_owned
        if nn is not None and n_ghosts > 0:
            gho_nn = nn[n_owned:]
            bot3 = gho_nn.topk(min(3, gho_nn.shape[0]), largest=False)
            print("    Bottom-3 neighbor-count GHOST atoms:")
            for val, ai in zip(bot3.values, bot3.indices):
                gi = ai.item() + n_owned  # global index in padded batch
                p = pos[gi]
                f = fmag[gi].item()
                print(
                    f"      atom {gi:4d} (ghost {ai.item():4d}): nn={val.item()} "
                    f"f={f:.6f} pos=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]"
                )

        # Closest pair among owned atoms (via NL)
        if nm is not None and nn is not None and n_owned > 0:
            min_dist = float("inf")
            min_pair = (-1, -1)
            # fill_value = n_total (invalid neighbor entries)
            for i in range(n_owned):
                n_nbr = nn[i].item()
                if n_nbr == 0:
                    continue
                nbrs = nm[i, :n_nbr]
                valid = nbrs[nbrs < n_total]
                if valid.numel() == 0:
                    continue
                dists = (pos[valid] - pos[i]).norm(dim=-1)
                d_min = dists.min().item()
                if d_min < min_dist:
                    min_dist = d_min
                    j = valid[dists.argmin()].item()
                    min_pair = (i, j)
            ghost_tag = " (ghost)" if min_pair[1] >= n_owned else ""
            print(
                f"    Closest pair (owned): atoms ({min_pair[0]}, {min_pair[1]}{ghost_tag}) "
                f"dist={min_dist:.4f} A"
            )


def analyze(paths: list[str]) -> None:
    """Analyze dd_debug_rank*.pt files."""
    ranks_data = {}
    for p in paths:
        d = torch.load(p, weights_only=False)
        r = d["rank"]
        ranks_data[r] = d
        n_owned = len(d.get("owned", d.get("steps", [])))
        n_padded = len(d.get("padded", []))
        print(
            f"Loaded {p}: rank={r} world_size={d['world_size']} "
            f"owned_snaps={n_owned} padded_snaps={n_padded}"
        )

    for r, data in sorted(ranks_data.items()):
        # Support both old format (data["steps"]) and new (data["owned"] + data["padded"])
        owned = data.get("owned", data.get("steps", []))
        padded = data.get("padded", [])

        print_owned_table(owned, r)
        if padded:
            print_padded_table(padded, r)
            print_padded_details(padded, r)
        else:
            print(f"\n  (No padded batch snapshots for rank {r})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_dd_debug.py dd_debug_rank0.pt [dd_debug_rank1.pt ...]"
        )
        sys.exit(1)
    analyze(sys.argv[1:])
