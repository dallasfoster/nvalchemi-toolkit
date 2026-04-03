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

from __future__ import annotations

import sys

import torch


def analyze(paths: list[str]) -> None:
    """Analyze dd_debug_rank*.pt files saved by distributed_lj_nve.py."""
    ranks = {}
    for p in paths:
        d = torch.load(p, weights_only=False)
        ranks[d["rank"]] = d["steps"]
        print(
            f"Loaded {p}: rank={d['rank']} world_size={d['world_size']} steps={len(d['steps'])}"
        )

    for r, steps in sorted(ranks.items()):
        print(f"\n{'=' * 72}")
        print(f"RANK {r}")
        print(f"{'=' * 72}")
        print(
            f"{'step':>5} {'n':>5} {'PE':>12} {'KE':>10} {'E_total':>12} "
            f"{'fmax':>10} {'fmean':>10} {'Z_min':>7} {'Z_max':>7} "
            f"{'avg_nn':>7} {'min_nn':>7}"
        )
        print("-" * 110)

        prev_n = None
        for snap in steps:
            step = snap["step"]
            n = snap["n_atoms"]
            pos = snap["positions"]
            forces = snap["forces"]
            vel = snap["velocities"]
            energies = snap["energies"]
            nm = snap.get("neighbor_matrix")
            nn = snap.get("num_neighbors")

            pe = energies.sum().item() if energies is not None else 0
            ke = snap.get("ke", 0)
            if ke == 0 and vel is not None:
                masses = torch.full((n,), 39.948)
                ke = (0.5 * masses.unsqueeze(-1) * vel**2).sum().item()

            fmax = forces.norm(dim=-1).max().item() if forces is not None else 0
            fmean = forces.norm(dim=-1).mean().item() if forces is not None else 0
            z_min = pos[:, 2].min().item()
            z_max = pos[:, 2].max().item()
            avg_nn = nn.float().mean().item() if nn is not None else -1
            min_nn = nn.min().item() if nn is not None else -1

            flag = ""
            if prev_n is not None and n != prev_n:
                flag += f" [n_change={n - prev_n:+d}]"
            if fmax > 1.0:
                flag += " [HIGH_FORCE]"
            if pe > 0:
                flag += " [PE_POSITIVE]"

            print(
                f"{step:5d} {n:5d} {pe:12.4f} {ke:10.4f} {pe + ke:12.4f} "
                f"{fmax:10.4f} {fmean:10.4f} {z_min:7.2f} {z_max:7.2f} "
                f"{avg_nn:7.1f} {min_nn:7.0f}{flag}"
            )
            prev_n = n

        # ── Detailed analysis of specific steps ──
        print("\n--- Per-atom details for first/last/worst steps ---")
        if len(steps) < 2:
            continue

        # Find worst step (highest PE)
        worst_idx = max(
            range(len(steps)),
            key=lambda i: (
                steps[i]["energies"].sum().item()
                if steps[i]["energies"] is not None
                else 0
            ),
        )
        for label, idx in [
            ("first", 1),
            ("worst", worst_idx),
            ("last", len(steps) - 1),
        ]:
            snap = steps[idx]
            if snap["forces"] is None:
                continue
            pos = snap["positions"]
            forces = snap["forces"]
            fmag = forces.norm(dim=-1)
            nn = snap.get("num_neighbors")
            nm = snap.get("neighbor_matrix")

            print(f"\n  [{label}] step={snap['step']} n={snap['n_atoms']}")

            # Top 5 highest-force atoms
            top5 = fmag.topk(min(5, fmag.shape[0]))
            print("    Top-5 force atoms:")
            for k, (val, ai) in enumerate(zip(top5.values, top5.indices)):
                p = pos[ai]
                n_nbr = nn[ai].item() if nn is not None else -1
                print(
                    f"      atom {ai.item():4d}: f={val:.6f} pos=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}] "
                    f"neighbors={n_nbr}"
                )

            # Atoms with fewest neighbors
            if nn is not None:
                bot5 = nn.topk(min(5, nn.shape[0]), largest=False)
                print("    Bottom-5 neighbor-count atoms:")
                for val, ai in zip(bot5.values, bot5.indices):
                    p = pos[ai]
                    f = fmag[ai].item()
                    print(
                        f"      atom {ai.item():4d}: neighbors={val.item()} "
                        f"f={f:.6f} pos=[{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}]"
                    )

            # Check for very close pairs via neighbor matrix
            if nm is not None and nn is not None:
                min_dist = float("inf")
                min_pair = (-1, -1)
                for i in range(pos.shape[0]):
                    nbrs = nm[i, : nn[i]]
                    if nbrs.numel() == 0:
                        continue
                    valid = nbrs[nbrs < pos.shape[0]]
                    if valid.numel() == 0:
                        continue
                    dists = (pos[valid] - pos[i]).norm(dim=-1)
                    d_min = dists.min().item()
                    if d_min < min_dist:
                        min_dist = d_min
                        j = valid[dists.argmin()].item()
                        min_pair = (i, j)
                print(
                    f"    Closest pair: atoms ({min_pair[0]}, {min_pair[1]}) "
                    f"dist={min_dist:.4f} A"
                )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            "Usage: python analyze_dd_debug.py dd_debug_rank0.pt [dd_debug_rank1.pt ...]"
        )
        sys.exit(1)
    analyze(sys.argv[1:])
