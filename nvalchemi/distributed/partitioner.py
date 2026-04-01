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
"""Spatial partitioner for domain decomposition."""

from __future__ import annotations

import math
import warnings

import torch

from nvalchemi.distributed.config import DomainConfig


class SpatialPartitioner:
    """Assigns atoms to spatial sub-domains on a Cartesian grid.

    The partitioner divides the simulation cell into axis-aligned blocks
    and maps each atom to the rank that owns its block.

    Parameters
    ----------
    config : DomainConfig
        Domain decomposition configuration.
    cell_matrix : torch.Tensor
        ``(3, 3)`` cell / box matrix describing the simulation domain.
    pbc : torch.Tensor
        ``(3,)`` boolean tensor of periodic boundary conditions per axis.
    """

    def __init__(
        self,
        config: DomainConfig,
        cell_matrix: torch.Tensor,
        pbc: torch.Tensor,
    ) -> None:
        self.config = config
        self.cell_matrix = cell_matrix
        self.pbc = pbc

        # Determine world size from mesh or default to 1.
        if config.mesh is not None:
            self.world_size: int = config.mesh.size()
        else:
            self.world_size = 1

        # Compute cells_per_dimension from cell geometry and cutoff.
        if config.grid_dims is not None:
            self.cells_per_dim: tuple[int, int, int] = config.grid_dims
        else:
            self.cells_per_dim = self._compute_cells_per_dim(cell_matrix, config.cutoff)

        # Refine the cell grid if there are fewer cells than ranks.
        total_cells = (
            self.cells_per_dim[0] * self.cells_per_dim[1] * self.cells_per_dim[2]
        )
        if total_cells < self.world_size:
            self.cells_per_dim = SpatialPartitioner.refine_grid_for_ranks(
                self.cells_per_dim, self.world_size
            )

        # Compute the rank grid (Px, Py, Pz).
        self.rank_grid: tuple[int, int, int] = SpatialPartitioner.compute_rank_grid(
            self.cells_per_dim, self.world_size
        )

        # Precompute neighbor ranks for every rank.
        self._neighbor_ranks: dict[int, list[int]] = self._compute_all_neighbor_ranks()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cells_per_dim(
        cell_matrix: torch.Tensor, cutoff: float
    ) -> tuple[int, int, int]:
        """Compute cells per dimension from cell geometry and cutoff.

        Uses the nvalchemiops formula:
            face_distance = 1.0 / norm(inverse_cell_T[dim])
            cells = max(floor(face_distance / cutoff), 1)
        """
        inv_cell = torch.linalg.inv(cell_matrix)
        inv_cell_T = inv_cell.T  # (3, 3)

        dims: list[int] = []
        for dim in range(3):
            face_distance = 1.0 / torch.linalg.norm(inv_cell_T[dim]).item()
            dims.append(max(int(math.floor(face_distance / cutoff)), 1))
        return (dims[0], dims[1], dims[2])

    @staticmethod
    def compute_rank_grid(
        cells_per_dim: tuple[int, int, int], world_size: int
    ) -> tuple[int, int, int]:
        """Compute ``(Px, Py, Pz)`` rank grid minimizing surface area.

        Enumerates all 3-factor factorizations of *world_size* and picks the
        one that minimizes the surface-area proxy
        ``2 * (dx*dy + dy*dz + dx*dz)`` where ``dx = Nx/Px``, etc.
        """
        Nx, Ny, Nz = cells_per_dim

        best_grid: tuple[int, int, int] | None = None
        best_surface = float("inf")

        for Px in range(1, world_size + 1):
            if world_size % Px != 0:
                continue
            remainder = world_size // Px
            for Py in range(1, remainder + 1):
                if remainder % Py != 0:
                    continue
                Pz = remainder // Py

                dx = Nx / Px
                dy = Ny / Py
                dz = Nz / Pz
                surface = 2.0 * (dx * dy + dy * dz + dx * dz)

                if surface < best_surface:
                    best_surface = surface
                    best_grid = (Px, Py, Pz)

        if best_grid is None:
            raise ValueError("No valid factorization found")
        return best_grid

    @staticmethod
    def refine_grid_for_ranks(
        cells_per_dim: tuple[int, int, int], world_size: int
    ) -> tuple[int, int, int]:
        """Subdivide cells until there are at least *world_size* cells.

        Doubles the smallest dimension iteratively. Warns if total cells
        remain less than *world_size* after 64 iterations (safety cap).
        """
        Nx, Ny, Nz = cells_per_dim
        max_iters = 64
        for _ in range(max_iters):
            if Nx * Ny * Nz >= world_size:
                break
            # Double the smallest dimension.
            min_val = min(Nx, Ny, Nz)
            if Nx == min_val:
                Nx *= 2
            elif Ny == min_val:
                Ny *= 2
            else:
                Nz *= 2

        if Nx * Ny * Nz < world_size:
            warnings.warn(
                f"Could not refine cell grid to {world_size} cells; "
                f"got {Nx * Ny * Nz} cells with grid ({Nx}, {Ny}, {Nz}).",
                stacklevel=2,
            )
        return (Nx, Ny, Nz)

    # ------------------------------------------------------------------
    # Cell ↔ rank mapping
    # ------------------------------------------------------------------

    def cell_to_rank(
        self, ix: int | torch.Tensor, iy: int | torch.Tensor, iz: int | torch.Tensor
    ) -> int | torch.Tensor:
        """Map cell indices ``(ix, iy, iz)`` to the owning rank.

        Works with both scalar ints and batched :class:`torch.Tensor` inputs.
        Uses ceiling-division block assignment.
        """
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        if isinstance(ix, torch.Tensor):
            rx = torch.clamp(ix // cx, max=Px - 1)
            ry = torch.clamp(iy // cy, max=Py - 1)
            rz = torch.clamp(iz // cz, max=Pz - 1)
            return rx + Px * (ry + Py * rz)
        else:
            rx = min(ix // cx, Px - 1)
            ry = min(iy // cy, Py - 1)
            rz = min(iz // cz, Pz - 1)
            return rx + Px * (ry + Py * rz)

    def rank_to_cell_bounds(
        self, rank: int
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        """Return cell index bounds ``(lo, hi)`` owned by *rank*.

        ``lo`` is inclusive, ``hi`` is exclusive.
        """
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        rx, ry, rz = self.rank_to_grid_coords(rank)

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        lo = (rx * cx, ry * cy, rz * cz)
        hi = (min((rx + 1) * cx, Nx), min((ry + 1) * cy, Ny), min((rz + 1) * cz, Nz))
        return lo, hi

    def rank_to_grid_coords(self, rank: int) -> tuple[int, int, int]:
        """Decompose a linear rank index into ``(rx, ry, rz)`` grid coords."""
        Px, Py, _Pz = self.rank_grid
        rx = rank % Px
        ry = (rank // Px) % Py
        rz = rank // (Px * Py)
        return (rx, ry, rz)

    # ------------------------------------------------------------------
    # Neighbor ranks
    # ------------------------------------------------------------------

    def _compute_all_neighbor_ranks(self) -> dict[int, list[int]]:
        """Precompute the set of neighbor ranks for every rank."""
        Px, Py, Pz = self.rank_grid
        total_ranks = Px * Py * Pz
        neighbor_map: dict[int, list[int]] = {}
        for rank in range(total_ranks):
            neighbor_map[rank] = self._compute_neighbor_ranks_for(rank)
        return neighbor_map

    def _compute_neighbor_ranks_for(self, rank: int) -> list[int]:
        """Return up to 26 spatial neighbor ranks for *rank*.

        For PBC dimensions, wrap around. For non-PBC, skip out-of-bounds.
        """
        Px, Py, Pz = self.rank_grid
        rx, ry, rz = self.rank_to_grid_coords(rank)
        pbc_x = bool(self.pbc[0])
        pbc_y = bool(self.pbc[1])
        pbc_z = bool(self.pbc[2])

        neighbors: list[int] = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dx == 0 and dy == 0 and dz == 0:
                        continue
                    nx = rx + dx
                    ny = ry + dy
                    nz = rz + dz

                    # Check bounds / wrap for each dimension.
                    if not self._in_bounds_or_wrap(nx, Px, pbc_x):
                        continue
                    if not self._in_bounds_or_wrap(ny, Py, pbc_y):
                        continue
                    if not self._in_bounds_or_wrap(nz, Pz, pbc_z):
                        continue

                    nx = nx % Px
                    ny = ny % Py
                    nz = nz % Pz

                    neighbor_rank = nx + Px * (ny + Py * nz)
                    if neighbor_rank not in neighbors:
                        neighbors.append(neighbor_rank)

        return neighbors

    @staticmethod
    def _in_bounds_or_wrap(coord: int, size: int, periodic: bool) -> bool:
        """Check if a neighbor coordinate is valid, considering PBC."""
        if 0 <= coord < size:
            return True
        if periodic:
            return True
        return False

    def get_neighbor_ranks(self, rank: int) -> list[int]:
        """Return precomputed neighbor ranks for *rank*."""
        return self._neighbor_ranks[rank]

    # ------------------------------------------------------------------
    # Atom assignment (vectorized)
    # ------------------------------------------------------------------

    def assign_atoms_to_ranks(self, positions: torch.Tensor) -> torch.Tensor:
        """Assign each atom to a rank based on its position.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` atom positions in Cartesian coordinates.

        Returns
        -------
        torch.Tensor
            ``(N,)`` integer tensor of rank assignments.
        """
        device = positions.device
        dtype = positions.dtype

        # Fractional coordinates.
        inv_cell_T = torch.linalg.inv(self.cell_matrix.to(device=device, dtype=dtype)).T
        frac = positions @ inv_cell_T  # (N, 3)

        cells_per_dim_t = torch.tensor(self.cells_per_dim, device=device, dtype=dtype)

        # Cell coordinates.
        cell_coords = torch.floor(frac * cells_per_dim_t).to(torch.int64)

        # PBC wrap for periodic dimensions; clamp for non-periodic.
        cells_per_dim_int = torch.tensor(
            self.cells_per_dim, device=device, dtype=torch.int64
        )
        pbc_mask = self.pbc.to(device=device)

        # Wrap periodic dims via modulo.
        wrapped = cell_coords % cells_per_dim_int
        # Clamp non-periodic dims.
        clamped = torch.clamp(
            cell_coords,
            min=torch.zeros_like(cells_per_dim_int),
            max=cells_per_dim_int - 1,
        )
        # Select based on pbc mask.
        cell_coords = torch.where(pbc_mask.unsqueeze(0), wrapped, clamped)

        # Vectorized cell_to_rank.
        Nx, Ny, Nz = self.cells_per_dim
        Px, Py, Pz = self.rank_grid

        cx = math.ceil(Nx / Px)
        cy = math.ceil(Ny / Py)
        cz = math.ceil(Nz / Pz)

        rx = torch.clamp(cell_coords[:, 0] // cx, max=Px - 1)
        ry = torch.clamp(cell_coords[:, 1] // cy, max=Py - 1)
        rz = torch.clamp(cell_coords[:, 2] // cz, max=Pz - 1)

        ranks = rx + Px * (ry + Py * rz)
        return ranks
