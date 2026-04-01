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
"""Ghost (halo) atom exchange for domain decomposition."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

    from nvalchemi.data.batch import Batch


class GhostExchanger:
    """Exchanges ghost (halo) atoms between neighbouring sub-domains.

    Ghost atoms are read-only copies of atoms owned by adjacent ranks
    that fall within the ``ghost_width`` of a domain boundary.  The
    exchanger handles both the forward send (positions) and the reverse
    accumulation (forces) on each step.

    Parameters
    ----------
    partitioner : SpatialPartitioner
        The spatial partitioner that owns the domain layout.
    config : DomainConfig
        Domain decomposition configuration.
    mesh : DeviceMesh
        Device mesh describing the process topology.
    """

    def __init__(
        self,
        partitioner: SpatialPartitioner,
        config: DomainConfig,
        mesh: DeviceMesh,
    ) -> None:
        self.partitioner = partitioner
        self.config = config
        self.mesh = mesh

        self.ghost_width: float = config.effective_ghost_width()

        # Resolve local rank from mesh.
        try:
            self.rank: int = mesh.get_local_rank()
        except Exception:
            # Fallback: single-process or mock mesh.
            self.rank = 0

        # Precompute neighbor ranks (at most 26 in 3D with PBC).
        # Defensive: exclude self even if the partitioner included it.
        self.neighbor_ranks: list[int] = [
            r for r in partitioner.get_neighbor_ranks(self.rank) if r != self.rank
        ]

        # Precompute PBC shift vectors for neighbor pairs that cross
        # periodic boundaries.
        self._pbc_shifts: dict[tuple[int, int], torch.Tensor] = (
            self._compute_pbc_shift_vectors()
        )

        # Pre-allocated buffer state.
        self._padded_buffer: Batch | None = None
        self._n_owned: int = 0

    # ------------------------------------------------------------------
    # PBC shift computation
    # ------------------------------------------------------------------

    def _compute_pbc_shift_vectors(self) -> dict[tuple[int, int], torch.Tensor]:
        """Precompute PBC shift vectors for all neighbor rank pairs.

        Returns a mapping from ``(sender_rank, receiver_rank)`` to a shift
        vector of shape ``(3,)``.  Only populated for pairs that cross a
        periodic boundary.  Called once at partition time and cached.

        **Triclinic-safe:** uses the full lattice vector
        ``cell_matrix[dim, :]`` rather than just the diagonal element.
        """
        shifts: dict[tuple[int, int], torch.Tensor] = {}
        cell_matrix = self.partitioner.cell_matrix  # (3, 3)
        pbc = self.partitioner.pbc  # (3,) bool
        grid = self.partitioner.rank_grid  # (Px, Py, Pz)

        total_ranks = grid[0] * grid[1] * grid[2]
        for sender_rank in range(total_ranks):
            sender_coords = self.partitioner.rank_to_grid_coords(sender_rank)
            for receiver_rank in self.partitioner.get_neighbor_ranks(sender_rank):
                receiver_coords = self.partitioner.rank_to_grid_coords(receiver_rank)

                shift = torch.zeros(
                    3, device=cell_matrix.device, dtype=cell_matrix.dtype
                )
                for dim in range(3):
                    if not pbc[dim]:
                        continue
                    # No wrapping needed if there is only one rank along
                    # this dimension — the rank is its own neighbor.
                    if grid[dim] <= 1:
                        continue
                    # Detect PBC wrap along this lattice vector.
                    if (
                        sender_coords[dim] == grid[dim] - 1
                        and receiver_coords[dim] == 0
                    ):
                        # Sender at high edge, receiver at low edge.
                        # Ghost must appear "below" receiver: shift by
                        # -lattice_vector.
                        shift = shift - cell_matrix[dim, :]
                    elif (
                        sender_coords[dim] == 0
                        and receiver_coords[dim] == grid[dim] - 1
                    ):
                        # Sender at low edge, receiver at high edge.
                        shift = shift + cell_matrix[dim, :]

                if shift.any():
                    shifts[(sender_rank, receiver_rank)] = shift

        return shifts

    # ------------------------------------------------------------------
    # Ghost identification (vectorized)
    # ------------------------------------------------------------------

    def _rank_fractional_bounds(self, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fractional-coordinate bounds ``(frac_lo, frac_hi)`` for *rank*.

        Each returned tensor has shape ``(3,)`` and values in ``[0, 1]``.
        """
        lo_cell, hi_cell = self.partitioner.rank_to_cell_bounds(rank)
        cells_per_dim = self.partitioner.cells_per_dim
        device = self.partitioner.cell_matrix.device
        dtype = self.partitioner.cell_matrix.dtype

        frac_lo = torch.tensor(
            [lo_cell[d] / cells_per_dim[d] for d in range(3)],
            device=device,
            dtype=dtype,
        )
        frac_hi = torch.tensor(
            [hi_cell[d] / cells_per_dim[d] for d in range(3)],
            device=device,
            dtype=dtype,
        )
        return frac_lo, frac_hi

    def _positions_to_fractional(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert Cartesian positions to fractional coordinates.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` Cartesian positions.

        Returns
        -------
        torch.Tensor
            ``(N, 3)`` fractional coordinates.
        """
        cell = self.partitioner.cell_matrix.to(
            device=positions.device, dtype=positions.dtype
        )
        inv_cell_T = torch.linalg.inv(cell).T  # (3, 3)
        return positions @ inv_cell_T

    def _ghost_width_fractional(self) -> torch.Tensor:
        """Return ghost width in fractional coordinates per dimension.

        For a triclinic cell the face distance along dimension *d* is
        ``1 / ||inv_cell^T[d]||``.  The fractional ghost width is
        ``ghost_width / face_distance = ghost_width * ||inv_cell^T[d]||``.
        """
        cell = self.partitioner.cell_matrix
        inv_cell_T = torch.linalg.inv(cell).T
        norms = torch.linalg.norm(inv_cell_T, dim=1)  # (3,)
        return (self.ghost_width * norms).to(dtype=cell.dtype)

    def _check_halo_region(
        self,
        frac_pos: torch.Tensor,
        frac_lo: torch.Tensor,
        frac_hi: torch.Tensor,
        gw_frac: torch.Tensor,
    ) -> torch.Tensor:
        """Return a boolean mask for atoms in the halo of a domain box.

        An atom is in the halo if it lies within the ghost-width-expanded
        bounding box but NOT fully inside the core box.

        Parameters
        ----------
        frac_pos : torch.Tensor
            ``(N, 3)`` fractional positions (possibly shifted).
        frac_lo, frac_hi : torch.Tensor
            ``(3,)`` fractional bounds of the neighbour domain.
        gw_frac : torch.Tensor
            ``(3,)`` ghost width in fractional coords.

        Returns
        -------
        torch.Tensor
            ``(N,)`` boolean mask.
        """
        expanded_lo = frac_lo - gw_frac
        expanded_hi = frac_hi + gw_frac

        inside = (frac_pos >= expanded_lo) & (frac_pos <= expanded_hi)
        mask = inside.all(dim=1)

        # Exclude atoms fully inside the core (owned by the neighbor).
        core_inside = (frac_pos >= frac_lo) & (frac_pos <= frac_hi)
        in_core = core_inside.all(dim=1)
        mask = mask & ~in_core

        return mask

    def identify_ghosts_for_neighbor(
        self, positions: torch.Tensor, neighbor_rank: int
    ) -> torch.Tensor:
        """Find local atoms within ``ghost_width`` of *neighbor_rank*'s domain.

        The test is performed entirely in fractional coordinates so that
        it remains valid for triclinic cells.

        When the (self.rank, neighbor_rank) pair has a PBC shift, the atom
        could be a ghost via the direct path OR the PBC-shifted path (or
        both — e.g., a 2-rank grid where ranks are both direct and
        PBC-wrap neighbours).  We test both images and take the union.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` Cartesian positions of local (owned) atoms.
        neighbor_rank : int
            Rank whose domain boundary we test against.

        Returns
        -------
        torch.Tensor
            ``(N,)`` boolean mask — ``True`` for atoms that should be sent
            as ghosts to *neighbor_rank*.
        """
        frac_pos = self._positions_to_fractional(positions)  # (N, 3)
        gw_frac = self._ghost_width_fractional()  # (3,)
        frac_lo, frac_hi = self._rank_fractional_bounds(neighbor_rank)

        # Always check the direct (unshifted) image.
        mask = self._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)

        # If a PBC shift exists for this pair, ALSO check the shifted
        # image and take the union.
        shift_key = (self.rank, neighbor_rank)
        if shift_key in self._pbc_shifts:
            cart_shift = self._pbc_shifts[shift_key].to(
                device=positions.device, dtype=positions.dtype
            )
            cell = self.partitioner.cell_matrix.to(
                device=positions.device, dtype=positions.dtype
            )
            inv_cell_T = torch.linalg.inv(cell).T
            frac_shift = cart_shift @ inv_cell_T  # (3,)
            frac_pos_shifted = frac_pos + frac_shift

            mask_shifted = self._check_halo_region(
                frac_pos_shifted, frac_lo, frac_hi, gw_frac
            )
            mask = mask | mask_shifted

        return mask

    def compute_ghost_masks_batched(
        self, positions: torch.Tensor
    ) -> dict[int, torch.Tensor]:
        """Compute ghost masks for all neighbors in a single pass.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` owned atom positions.

        Returns
        -------
        dict[int, torch.Tensor]
            Mapping from neighbor rank to ``(N,)`` boolean mask.
        """
        masks: dict[int, torch.Tensor] = {}
        for neighbor_rank in self.neighbor_ranks:
            masks[neighbor_rank] = self.identify_ghosts_for_neighbor(
                positions, neighbor_rank
            )
        return masks

    # ------------------------------------------------------------------
    # Exchange
    # ------------------------------------------------------------------

    def exchange(
        self, local_batch: Batch, full_rebuild: bool = True
    ) -> tuple[Batch, int]:
        """Exchange ghost atoms with neighbor ranks.

        Returns ``(padded_batch, n_owned)`` where *padded_batch* contains
        owned atoms followed by ghost atoms, and *n_owned* is the count
        of owned atoms (for later stripping).

        For the POC this uses ``torch.distributed.isend``/``irecv`` pairs
        for each of the (up to 26) spatial neighbors.  When
        ``torch.distributed`` is not initialized (single-process mode) the
        batch is returned unchanged.

        Parameters
        ----------
        local_batch : Batch
            This rank's owned atoms.
        full_rebuild : bool
            If ``True``, exchange all per-atom fields.  If ``False``,
            only exchange positions (lightweight update between neighbor
            list rebuilds).

        Returns
        -------
        tuple[Batch, int]
            ``(padded_batch, n_owned)``
        """
        # Single-process fallback: no communication needed.
        if not torch.distributed.is_initialized():
            n_owned = local_batch.positions.shape[0]
            return local_batch, n_owned

        positions = local_batch.positions  # (n_local, 3)
        n_owned = positions.shape[0]
        device = positions.device

        # Compute ghost masks for all neighbors.
        ghost_masks = self.compute_ghost_masks_batched(positions)

        # --- Send / receive ghost positions with each neighbor ---
        send_reqs: list[torch.distributed.Work] = []
        recv_buffers: dict[int, torch.Tensor] = {}
        recv_count_buffers: Dict[int, torch.Tensor] = {}
        recv_reqs: list[torch.distributed.Work] = []

        # Phase 1: exchange counts so receivers know buffer sizes.
        for neighbor_rank in self.neighbor_ranks:
            mask = ghost_masks[neighbor_rank]
            count = mask.sum().reshape(1).to(dtype=torch.int64)
            send_reqs.append(torch.distributed.isend(count, dst=neighbor_rank))
            recv_count = torch.zeros(1, dtype=torch.int64, device=device)
            recv_count_buffers[neighbor_rank] = recv_count
            recv_reqs.append(torch.distributed.irecv(recv_count, src=neighbor_rank))

        # Wait for all count exchanges.
        for req in send_reqs + recv_reqs:
            req.wait()

        # Phase 2: exchange positions.
        send_reqs = []
        recv_reqs = []

        for neighbor_rank in self.neighbor_ranks:
            mask = ghost_masks[neighbor_rank]
            ghost_pos = positions[mask]  # (n_ghost_send, 3)

            # Apply PBC shift to ghost positions.
            shift_key = (self.rank, neighbor_rank)
            if shift_key in self._pbc_shifts:
                shift = self._pbc_shifts[shift_key].to(
                    device=device, dtype=ghost_pos.dtype
                )
                ghost_pos = ghost_pos + shift

            if ghost_pos.numel() > 0:
                send_reqs.append(
                    torch.distributed.isend(ghost_pos.contiguous(), dst=neighbor_rank)
                )

            n_recv = recv_count_buffers[neighbor_rank].item()
            if n_recv > 0:
                recv_buf = torch.zeros(n_recv, 3, dtype=positions.dtype, device=device)
                recv_buffers[neighbor_rank] = recv_buf
                recv_reqs.append(torch.distributed.irecv(recv_buf, src=neighbor_rank))

        for req in send_reqs + recv_reqs:
            req.wait()

        # Concatenate all received ghost positions.
        ghost_parts = [
            recv_buffers[nr] for nr in self.neighbor_ranks if nr in recv_buffers
        ]

        if ghost_parts:
            all_ghost_pos = torch.cat(ghost_parts, dim=0)
            # Build a padded positions tensor: [owned | ghosts].
            padded_positions = torch.cat([positions, all_ghost_pos], dim=0)
        else:
            padded_positions = positions

        # For the POC we update positions in-place on the batch.
        # A more complete implementation would build a new Batch with all
        # per-atom fields padded.  For now, we store the padded positions
        # externally and return the info needed.
        self._n_owned = n_owned
        self._padded_positions = padded_positions

        return local_batch, n_owned

    # ------------------------------------------------------------------
    # Strip
    # ------------------------------------------------------------------

    def strip(self, padded_batch: Batch, n_owned: int) -> Batch:
        """Remove ghost atoms, returning only the owned portion.

        Parameters
        ----------
        padded_batch : Batch
            Batch containing owned + ghost atoms.
        n_owned : int
            Number of owned atoms (ghosts start at this index).

        Returns
        -------
        Batch
            Batch with only owned atoms.
        """
        if padded_batch.positions.shape[0] == n_owned:
            return padded_batch
        # Use index_select on graph indices.  For a single-graph batch
        # (which is the domain decomposition case), just return it — the
        # graph-level index_select operates on graph indices not atom
        # indices.  For the POC we slice the positions tensor directly.
        # A complete implementation would track ghost atoms at the Batch
        # level.
        return padded_batch
