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

        Returns the union of direct-path and PBC-path masks.  For separate
        masks (needed by :meth:`exchange` to apply shifts correctly), use
        :meth:`identify_ghosts_split`.

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
        direct_mask, pbc_mask = self.identify_ghosts_split(positions, neighbor_rank)
        return direct_mask | pbc_mask

    def identify_ghosts_split(
        self, positions: torch.Tensor, neighbor_rank: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return separate direct-path and PBC-path ghost masks.

        This is the core ghost identification method.  The direct mask
        selects atoms in the halo via the direct (unshifted) image; the
        PBC mask selects atoms in the halo via the PBC-shifted image.
        The two sets may overlap (e.g. in a 2-rank grid).

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` Cartesian positions of local (owned) atoms.
        neighbor_rank : int
            Rank whose domain boundary we test against.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            ``(direct_mask, pbc_mask)`` — each ``(N,)`` boolean.
        """
        frac_pos = self._positions_to_fractional(positions)  # (N, 3)
        gw_frac = self._ghost_width_fractional()  # (3,)
        frac_lo, frac_hi = self._rank_fractional_bounds(neighbor_rank)

        # Direct (unshifted) image.
        direct_mask = self._check_halo_region(frac_pos, frac_lo, frac_hi, gw_frac)

        # PBC-shifted image (if applicable).
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

            pbc_mask = self._check_halo_region(
                frac_pos_shifted, frac_lo, frac_hi, gw_frac
            )
        else:
            pbc_mask = torch.zeros(
                positions.shape[0], dtype=torch.bool, device=positions.device
            )

        return direct_mask, pbc_mask

    def compute_ghost_masks_batched(
        self, positions: torch.Tensor
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Compute split ghost masks for all neighbors in a single pass.

        Parameters
        ----------
        positions : torch.Tensor
            ``(N, 3)`` owned atom positions.

        Returns
        -------
        dict[int, tuple[torch.Tensor, torch.Tensor]]
            Mapping from neighbor rank to ``(direct_mask, pbc_mask)``.
        """
        masks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        for neighbor_rank in self.neighbor_ranks:
            masks[neighbor_rank] = self.identify_ghosts_split(positions, neighbor_rank)
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
        import logging as _logging

        _log = _logging.getLogger(__name__)

        # Single-process fallback: no communication needed.
        if not torch.distributed.is_initialized():
            n_owned = local_batch.positions.shape[0]
            return local_batch, n_owned

        positions = local_batch.positions  # (n_local, 3)
        n_owned = positions.shape[0]
        device = positions.device

        _log.info(
            "[rank %d] exchange: n_owned=%d, %d neighbors: %s",
            self.rank,
            n_owned,
            len(self.neighbor_ranks),
            self.neighbor_ranks,
        )

        # Compute split ghost masks (direct vs PBC) for all neighbors.
        ghost_masks = self.compute_ghost_masks_batched(positions)

        for nr, (dm, pm) in ghost_masks.items():
            n_direct = dm.sum().item()
            n_pbc = pm.sum().item()
            # Total unique ghosts (direct | pbc).
            n_total = (dm | pm).sum().item()
            _log.info(
                "[rank %d] ghost mask for neighbor %d: %d direct + %d pbc = %d unique (of %d atoms)",
                self.rank,
                nr,
                n_direct,
                n_pbc,
                n_total,
                dm.shape[0],
            )

        # --- Use batched_isend_irecv to avoid unbatched P2P warnings ---

        # Build per-neighbor send buffers: direct ghosts (unshifted) then
        # PBC ghosts (shifted).  Atoms in both masks are sent TWICE — once
        # at each image — so the receiver sees both.
        send_buffers_all: dict[int, torch.Tensor] = {}
        for neighbor_rank in self.neighbor_ranks:
            direct_mask, pbc_mask = ghost_masks[neighbor_rank]
            parts: list[torch.Tensor] = []

            # Direct ghosts — no shift.
            if direct_mask.any():
                parts.append(positions[direct_mask])

            # PBC ghosts — apply shift.
            if pbc_mask.any():
                shift_key = (self.rank, neighbor_rank)
                pbc_pos = positions[pbc_mask]
                if shift_key in self._pbc_shifts:
                    shift = self._pbc_shifts[shift_key].to(
                        device=device, dtype=pbc_pos.dtype
                    )
                    pbc_pos = pbc_pos + shift
                parts.append(pbc_pos)

            if parts:
                send_buffers_all[neighbor_rank] = torch.cat(parts, dim=0)
            else:
                send_buffers_all[neighbor_rank] = torch.zeros(
                    0, 3, dtype=positions.dtype, device=device
                )

        # Phase 1: exchange counts.
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        _log.info("[rank %d] exchange: Phase 1 — exchanging counts", self.rank)

        send_counts: dict[int, torch.Tensor] = {}
        recv_counts: dict[int, torch.Tensor] = {}
        p2p_ops: list[torch.distributed.P2POp] = []

        for neighbor_rank in self.neighbor_ranks:
            n_send = send_buffers_all[neighbor_rank].shape[0]
            count_send = torch.tensor([n_send], dtype=torch.int64, device=device)
            send_counts[neighbor_rank] = count_send
            count_recv = torch.zeros(1, dtype=torch.int64, device=device)
            recv_counts[neighbor_rank] = count_recv
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.isend, count_send, neighbor_rank
                )
            )
            p2p_ops.append(
                torch.distributed.P2POp(
                    torch.distributed.irecv, count_recv, neighbor_rank
                )
            )

        if p2p_ops:
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        for nr, c in recv_counts.items():
            _log.info(
                "[rank %d] will receive %d ghosts from rank %d", self.rank, c.item(), nr
            )

        # Phase 2: exchange positions.
        _log.info("[rank %d] exchange: Phase 2 — exchanging positions", self.rank)
        recv_buffers: dict[int, torch.Tensor] = {}
        p2p_ops = []

        for neighbor_rank in self.neighbor_ranks:
            send_buf = send_buffers_all[neighbor_rank]
            n_send = send_buf.shape[0]
            n_recv = recv_counts[neighbor_rank].item()

            _log.info(
                "[rank %d] neighbor %d: sending %d, receiving %d",
                self.rank,
                neighbor_rank,
                n_send,
                n_recv,
            )

            if n_send > 0:
                p2p_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.isend, send_buf.contiguous(), neighbor_rank
                    )
                )
            if n_recv > 0:
                recv_buf = torch.zeros(n_recv, 3, dtype=positions.dtype, device=device)
                recv_buffers[neighbor_rank] = recv_buf
                p2p_ops.append(
                    torch.distributed.P2POp(
                        torch.distributed.irecv, recv_buf, neighbor_rank
                    )
                )

        if p2p_ops:
            _log.info("[rank %d] exchange: %d P2P ops queued", self.rank, len(p2p_ops))
            reqs = torch.distributed.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        _log.info("[rank %d] exchange: Phase 2 complete", self.rank)

        # Concatenate all received ghost positions.
        ghost_parts = [
            recv_buffers[nr] for nr in self.neighbor_ranks if nr in recv_buffers
        ]

        if ghost_parts:
            all_ghost_pos = torch.cat(ghost_parts, dim=0)
        else:
            all_ghost_pos = torch.zeros(0, 3, dtype=positions.dtype, device=device)

        n_ghosts = all_ghost_pos.shape[0]
        _log.info(
            "[rank %d] exchange: done — %d owned + %d ghosts = %d total",
            self.rank,
            n_owned,
            n_ghosts,
            n_owned + n_ghosts,
        )

        self._n_owned = n_owned

        if n_ghosts == 0:
            return local_batch, n_owned

        # Build a padded Batch by creating ghost AtomicData and appending
        # to the local batch.  Ghost atoms get dummy atomic_numbers and
        # masses copied from the first owned atom (all same element in
        # typical LJ/MLIP simulations; a full implementation would exchange
        # these fields too).
        from nvalchemi.data.atomic_data import AtomicData
        from nvalchemi.data.batch import Batch as BatchCls

        ghost_data = AtomicData(
            positions=all_ghost_pos,
            atomic_numbers=local_batch.atomic_numbers[:1].expand(n_ghosts).clone(),
            atomic_masses=(
                local_batch.atomic_masses[:1].expand(n_ghosts).clone()
                if hasattr(local_batch, "atomic_masses")
                and local_batch.atomic_masses is not None
                else torch.ones(n_ghosts, device=device)
            ),
            cell=local_batch.cell.clone(),
            pbc=local_batch.pbc.clone()
            if hasattr(local_batch, "pbc") and local_batch.pbc is not None
            else None,
        )

        # If the local batch has velocities, add zero velocities for ghosts
        # (ghosts are read-only — their velocities aren't used).
        if hasattr(local_batch, "velocities") and local_batch.velocities is not None:
            ghost_data.add_node_property(
                "velocities",
                torch.zeros(n_ghosts, 3, dtype=positions.dtype, device=device),
            )

        # If the local batch has forces, add zero forces for ghosts.
        if hasattr(local_batch, "forces") and local_batch.forces is not None:
            ghost_data.add_node_property(
                "forces", torch.zeros(n_ghosts, 3, dtype=positions.dtype, device=device)
            )

        ghost_batch = BatchCls.from_data_list([ghost_data], device=device)

        # Build padded batch: clone owned, then append ghosts.
        # Clone so we don't modify the caller's local_batch.
        padded_batch = local_batch.clone()
        padded_batch.append(ghost_batch)

        _log.info(
            "[rank %d] padded batch: num_nodes=%d num_graphs=%d",
            self.rank,
            padded_batch.num_nodes,
            padded_batch.num_graphs,
        )

        return padded_batch, n_owned

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

        # The padded batch has 2 graphs: [owned (graph 0) | ghosts (graph 1)].
        # Select only graph 0 to strip the ghost atoms.
        import logging as _logging

        _log2 = _logging.getLogger(__name__)
        _log2.info(
            "[rank %d] strip: %d total atoms → keeping graph 0 (%d owned)",
            self.rank,
            padded_batch.num_nodes,
            n_owned,
        )
        stripped = padded_batch.index_select(
            torch.tensor([0], device=padded_batch.positions.device)
        )
        return stripped
