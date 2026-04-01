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
"""Atom migration between sub-domains after integration steps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from nvalchemi.distributed.config import DomainConfig
from nvalchemi.distributed.partitioner import SpatialPartitioner

if TYPE_CHECKING:
    from torch.distributed import DeviceMesh

    from nvalchemi.data.batch import Batch

# Atom-level fields to pack for migration, in order.
# Float fields first, then integer fields (cast to float for packing).
_FLOAT_FIELDS: tuple[str, ...] = (
    "positions",
    "velocities",
    "forces",
)
_INT_FIELDS: tuple[str, ...] = (
    "atomic_numbers",
    "atomic_masses",
)
_ALL_FIELDS: tuple[str, ...] = _FLOAT_FIELDS + _INT_FIELDS


# ---------------------------------------------------------------------------
# Packing / unpacking helpers
# ---------------------------------------------------------------------------


def _field_width(batch: Batch, name: str) -> int:
    """Return the last-dimension size of a field, or 1 if the tensor is 1-D."""
    t = getattr(batch, name, None)
    if t is None:
        return 0
    return t.shape[-1] if t.ndim > 1 else 1


def pack_atom_fields(
    batch: Batch,
    indices: torch.Tensor,
) -> torch.Tensor:
    """Select atoms at *indices* and pack all fields into a single ``(N, D)`` tensor.

    Integer fields (``atomic_numbers``, ``atomic_masses``) are cast to the same
    float dtype as ``positions`` so that all columns share one dtype.

    Parameters
    ----------
    batch : Batch
        Source batch containing atom-level data.
    indices : torch.Tensor
        ``(N,)`` integer indices selecting atoms to pack.

    Returns
    -------
    torch.Tensor
        ``(N, D)`` packed tensor where *D* is the sum of per-field widths.
    """
    dtype = batch.positions.dtype
    columns: list[torch.Tensor] = []

    for name in _ALL_FIELDS:
        t = getattr(batch, name, None)
        if t is None:
            continue
        selected = t[indices]
        if selected.ndim == 1:
            selected = selected.unsqueeze(-1)
        columns.append(selected.to(dtype))

    return torch.cat(columns, dim=-1)


def _compute_field_layout(
    batch: Batch,
) -> list[tuple[str, int, torch.dtype]]:
    """Return ``(field_name, width, original_dtype)`` for each present field."""
    layout: list[tuple[str, int, torch.dtype]] = []
    for name in _ALL_FIELDS:
        t = getattr(batch, name, None)
        if t is None:
            continue
        width = t.shape[-1] if t.ndim > 1 else 1
        layout.append((name, width, t.dtype))
    return layout


def unpack_atom_fields(
    packed: torch.Tensor,
    layout: list[tuple[str, int, torch.dtype]],
) -> dict[str, torch.Tensor]:
    """Reverse of :func:`pack_atom_fields`.

    Parameters
    ----------
    packed : torch.Tensor
        ``(N, D)`` packed tensor.
    layout : list[tuple[str, int, torch.dtype]]
        Field layout from :func:`_compute_field_layout`.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from field name to ``(N, ...)`` tensor with the original dtype
        and shape restored.
    """
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, width, orig_dtype in layout:
        col = packed[:, offset : offset + width]
        if width == 1:
            col = col.squeeze(-1)
        result[name] = col.to(orig_dtype)
        offset += width
    return result


def packed_dim(batch: Batch) -> int:
    """Total number of packed float columns for *batch*."""
    total = 0
    for name in _ALL_FIELDS:
        t = getattr(batch, name, None)
        if t is None:
            continue
        total += t.shape[-1] if t.ndim > 1 else 1
    return total


# ---------------------------------------------------------------------------
# AtomMigrator
# ---------------------------------------------------------------------------


class AtomMigrator:
    """Migrates atoms that have crossed sub-domain boundaries.

    After each integration step atoms may leave the spatial region owned
    by the current rank.  The migrator detects boundary crossings and
    transfers full atom state to the new owning rank via point-to-point
    communication over the ``DeviceMesh``.

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
        self.rank: int = mesh.get_local_rank()
        self.world_size: int = mesh.size()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def needs_migration(self, local_batch: Batch) -> bool:
        """Check whether any atoms have left their domain.

        For the POC this returns ``False`` — migration is disabled until
        Verlet skin invalidation detection is wired up.  At 300 K with
        1 fs timestep, atoms move ~0.03 A/step which is well within the
        ghost shell for hundreds of steps.
        """
        return False

    def migrate(self, local_batch: Batch) -> Batch:
        """Detect atoms that left this rank's domain and redistribute them.

        Uses sort-based index construction (``argsort`` + ``bincount``)
        to avoid per-rank ``nonzero()`` loops that would each force a
        CPU-GPU sync.

        Parameters
        ----------
        local_batch : Batch
            The local batch on this rank.

        Returns
        -------
        Batch
            Updated local batch with migrated atoms removed and incoming
            atoms appended.
        """
        # Guard: if distributed is not initialised, nothing to do.
        if not dist.is_initialized():
            return local_batch

        world_size = self.world_size
        rank = self.rank
        device = local_batch.positions.device

        # 1. Recompute rank assignments for all local atoms.
        new_rank = self.partitioner.assign_atoms_to_ranks(local_batch.positions)
        # Ensure int64 for bincount / argsort.
        new_rank = new_rank.to(torch.int64)

        # 2. Sort-based index construction — ONE sort, no per-rank nonzero.
        counts = torch.bincount(new_rank, minlength=world_size)
        sorted_idx = torch.argsort(new_rank)
        offsets = torch.cat(
            [torch.zeros(1, dtype=counts.dtype, device=device), counts.cumsum(0)]
        )

        # 3. Build per-rank send index slices.
        send_indices: list[torch.Tensor] = [
            sorted_idx[offsets[r] : offsets[r + 1]] for r in range(world_size)
        ]

        # 4. Exchange counts via all_to_all_single so every rank knows how
        #    many atoms it will receive from each other rank.
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts, counts, group=self.mesh.get_group())

        # 5. Determine field layout and packed dimension.
        layout = _compute_field_layout(local_batch)
        pdim = packed_dim(local_batch)
        dtype = local_batch.positions.dtype

        # 6-8. Exchange atom data using batch_isend_irecv to avoid
        #       NCCL unbatched P2P communicator issues.
        p2p_ops: list[dist.P2POp] = []
        _send_bufs: list[torch.Tensor] = []  # prevent GC before send completes
        recv_buffers: list[torch.Tensor] = []
        recv_src_ranks: list[int] = []

        for r in range(world_size):
            if r == rank:
                continue
            n_send = counts[r].item()
            n_recv = recv_counts[r].item()

            if n_send > 0:
                buf = pack_atom_fields(local_batch, send_indices[r])
                _send_bufs.append(buf)
                p2p_ops.append(dist.P2POp(dist.isend, buf, r))

            if n_recv > 0:
                recv_buf = torch.empty(n_recv, pdim, dtype=dtype, device=device)
                recv_buffers.append(recv_buf)
                recv_src_ranks.append(r)
                p2p_ops.append(dist.P2POp(dist.irecv, recv_buf, r))

        if p2p_ops:
            reqs = dist.batch_isend_irecv(p2p_ops)
            for req in reqs:
                req.wait()

        # 9. Build the new local batch.
        #    Staying atoms: the slice of sorted_idx that maps to our rank.
        staying_idx = send_indices[rank]

        new_batch = local_batch.index_select(
            _graph_indices_for_atoms(local_batch, staying_idx)
        )

        # If we received atoms, unpack them and build a temporary batch to
        # append.  For the POC we construct per-atom AtomicData and batch them.
        if recv_buffers:
            all_received = torch.cat(recv_buffers, dim=0)
            incoming_fields = unpack_atom_fields(all_received, layout)
            incoming_batch = _build_batch_from_fields(incoming_fields, device)
            new_batch.append(incoming_batch)

        return new_batch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _graph_indices_for_atoms(batch: Batch, atom_indices: torch.Tensor) -> torch.Tensor:
    """Return **graph-level** indices that contain at least one atom from *atom_indices*.

    ``Batch.index_select`` operates on graphs (systems), not individual atoms.
    In a domain-decomposed context each graph is a single system, so we map
    from atom indices back to their owning graph indices.

    For the common single-system-per-rank case this simply returns ``[0]``.
    """
    batch_vec = batch.batch  # (num_atoms,) graph assignment
    graph_ids = batch_vec[atom_indices]
    unique_graphs = torch.unique(graph_ids)
    return unique_graphs


def _build_batch_from_fields(
    fields: dict[str, torch.Tensor], device: torch.device
) -> Batch:
    """Construct a minimal single-graph :class:`Batch` from unpacked field tensors."""
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch

    n_atoms = next(iter(fields.values())).shape[0]

    # Build AtomicData with required keys.
    kwargs: dict[str, torch.Tensor] = {}
    if "positions" in fields:
        kwargs["positions"] = fields["positions"]
    else:
        kwargs["positions"] = torch.zeros(n_atoms, 3, device=device)
    if "atomic_numbers" in fields:
        kwargs["atomic_numbers"] = fields["atomic_numbers"]
    else:
        kwargs["atomic_numbers"] = torch.zeros(n_atoms, dtype=torch.long, device=device)

    data = AtomicData(**kwargs)

    # Attach optional fields.
    for name in ("velocities", "forces"):
        if name in fields:
            data.add_node_property(name, fields[name])
    if "atomic_masses" in fields:
        data.add_node_property("atomic_masses", fields["atomic_masses"])

    return Batch.from_data_list([data], device=device)
