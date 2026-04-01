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
"""Configuration types for spatial domain decomposition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from pydantic import BaseModel


class HookScope(Enum):
    """Determines which ranks execute a hook callback.

    Attributes
    ----------
    LOCAL : str
        Hook runs on every rank with its local subdomain batch.
    GLOBAL : str
        Hook runs on every rank after an all-gather produces the full batch.
    RANK_ZERO : str
        Hook runs only on rank 0 after gathering.
    """

    LOCAL = "local"
    GLOBAL = "global"
    RANK_ZERO = "rank_zero"


class DomainConfig(BaseModel):
    """Configuration for spatial domain decomposition.

    Parameters
    ----------
    cutoff : float
        Interaction cutoff radius used by the model.
    skin : float
        Additional skin distance for neighbor-list buffering.
    mesh : DeviceMesh | None
        Optional ``torch.distributed.DeviceMesh`` describing the process
        topology.  Kept behind ``TYPE_CHECKING`` to avoid a hard dependency.
    mesh_dim : str
        Name of the mesh dimension used for domain parallelism.
    ghost_width : float | None
        Width of the ghost (halo) region.  When ``None`` the effective width
        is computed as ``cutoff + skin``.
    grid_dims : tuple[int, int, int] | None
        Explicit grid dimensions for the spatial decomposition.  When
        ``None`` the partitioner will choose automatically.
    """

    model_config = {"arbitrary_types_allowed": True}

    cutoff: float
    skin: float = 0.0
    mesh: Any = None  # DeviceMesh at runtime; typed as Any to avoid Pydantic resolution of TYPE_CHECKING import
    mesh_dim: str = "domain"
    ghost_width: float | None = None
    grid_dims: tuple[int, int, int] | None = None

    def effective_ghost_width(self) -> float:
        """Return the ghost region width, defaulting to ``cutoff + skin``."""
        return (
            self.ghost_width
            if self.ghost_width is not None
            else self.cutoff + self.skin
        )


@dataclass
class _GeometrySnapshot:
    """Saved geometry state for the prepare/unprepare cycle.

    Attributes
    ----------
    original_cell : torch.Tensor
        Cell matrix before domain decomposition adjustments.
    original_pbc : torch.Tensor
        Periodic boundary condition flags before adjustments.
    pos_min : torch.Tensor | None
        Minimum position coordinates observed during partitioning.
    """

    original_cell: torch.Tensor
    original_pbc: torch.Tensor
    pos_min: torch.Tensor | None = None
