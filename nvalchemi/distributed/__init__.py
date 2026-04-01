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
"""Spatial domain decomposition for distributed molecular dynamics."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nvalchemi.distributed.atom_migrator import AtomMigrator as AtomMigrator
    from nvalchemi.distributed.config import (
        DomainConfig as DomainConfig,
    )
    from nvalchemi.distributed.config import (
        HookScope as HookScope,
    )
    from nvalchemi.distributed.domain_parallel import DomainParallel as DomainParallel
    from nvalchemi.distributed.ghost_exchanger import GhostExchanger as GhostExchanger
    from nvalchemi.distributed.partitioner import (
        SpatialPartitioner as SpatialPartitioner,
    )


def __getattr__(name: str):  # noqa: ANN201
    """Lazy-import public symbols on first access."""
    _imports = {
        "DomainConfig": ("nvalchemi.distributed.config", "DomainConfig"),
        "HookScope": ("nvalchemi.distributed.config", "HookScope"),
        "SpatialPartitioner": (
            "nvalchemi.distributed.partitioner",
            "SpatialPartitioner",
        ),
        "GhostExchanger": ("nvalchemi.distributed.ghost_exchanger", "GhostExchanger"),
        "AtomMigrator": ("nvalchemi.distributed.atom_migrator", "AtomMigrator"),
        "DomainParallel": ("nvalchemi.distributed.domain_parallel", "DomainParallel"),
    }
    if name in _imports:
        module_path, attr = _imports[name]
        import importlib

        module = importlib.import_module(module_path)
        return getattr(module, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AtomMigrator",
    "DomainConfig",
    "DomainParallel",
    "GhostExchanger",
    "HookScope",
    "SpatialPartitioner",
]
