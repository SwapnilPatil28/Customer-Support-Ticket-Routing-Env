# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Incident Command Center environment for OpenEnv.

The client module depends on the optional `openenv-core` package. We import
it lazily so that pure-domain consumers (such as the pytest domain suite)
can import this package even when OpenEnv is not installed.
"""

from __future__ import annotations

from .models import IncidentAction, IncidentObservation, IncidentState

__version__ = "3.0.0"

try:  # Optional runtime dependency — only required for HTTP clients.
    from .client import IncidentCommandEnvClient, SREEnvClient
except Exception:  # pragma: no cover - defensive fallback for domain-only users
    IncidentCommandEnvClient = None  # type: ignore[assignment]
    SREEnvClient = None  # type: ignore[assignment]

__all__ = [
    "IncidentAction",
    "IncidentObservation",
    "IncidentState",
    "IncidentCommandEnvClient",
    "SREEnvClient",
    "__version__",
]
