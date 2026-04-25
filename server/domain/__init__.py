"""Domain package for the Incident Command Center environment.

This package contains the core business logic separated from the HTTP transport
layer. Keeping the domain logic pure (no FastAPI, no OpenEnv imports) lets us
unit-test it easily and reason about it independently.
"""

from server.domain.incidents import (
    Incident,
    IncidentLibrary,
    IncidentTemplate,
    build_incident_library,
)
from server.domain.reward import (
    RewardBreakdown,
    RewardEngine,
)
from server.domain.rng import SeededRNG
from server.domain.roles import (
    ALL_ACTIONS,
    ALL_ROLES,
    RolePermissions,
    check_actor_allowed,
)

__all__ = [
    "Incident",
    "IncidentLibrary",
    "IncidentTemplate",
    "build_incident_library",
    "RewardBreakdown",
    "RewardEngine",
    "SeededRNG",
    "ALL_ACTIONS",
    "ALL_ROLES",
    "RolePermissions",
    "check_actor_allowed",
]
