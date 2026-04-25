"""Role-based permissions for the three specialist agents.

In a real incident-response organization different roles have different
authority. We encode that so the environment can reward or penalize actions
taken by the wrong specialist, and so downstream policies learn realistic
coordination patterns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Set

ALL_ROLES: tuple[str, ...] = (
    "triage_agent",
    "investigator_agent",
    "ops_manager_agent",
)

ALL_ACTIONS: tuple[str, ...] = (
    "inspect_logs",
    "inspect_metrics",
    "consult_kb",
    "negotiate_handoff",
    "apply_fix",
    "close_incident",
    "escalate",
    "rollback",
    "submit_postmortem",
)


@dataclass(frozen=True)
class RolePermissions:
    """Allowed actions per role and a list of role-gated actions."""

    allowed: Dict[str, Set[str]]

    def is_allowed(self, actor: str, action_type: str) -> bool:
        allowed_set = self.allowed.get(actor, set())
        return action_type in allowed_set

    def allowed_actions(self, actor: str) -> Set[str]:
        return set(self.allowed.get(actor, set()))


def default_role_permissions() -> RolePermissions:
    """Default policy used by the environment.

    - triage_agent: first-line observability + initial handoff
    - investigator_agent: deep diagnostics, knowledge base, fix proposals
    - ops_manager_agent: coordination actions (handoff, escalate, rollback),
      and is the only role authorized to close an incident or submit a
      postmortem.
    """
    allowed: Dict[str, Set[str]] = {
        "triage_agent": {
            "inspect_logs",
            "inspect_metrics",
            "consult_kb",
            "negotiate_handoff",
        },
        "investigator_agent": {
            "inspect_logs",
            "inspect_metrics",
            "consult_kb",
            "apply_fix",
            "rollback",
        },
        "ops_manager_agent": {
            "negotiate_handoff",
            "escalate",
            "rollback",
            "close_incident",
            "submit_postmortem",
        },
    }
    return RolePermissions(allowed=allowed)


def check_actor_allowed(
    actor: str, action_type: str, permissions: RolePermissions | None = None
) -> bool:
    """Return True if `actor` is permitted to run `action_type`.

    Returns False for unknown roles or actions so the caller can apply the
    policy's wrong-actor penalty uniformly.
    """
    if actor not in ALL_ROLES or action_type not in ALL_ACTIONS:
        return False
    permissions = permissions or default_role_permissions()
    return permissions.is_allowed(actor, action_type)


def allowed_actors_for(action_type: str, permissions: RolePermissions | None = None) -> Iterable[str]:
    permissions = permissions or default_role_permissions()
    return tuple(
        actor for actor in ALL_ROLES if permissions.is_allowed(actor, action_type)
    )
