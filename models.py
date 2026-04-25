"""Pydantic schemas for the Incident Command Center environment.

These are the wire types shared by the HTTP server and the client. They are
designed to be:

- **Forwards-compatible**: new observation fields have default values so old
  clients keep working.
- **Strict on the server**: every action field has a validator that ensures
  the server never receives malformed data.
- **Self-documenting**: every field has a `description` that renders into
  the OpenAPI schema at `/docs`.
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import ConfigDict, Field, field_validator

# ----- Constants shared with server code -----------------------------------

ActionType = Literal[
    "inspect_logs",
    "inspect_metrics",
    "consult_kb",
    "negotiate_handoff",
    "apply_fix",
    "close_incident",
    "escalate",
    "rollback",
    "submit_postmortem",
]

RoleName = Literal[
    "triage_agent",
    "investigator_agent",
    "ops_manager_agent",
]

CustomerTier = Literal["free", "standard", "premium", "enterprise"]


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------


class IncidentAction(Action):
    """Structured action payload accepted by the environment.

    Validators reject obviously malformed input (empty targets, invalid roles)
    and trim whitespace so training-time and inference-time JSON is normalised
    identically.
    """

    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    action_type: ActionType = Field(
        ..., description="Selected action from the supported action space."
    )
    actor: RoleName = Field(
        "triage_agent",
        description="Specialist role acting in the environment during this turn.",
    )
    target: Optional[str] = Field(
        None,
        description=(
            "Service id for inspect_logs/inspect_metrics, KB id for consult_kb, "
            "team name for negotiate_handoff/escalate."
        ),
    )
    root_cause: Optional[str] = Field(
        None, description="Predicted root cause for close_incident."
    )
    resolution_summary: Optional[str] = Field(
        None,
        description="Human-readable fix summary for apply_fix, rollback and close_incident.",
    )
    postmortem_note: Optional[str] = Field(
        None,
        description="Postmortem text for submit_postmortem actions.",
    )
    confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Optional self-reported confidence of the agent in this action.",
    )
    reason: Optional[str] = Field(
        None,
        description="Optional free-text rationale for audit logs and traceability.",
    )

    @field_validator("target", "root_cause", "resolution_summary", "postmortem_note", "reason")
    @classmethod
    def _empty_string_to_none(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------


class IncidentObservation(Observation):
    """Observation returned to the agent after each action.

    All newly added fields carry defaults so older clients continue to
    deserialize this type correctly.
    """

    model_config = ConfigDict(extra="ignore")

    incident_id: str = ""
    incident_title: str = ""
    incident_description: str = ""
    incident_category: str = ""
    incident_difficulty: str = "easy"

    customer_tier: CustomerTier = "standard"
    affected_users_estimate: int = 0
    revenue_impact_usd_per_min: int = 0
    postmortem_required: bool = False

    available_actions: List[str] = Field(default_factory=list)
    available_teams: List[str] = Field(default_factory=list)
    allowed_actors_by_action: Dict[str, List[str]] = Field(default_factory=dict)

    visible_signals: List[str] = Field(default_factory=list)
    investigation_targets: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Per-tool list of known investigation ids (logs/metrics/kb).",
    )
    playbook_hints: List[str] = Field(default_factory=list)

    terminal_output: str = ""
    budget_remaining: int = 0
    sla_minutes_remaining: int = 0
    incidents_remaining: int = 0
    episode_step: int = 0
    incident_step: int = 0
    clues_found: int = 0
    mitigation_applied: bool = False
    postmortem_submitted: bool = False

    reward_components: Dict[str, float] = Field(default_factory=dict)
    last_action_notes: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class IncidentState(State):
    """Full environment state exposed at `/state` for observability."""

    model_config = ConfigDict(extra="ignore")

    task_id: str = "easy"
    seed: int = 0
    version: str = "3.0.0"

    current_incident_index: int = 0
    incidents_resolved: int = 0
    incidents_failed: int = 0

    budget_remaining: int = 0
    sla_minutes_remaining: int = 0
    cumulative_reward: float = 0.0

    mitigation_applied: bool = False
    postmortem_submitted: bool = False
    clue_keywords_used: List[str] = Field(default_factory=list)
    investigation_keys_used: List[str] = Field(default_factory=list)
    handoff_history: List[str] = Field(default_factory=list)
    action_trace: List[str] = Field(default_factory=list)
    per_incident_steps: Dict[str, int] = Field(default_factory=dict)
    reward_trace: List[Dict[str, float]] = Field(default_factory=list)
    terminated_reason: Optional[str] = None
