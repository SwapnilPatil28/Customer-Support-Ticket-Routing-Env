"""Typed client for the Incident Command Center environment.

Built on OpenEnv's generic `EnvClient` so it exposes the full gym-style API
(`reset`, `step`, `state`, `close`) plus the rich typed fields added by this
environment (reward breakdowns, investigation targets, playbook hints, etc).
"""

from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

from models import IncidentAction, IncidentObservation, IncidentState


class IncidentCommandEnvClient(
    EnvClient[IncidentAction, IncidentObservation, IncidentState]
):
    """Client-side wrapper around the environment's HTTP contract."""

    def _step_payload(self, action: IncidentAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult:
        obs_data: Dict[str, Any] = payload.get("observation", {}) or {}
        observation = IncidentObservation.model_validate(obs_data)
        return StepResult(
            observation=observation,
            reward=float(payload.get("reward", 0.0)),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IncidentState:
        return IncidentState.model_validate(payload)


# Backward-compatible alias for older imports from round 1.
SREEnvClient = IncidentCommandEnvClient

__all__ = ["IncidentCommandEnvClient", "SREEnvClient"]
