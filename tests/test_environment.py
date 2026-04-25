"""Environment-level integration tests (require openenv installed)."""

from __future__ import annotations

import importlib

import pytest

openenv = pytest.importorskip(
    "openenv.core.env_server",
    reason="openenv-core not installed; skipping environment tests.",
)

environment_module = importlib.import_module("server.environment")
models_module = importlib.import_module("models")

IncidentCommandCenterEnvironment = environment_module.IncidentCommandCenterEnvironment
IncidentAction = models_module.IncidentAction


def test_reset_returns_valid_observation() -> None:
    env = IncidentCommandCenterEnvironment()
    obs = env.reset(task_name="easy", seed=123)
    assert obs.done is False
    assert obs.incident_id
    assert obs.budget_remaining > 0
    assert obs.sla_minutes_remaining > 0
    assert "inspect_logs" in obs.available_actions
    assert obs.investigation_targets
    assert obs.customer_tier in {"free", "standard", "premium", "enterprise"}


def test_reset_is_seeded_deterministic() -> None:
    env = IncidentCommandCenterEnvironment()
    a = env.reset(task_name="medium", seed=7)
    b = env.reset(task_name="medium", seed=7)
    assert a.incident_id == b.incident_id
    assert a.investigation_targets == b.investigation_targets


def test_inspect_logs_step_returns_reward_components() -> None:
    env = IncidentCommandCenterEnvironment()
    obs = env.reset(task_name="easy", seed=1)
    log_target = next(iter(obs.investigation_targets.get("logs", []) or [""]))
    result = env.step(
        IncidentAction(
            actor="triage_agent",
            action_type="inspect_logs",
            target=log_target or "payments-api",
        )
    )
    assert isinstance(result.reward_components, dict)
    assert "step_cost" in result.reward_components


def test_wrong_actor_incurs_penalty() -> None:
    env = IncidentCommandCenterEnvironment()
    env.reset(task_name="easy", seed=1)
    res = env.step(
        IncidentAction(
            actor="triage_agent",
            action_type="close_incident",
            root_cause="unknown",
        )
    )
    assert res.reward_components.get("wrong_actor_penalty", 0.0) < 0


def test_budget_exhaustion_terminates_episode() -> None:
    env = IncidentCommandCenterEnvironment()
    env.reset(task_name="easy", seed=2)
    done = False
    steps = 0
    while not done and steps < 200:
        res = env.step(
            IncidentAction(actor="triage_agent", action_type="inspect_logs", target="foo")
        )
        done = bool(res.done)
        steps += 1
    assert done, "Episode should terminate when budget/SLA is exhausted"


def test_close_correct_root_cause_awards_positive_reward() -> None:
    env = IncidentCommandCenterEnvironment()
    obs = env.reset(task_name="easy", seed=3)
    incident = env._incidents[env.state.current_incident_index]  # type: ignore[attr-defined]
    expected_root_cause = incident.root_cause

    env.step(
        IncidentAction(
            actor="investigator_agent",
            action_type="apply_fix",
            resolution_summary=" ".join(incident.accepted_fix_keywords[0]),
        )
    )
    res = env.step(
        IncidentAction(
            actor="ops_manager_agent",
            action_type="close_incident",
            root_cause=expected_root_cause,
        )
    )
    assert any(v > 0 for v in res.reward_components.values()), res.reward_components
