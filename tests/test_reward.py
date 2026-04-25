"""Reward engine invariants."""

from __future__ import annotations

from server.domain.incidents import build_incident_library, instantiate_incident
from server.domain.reward import (
    CLOSURE_CORRECT_BASE,
    CLUE_CAP_PER_INCIDENT,
    CLUE_REWARD,
    HANDOFF_CORRECT_REWARD,
    MITIGATION_CORRECT_REWARD,
    RewardEngine,
)
from server.domain.rng import SeededRNG


LIBRARY = build_incident_library()


def _sample_incident(task: str = "easy", idx: int = 0):
    template = LIBRARY.templates_for(task)[idx]
    return instantiate_incident(template, SeededRNG(1))


def test_step_cost_applied_for_inspect() -> None:
    engine = RewardEngine()
    br = engine.step_cost("inspect_logs")
    assert br.total() < 0


def test_wrong_actor_penalty_applied_only_when_disallowed() -> None:
    engine = RewardEngine()
    disallowed = engine.wrong_actor("triage_agent", "close_incident", allowed=False)
    allowed = engine.wrong_actor("triage_agent", "inspect_logs", allowed=True)
    assert disallowed.total() < 0
    assert allowed.total() == 0.0


def test_correct_handoff_is_positive() -> None:
    engine = RewardEngine()
    incident = _sample_incident()
    br = engine.handoff(incident, incident.good_handoff)
    assert br.total() >= HANDOFF_CORRECT_REWARD


def test_mitigation_keyword_match() -> None:
    engine = RewardEngine()
    incident = _sample_incident("easy", 0)  # redis pool
    br, ok = engine.mitigation(incident, "increase redis pool size and recycle connections")
    assert ok
    assert br.total() >= MITIGATION_CORRECT_REWARD

    bad_br, bad_ok = engine.mitigation(incident, "delete caches randomly")
    assert not bad_ok
    assert bad_br.total() < 0


def test_clue_reward_capped_and_deduped() -> None:
    engine = RewardEngine()
    incident = _sample_incident("easy", 0)
    used: list[str] = []
    total_new_clue_rewards = 0.0

    for _ in range(10):
        br, was_new, matched = engine.clue_reward(
            incident,
            "redis pool exhaustion in checkout-worker",
            already_used_keys=used,
            current_clue_count=len(used),
        )
        if was_new and matched is not None:
            used.append(matched)
            total_new_clue_rewards += br.total()

    assert len(used) <= CLUE_CAP_PER_INCIDENT
    assert total_new_clue_rewards <= CLUE_CAP_PER_INCIDENT * CLUE_REWARD + 1e-6


def test_closure_correct_scales_with_tier() -> None:
    engine = RewardEngine()
    incident = _sample_incident("medium", 0)  # premium tier
    br, correct = engine.closure(
        incident,
        predicted_root_cause=incident.root_cause,
        mitigation_applied=True,
        clues_count=incident.required_investigations,
        steps_on_incident=3,
        postmortem_submitted=incident.postmortem_required,
    )
    assert correct
    assert br.total() >= CLOSURE_CORRECT_BASE


def test_closure_wrong_is_negative() -> None:
    engine = RewardEngine()
    incident = _sample_incident("easy", 0)
    br, correct = engine.closure(
        incident,
        predicted_root_cause="completely unrelated guess",
        mitigation_applied=False,
        clues_count=0,
        steps_on_incident=1,
        postmortem_submitted=False,
    )
    assert not correct
    assert br.total() < 0
