"""Invariants for the incident catalog.

These tests are pure-domain (no OpenEnv, no FastAPI) so they run on any
Python environment with pytest and pydantic installed.
"""

from __future__ import annotations

import pytest

from server.domain.incidents import build_incident_library, instantiate_incident
from server.domain.rng import SeededRNG
from server.domain.roles import ALL_ROLES


LIBRARY = build_incident_library()


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_library_has_incidents(task: str) -> None:
    templates = LIBRARY.templates_for(task)
    assert len(templates) >= 3, f"Task {task} must have at least 3 incidents"


@pytest.mark.parametrize("task", ["easy", "medium", "hard"])
def test_incident_template_completeness(task: str) -> None:
    for template in LIBRARY.templates_for(task):
        assert template.id
        assert template.title
        assert template.root_cause
        assert template.clue_keywords, f"{template.id} needs clue keywords"
        assert template.signals, f"{template.id} needs visible signals"
        assert template.logs, f"{template.id} needs at least one log"
        assert template.metrics, f"{template.id} needs at least one metric"
        assert template.kb, f"{template.id} needs at least one KB entry"
        assert template.good_handoff in ALL_ROLES, f"{template.id} handoff invalid"
        assert template.accepted_fix_keywords, f"{template.id} needs fix keywords"
        assert template.customer_tier in {"free", "standard", "premium", "enterprise"}


def test_unique_incident_ids() -> None:
    ids = [
        template.id
        for task in LIBRARY.tasks()
        for template in LIBRARY.templates_for(task)
    ]
    assert len(ids) == len(set(ids)), "Incident ids must be globally unique"


def test_instantiate_is_deterministic() -> None:
    rng_a = SeededRNG(42)
    rng_b = SeededRNG(42)
    template = LIBRARY.templates_for("easy")[0]
    inc_a = instantiate_incident(template, rng_a)
    inc_b = instantiate_incident(template, rng_b)
    assert list(inc_a.logs.keys()) == list(inc_b.logs.keys())
    assert list(inc_a.metrics.keys()) == list(inc_b.metrics.keys())
