"""Incident Command Center environment (OpenEnv compliant).

This module wires the transport-agnostic domain logic (incidents, rewards,
role permissions) into OpenEnv's `Environment` contract.

Key design notes:

- **Deterministic**: every reset derives per-incident randomness from a
  seeded RNG so results are reproducible and debuggable.
- **Role-aware**: actions run by the wrong specialist incur a small
  penalty but are still allowed, mirroring real-world process friction.
- **Transparent rewards**: every step attaches a `reward_components` dict
  to the observation so agents, evaluators, and humans can see *why* a
  step was scored the way it was.
- **Safe serialization**: only wire types ever leave this module; the
  runtime `Incident` dataclass stays server-side.
"""

from __future__ import annotations

import logging
import uuid
from typing import Dict, List, Optional

from openenv.core.env_server import Environment

from models import IncidentAction, IncidentObservation, IncidentState
from server.config import EnvConfig
from server.domain import (
    Incident,
    IncidentLibrary,
    SeededRNG,
    build_incident_library,
    check_actor_allowed,
)
from server.domain.incidents import instantiate_incident
from server.domain.reward import RewardBreakdown, RewardEngine
from server.domain.roles import (
    ALL_ACTIONS,
    ALL_ROLES,
    allowed_actors_for,
    default_role_permissions,
)
from server.logging_utils import configure_logging, log_event

_LOG = logging.getLogger("icc.env")


class IncidentCommandCenterEnvironment(Environment):
    """Multi-agent incident response simulation.

    The environment maintains a sequential queue of incidents per task. A
    single action progresses the currently active incident. Closure advances
    to the next incident; the episode ends when all incidents are closed,
    when the investigation budget is exhausted, or when the global SLA
    minute budget hits zero.
    """

    def __init__(
        self,
        config: Optional[EnvConfig] = None,
        library: Optional[IncidentLibrary] = None,
    ) -> None:
        super().__init__()
        self.config = config or EnvConfig.from_env()
        self.library = library or build_incident_library()
        self.reward_engine = RewardEngine()
        self.permissions = default_role_permissions()

        configure_logging(
            level=self.config.log_level,
            structured=self.config.structured_logging,
        )
        log_event(
            _LOG,
            "environment_boot",
            env=self.config.name,
            version=self.config.version,
            tasks=self.library.tasks(),
            incidents=self.library.total_incidents(),
        )

        # Runtime containers — populated by `reset`.
        self._incidents: List[Incident] = []
        self._episode_seed: int = self.config.default_seed
        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            task_id="easy",
            seed=self._episode_seed,
            version=self.config.version,
        )

    # ------------------------------------------------------------------
    # OpenEnv Environment contract
    # ------------------------------------------------------------------

    def reset(
        self,
        task_name: str = "easy",
        seed: Optional[int] = None,
    ) -> IncidentObservation:
        """Prepare a new episode.

        Parameters
        ----------
        task_name:
            One of `easy`, `medium`, `hard`. Unknown task names fall back to
            `easy` rather than raising, to maximize client robustness.
        seed:
            Optional seed for deterministic incident ordering and noise.
            Falls back to `EnvConfig.default_seed` when omitted.
        """
        selected = task_name if task_name in self.library.tasks() else "easy"
        self._episode_seed = int(seed) if seed is not None else self.config.default_seed

        rng = SeededRNG(self._episode_seed).child(f"task:{selected}")
        templates = self.library.templates_for(selected)
        self._incidents = [instantiate_incident(t, rng) for t in templates]

        self._state = IncidentState(
            episode_id=str(uuid.uuid4()),
            task_id=selected,
            seed=self._episode_seed,
            version=self.config.version,
            current_incident_index=0,
            budget_remaining=self.config.budget_for(selected),
            sla_minutes_remaining=self.config.sla_for(selected),
        )

        log_event(
            _LOG,
            "episode_start",
            episode_id=self._state.episode_id,
            task=selected,
            seed=self._episode_seed,
            incidents=[i.id for i in self._incidents],
        )

        return self._observation(
            reward=0.0,
            reward_components={},
            notes=["episode_started"],
            terminal_output=(
                "Incident Command Center initialized. "
                "Coordinate triage_agent, investigator_agent and "
                "ops_manager_agent to resolve the incident queue."
            ),
            done=False,
        )

    def step(self, action: IncidentAction) -> IncidentObservation:
        """Advance one turn.

        Returns an observation whose `reward_components` dict explains how
        the step reward was composed.
        """
        self._state.step_count += 1
        self._state.sla_minutes_remaining = max(
            0, self._state.sla_minutes_remaining - self.config.sla_tick_minutes
        )
        self._state.budget_remaining -= 1

        # Episode-level terminations -------------------------------------
        if self._state.current_incident_index >= len(self._incidents):
            return self._terminate(
                reason="already_completed",
                reward=0.0,
                breakdown=RewardBreakdown(),
                terminal_output="All incidents already resolved.",
            )

        if self._state.budget_remaining < 0:
            breakdown = self.reward_engine.budget_exhausted()
            return self._terminate(
                reason="budget_exhausted",
                reward=breakdown.total(),
                breakdown=breakdown,
                terminal_output="Episode terminated: investigation budget exhausted.",
            )

        if self._state.sla_minutes_remaining <= 0:
            current = self._incidents[self._state.current_incident_index]
            breakdown = self.reward_engine.sla_exhaustion(current)
            self._state.incidents_failed += 1
            return self._terminate(
                reason="sla_exhausted",
                reward=breakdown.total(),
                breakdown=breakdown,
                terminal_output="Episode terminated: global SLA budget reached zero.",
            )

        # Per-turn scoring -----------------------------------------------
        incident = self._incidents[self._state.current_incident_index]
        incident_id = incident.id
        self._state.per_incident_steps[incident_id] = (
            self._state.per_incident_steps.get(incident_id, 0) + 1
        )
        trace_line = f"{action.actor}:{action.action_type}:{action.target or '-'}"
        self._state.action_trace.append(trace_line)

        breakdown = RewardBreakdown()
        breakdown.merge(self.reward_engine.step_cost(action.action_type))

        actor_allowed = check_actor_allowed(
            action.actor, action.action_type, self.permissions
        )
        breakdown.merge(
            self.reward_engine.wrong_actor(action.actor, action.action_type, actor_allowed)
        )

        terminal_output = ""
        episode_done = False

        handler = self._handlers().get(action.action_type)
        if handler is None:
            breakdown.merge(self.reward_engine.invalid_action(action.action_type))
            terminal_output = f"Unsupported action_type: {action.action_type}"
        else:
            terminal_output, episode_done = handler(action, incident, breakdown)

        reward = breakdown.total()
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + reward, 6
        )
        if len(self._state.reward_trace) < self.config.max_reward_trace_len:
            self._state.reward_trace.append(breakdown.to_public_dict())

        log_event(
            _LOG,
            "step",
            episode_id=self._state.episode_id,
            action=trace_line,
            reward=reward,
            components=breakdown.to_public_dict(),
            cumulative_reward=self._state.cumulative_reward,
            budget_remaining=self._state.budget_remaining,
            sla_minutes_remaining=self._state.sla_minutes_remaining,
        )

        return self._observation(
            reward=reward,
            reward_components=breakdown.to_public_dict(),
            notes=breakdown.notes,
            terminal_output=terminal_output,
            done=episode_done,
        )

    @property
    def state(self) -> IncidentState:
        return self._state

    # ------------------------------------------------------------------
    # Action handlers
    # ------------------------------------------------------------------

    def _handlers(self):
        return {
            "inspect_logs": self._handle_inspect_logs,
            "inspect_metrics": self._handle_inspect_metrics,
            "consult_kb": self._handle_consult_kb,
            "negotiate_handoff": self._handle_handoff,
            "apply_fix": self._handle_apply_fix,
            "escalate": self._handle_escalate,
            "rollback": self._handle_rollback,
            "submit_postmortem": self._handle_postmortem,
            "close_incident": self._handle_close,
        }

    # -- inspection actions --------------------------------------------

    def _handle_inspect_logs(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        lookup = (action.target or "").strip()
        text = incident.logs.get(lookup, f"No logs found for target '{lookup}'.")
        self._award_clue(incident, lookup, text, breakdown, scope="logs")
        return text, False

    def _handle_inspect_metrics(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        lookup = (action.target or "").strip()
        text = incident.metrics.get(lookup, f"No metrics found for target '{lookup}'.")
        self._award_clue(incident, lookup, text, breakdown, scope="metrics")
        return text, False

    def _handle_consult_kb(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        lookup = (action.target or "").strip()
        text = incident.kb.get(lookup, f"No KB article found for key '{lookup}'.")
        self._award_clue(incident, lookup, text, breakdown, scope="kb")
        return text, False

    def _award_clue(
        self,
        incident: Incident,
        lookup_key: str,
        text: str,
        breakdown: RewardBreakdown,
        scope: str,
    ) -> None:
        scoped_key = f"{scope}:{lookup_key}"
        clue_breakdown, was_new, _matched = self.reward_engine.clue_reward(
            incident,
            text,
            already_used_keys=self._state.clue_keywords_used,
            current_clue_count=len([k for k in self._state.clue_keywords_used]),
        )
        breakdown.merge(clue_breakdown)
        if was_new and _matched is not None:
            self._state.clue_keywords_used.append(_matched)
        if scoped_key not in self._state.investigation_keys_used:
            self._state.investigation_keys_used.append(scoped_key)

    # -- coordination actions ------------------------------------------

    def _handle_handoff(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        team = (action.target or "").strip()
        self._state.handoff_history.append(team)
        breakdown.merge(self.reward_engine.handoff(incident, team))
        if team == incident.good_handoff:
            text = f"Handoff accepted by {team}. Hypothesis confidence increased."
        else:
            text = (
                f"Handoff to {team} introduced delay. "
                f"Expected owner: {incident.good_handoff}."
            )
        return text, False

    def _handle_apply_fix(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        mitigation_breakdown, is_good = self.reward_engine.mitigation(
            incident, action.resolution_summary or ""
        )
        breakdown.merge(mitigation_breakdown)
        if is_good:
            self._state.mitigation_applied = True
            text = "Mitigation accepted. Error rate is stabilizing."
        else:
            text = "Applied mitigation appears ineffective; diagnostics continue."
        return text, False

    def _handle_escalate(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        scope_limit = (
            incident.template.affected_users_estimate >= 50_000
            or incident.template.revenue_impact_usd_per_min >= 800
            or incident.template.postmortem_required
        )
        breakdown.merge(self.reward_engine.escalation(incident, scope_limit))
        if scope_limit:
            text = "Escalation paged: leadership channel opened; war room requested."
        else:
            text = "Escalation declined: impact below paging threshold."
        return text, False

    def _handle_rollback(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        text = (action.resolution_summary or "").lower()
        if any(
            token in text
            for keyword_set in incident.accepted_fix_keywords
            for token in keyword_set
            if "rollback" in token or "roll back" in token
        ):
            breakdown.add("rollback_effective", 0.20, "rollback aligned with playbook")
            self._state.mitigation_applied = True
            output = "Rollback applied: change reverted to last known good."
        else:
            breakdown.add("rollback_ineffective", -0.15, "rollback did not match accepted fix")
            output = "Rollback attempted but incident not stabilized."
        return output, False

    def _handle_postmortem(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        note = (action.postmortem_note or "").strip()
        if not note:
            breakdown.add(
                "postmortem_empty", -0.10, "submit_postmortem without postmortem_note"
            )
            return "Postmortem rejected: note missing.", False

        self._state.postmortem_submitted = True
        breakdown.add(
            "postmortem_logged",
            0.05,
            f"postmortem stored ({len(note)} chars)",
        )
        return "Postmortem filed for review.", False

    # -- closure --------------------------------------------------------

    def _handle_close(
        self, action: IncidentAction, incident: Incident, breakdown: RewardBreakdown
    ) -> tuple[str, bool]:
        guess = (action.root_cause or "").strip()
        steps = self._state.per_incident_steps.get(incident.id, 1)
        clues = len(self._state.clue_keywords_used)
        postmortem = self._state.postmortem_submitted

        closure_breakdown, correct = self.reward_engine.closure(
            incident,
            predicted_root_cause=guess,
            mitigation_applied=self._state.mitigation_applied,
            clues_count=clues,
            steps_on_incident=steps,
            postmortem_submitted=postmortem,
        )
        breakdown.merge(closure_breakdown)

        if correct:
            self._state.incidents_resolved += 1
            outcome_text = (
                "Incident resolved successfully. "
                f"Root cause acknowledged: {incident.root_cause}."
            )
        else:
            self._state.incidents_failed += 1
            outcome_text = (
                "Incident closure rejected by postmortem checker. "
                f"Prediction '{guess or 'unknown'}' did not match ground truth."
            )

        self._advance_incident()
        episode_done = self._state.current_incident_index >= len(self._incidents)
        if episode_done:
            outcome_text += " All assigned incidents processed."
        else:
            outcome_text += f" Next incident: {self._incidents[self._state.current_incident_index].id}."
        return outcome_text, episode_done

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _advance_incident(self) -> None:
        self._state.current_incident_index += 1
        self._state.mitigation_applied = False
        self._state.postmortem_submitted = False
        self._state.clue_keywords_used = []
        self._state.investigation_keys_used = []

    def _terminate(
        self,
        reason: str,
        reward: float,
        breakdown: RewardBreakdown,
        terminal_output: str,
    ) -> IncidentObservation:
        self._state.terminated_reason = reason
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + reward, 6
        )
        log_event(
            _LOG,
            "episode_terminate",
            episode_id=self._state.episode_id,
            reason=reason,
            cumulative_reward=self._state.cumulative_reward,
            incidents_resolved=self._state.incidents_resolved,
            incidents_failed=self._state.incidents_failed,
        )
        return IncidentObservation(
            done=True,
            reward=reward,
            incident_id="EOF",
            incident_title="Episode ended",
            incident_description="No further actions accepted.",
            incident_category="",
            incident_difficulty=self._state.task_id,
            customer_tier="standard",
            affected_users_estimate=0,
            revenue_impact_usd_per_min=0,
            postmortem_required=False,
            available_actions=[],
            available_teams=list(ALL_ROLES),
            allowed_actors_by_action={},
            visible_signals=[],
            investigation_targets={},
            playbook_hints=[],
            terminal_output=terminal_output,
            budget_remaining=max(self._state.budget_remaining, 0),
            sla_minutes_remaining=self._state.sla_minutes_remaining,
            incidents_remaining=max(
                len(self._incidents) - self._state.current_incident_index, 0
            ),
            episode_step=self._state.step_count,
            incident_step=0,
            clues_found=len(self._state.clue_keywords_used),
            mitigation_applied=self._state.mitigation_applied,
            postmortem_submitted=self._state.postmortem_submitted,
            reward_components=breakdown.to_public_dict(),
            last_action_notes=breakdown.notes,
        )

    def _observation(
        self,
        reward: float,
        reward_components: Dict[str, float],
        notes: List[str],
        terminal_output: str,
        done: bool,
    ) -> IncidentObservation:
        if done or self._state.current_incident_index >= len(self._incidents):
            return IncidentObservation(
                done=True,
                reward=reward,
                incident_id="EOF",
                incident_title="All incidents completed",
                incident_description="Episode ended.",
                incident_category="",
                incident_difficulty=self._state.task_id,
                customer_tier="standard",
                affected_users_estimate=0,
                revenue_impact_usd_per_min=0,
                postmortem_required=False,
                available_actions=[],
                available_teams=list(ALL_ROLES),
                allowed_actors_by_action={},
                visible_signals=[],
                investigation_targets={},
                playbook_hints=[],
                terminal_output=terminal_output,
                budget_remaining=max(self._state.budget_remaining, 0),
                sla_minutes_remaining=self._state.sla_minutes_remaining,
                incidents_remaining=0,
                episode_step=self._state.step_count,
                incident_step=0,
                clues_found=len(self._state.clue_keywords_used),
                mitigation_applied=self._state.mitigation_applied,
                postmortem_submitted=self._state.postmortem_submitted,
                reward_components=reward_components,
                last_action_notes=notes,
            )

        incident = self._incidents[self._state.current_incident_index]
        investigation_targets = {
            "logs": list(incident.logs.keys()),
            "metrics": list(incident.metrics.keys()),
            "kb": list(incident.kb.keys()),
        }
        allowed_actors_by_action = {
            action_type: list(allowed_actors_for(action_type, self.permissions))
            for action_type in ALL_ACTIONS
        }
        incident_step = self._state.per_incident_steps.get(incident.id, 0)

        return IncidentObservation(
            done=False,
            reward=reward,
            incident_id=incident.id,
            incident_title=incident.title,
            incident_description=incident.description,
            incident_category=incident.template.category,
            incident_difficulty=incident.template.difficulty,
            customer_tier=incident.customer_tier,
            affected_users_estimate=incident.affected_users_estimate,
            revenue_impact_usd_per_min=incident.revenue_impact_usd_per_min,
            postmortem_required=incident.postmortem_required,
            available_actions=list(ALL_ACTIONS),
            available_teams=list(ALL_ROLES),
            allowed_actors_by_action=allowed_actors_by_action,
            visible_signals=list(incident.signals),
            investigation_targets=investigation_targets,
            playbook_hints=list(incident.playbook_hints),
            terminal_output=terminal_output,
            budget_remaining=max(self._state.budget_remaining, 0),
            sla_minutes_remaining=self._state.sla_minutes_remaining,
            incidents_remaining=len(self._incidents) - self._state.current_incident_index,
            episode_step=self._state.step_count,
            incident_step=incident_step,
            clues_found=len(self._state.clue_keywords_used),
            mitigation_applied=self._state.mitigation_applied,
            postmortem_submitted=self._state.postmortem_submitted,
            reward_components=reward_components,
            last_action_notes=notes,
        )
