"""Baseline inference for the Incident Command Center environment.

Two policies are provided:

- `HeuristicCoordinator` — a deterministic state machine that exercises the
  full action space, picks role-appropriate actors, and consults the
  observation's `investigation_targets` and `playbook_hints` so the heuristic
  adapts to whatever the server is currently serving.
- `random_action` — a pure random baseline for comparison.

Running this script hits a deployed environment (local or Hugging Face Space)
and prints a structured trace the hackathon judges can follow.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
from typing import Dict, List, Optional

from client import IncidentCommandEnvClient
from models import IncidentAction, IncidentObservation

ENV_URL = os.getenv("ENV_URL", "http://127.0.0.1:8000")
BENCHMARK = "incident_command_center_env"
RANDOM_BASELINE = os.getenv("RANDOM_BASELINE", "false").lower() == "true"
# When set, run an LLM-backed policy (base or fine-tuned checkpoint) instead
# of the heuristic / random ones. Point this at a HF hub id or a local dir.
POLICY_MODEL = os.getenv("POLICY_MODEL", "").strip()


# ---------------------------------------------------------------------------
# Logging helpers (structured line format, easy to grep)
# ---------------------------------------------------------------------------


def log_start(task: str, env: str, policy: str) -> None:
    print(f"[START] task={task} env={env} policy={policy}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str] = None,
    components: Optional[Dict[str, float]] = None,
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    comp_val = "-" if not components else ",".join(f"{k}={v:+.2f}" for k, v in components.items())
    print(
        f"[STEP] step={step} action={action} reward={reward:+.2f} "
        f"done={done_val} error={error_val} components={comp_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:+.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:+.3f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Heuristic coordinator
# ---------------------------------------------------------------------------


class HeuristicCoordinator:
    """Deterministic multi-agent playbook agent.

    The state machine runs per incident and picks the correct specialist for
    each action so it never eats the wrong-actor penalty:

    1. Triage inspects logs + metrics using observation-provided targets.
    2. Investigator consults a KB article for the playbook.
    3. Ops Manager negotiates handoff to the owner the incident expects.
    4. Investigator applies a fix matched to inferred root cause.
    5. Ops Manager submits a postmortem when the incident marks it required.
    6. Ops Manager closes the incident with the inferred root cause.
    """

    def __init__(self) -> None:
        self._phase_by_incident: Dict[str, int] = {}
        self._root_cause_by_incident: Dict[str, str] = {}

    def select_action(self, observation: IncidentObservation) -> IncidentAction:
        incident_id = observation.incident_id
        phase = self._phase_by_incident.get(incident_id, 0)
        targets = observation.investigation_targets or {}
        log_targets = targets.get("logs", []) or []
        metric_targets = targets.get("metrics", []) or []
        kb_targets = targets.get("kb", []) or observation.playbook_hints

        # Haystack of all visible text we can mine for clues.
        haystack = " ".join(
            [
                observation.incident_title or "",
                observation.incident_description or "",
                observation.terminal_output or "",
                " ".join(observation.visible_signals or []),
            ]
        ).lower()

        if phase == 0 and log_targets:
            self._phase_by_incident[incident_id] = 1
            return IncidentAction(
                actor="triage_agent",
                action_type="inspect_logs",
                target=self._best_target(haystack, log_targets),
                reason="Initial triage: scan top logs for failure signature.",
            )

        if phase <= 1 and metric_targets:
            self._phase_by_incident[incident_id] = 2
            return IncidentAction(
                actor="triage_agent",
                action_type="inspect_metrics",
                target=self._best_target(haystack, metric_targets),
                reason="Correlate logs with dashboards.",
            )

        if phase <= 2 and kb_targets:
            self._phase_by_incident[incident_id] = 3
            return IncidentAction(
                actor="investigator_agent",
                action_type="consult_kb",
                target=self._best_target(haystack, list(kb_targets)),
                reason="Review runbook for candidate fix.",
            )

        if phase <= 3:
            self._phase_by_incident[incident_id] = 4
            owner = self._infer_owner(haystack, observation.customer_tier)
            return IncidentAction(
                actor="ops_manager_agent",
                action_type="negotiate_handoff",
                target=owner,
                reason="Route to accountable specialist.",
            )

        if phase <= 4:
            self._phase_by_incident[incident_id] = 5
            guess = self._infer_root_cause(haystack)
            self._root_cause_by_incident[incident_id] = guess
            return IncidentAction(
                actor="investigator_agent",
                action_type="apply_fix",
                resolution_summary=self._generate_fix_plan(guess),
                reason=f"Attempt mitigation for {guess}",
            )

        if phase <= 5 and observation.postmortem_required and not observation.postmortem_submitted:
            self._phase_by_incident[incident_id] = 6
            guess = self._root_cause_by_incident.get(
                incident_id, self._infer_root_cause(haystack)
            )
            return IncidentAction(
                actor="ops_manager_agent",
                action_type="submit_postmortem",
                postmortem_note=(
                    f"Incident {incident_id}: identified root cause {guess}. "
                    "Mitigation applied. Follow-up actions queued for "
                    "reliability review."
                ),
                reason="High-impact incident — postmortem required.",
            )

        guess = self._root_cause_by_incident.get(
            incident_id, self._infer_root_cause(haystack)
        )
        return IncidentAction(
            actor="ops_manager_agent",
            action_type="close_incident",
            root_cause=guess,
            resolution_summary=f"Closed with hypothesis {guess}.",
            confidence=0.75,
            reason="Enough evidence gathered to close incident.",
        )

    # -- helpers ------------------------------------------------------------

    def _best_target(self, haystack: str, candidates: List[str]) -> str:
        """Pick the candidate target whose tokens most overlap with the haystack."""
        best = candidates[0]
        best_score = -1
        for candidate in candidates:
            score = sum(1 for token in candidate.lower().split("-") if token in haystack)
            if score > best_score:
                best = candidate
                best_score = score
        return best

    def _infer_owner(self, haystack: str, tier: str) -> str:
        if tier == "enterprise":
            return "ops_manager_agent"
        if any(
            token in haystack
            for token in ["deploy", "rate", "sla", "rotation", "cert", "mtls"]
        ):
            return "ops_manager_agent"
        if any(
            token in haystack
            for token in ["schema", "export", "cache", "inventory", "search", "ranking"]
        ):
            return "investigator_agent"
        return "triage_agent"

    def _infer_root_cause(self, haystack: str) -> str:
        table = [
            (("redis", "pool"), "redis_connection_pool_exhausted"),
            (("jwt",), "jwt_clock_skew_mismatch"),
            (("token", "clock"), "jwt_clock_skew_mismatch"),
            (("spf",), "spf_record_misconfiguration"),
            (("cache", "invalidation"), "cache_invalidation_topic_lag"),
            (("timezone",), "timezone_normalization_bug"),
            (("offset",), "timezone_normalization_bug"),
            (("idempotency",), "idempotency_key_regression"),
            (("duplicate", "invoice"), "idempotency_key_regression"),
            (("mtls",), "mtls_cert_chain_mismatch"),
            (("certificate", "chain"), "mtls_cert_chain_mismatch"),
            (("feature", "flag"), "feature_flag_scope_misconfigured"),
            (("429",), "rate_limit_misconfigured_for_promo_segment"),
            (("promo",), "rate_limit_misconfigured_for_promo_segment"),
            (("schema", "drift"), "schema_version_drift"),
            (("schema", "mismatch"), "schema_version_drift"),
            (("dedupe",), "dedupe_rule_disabled"),
            (("alert", "storm"), "dedupe_rule_disabled"),
            (("out-of-order",), "event_ordering_race_condition"),
            (("oversell",), "event_ordering_race_condition"),
            (("deadlock",), "lock_escalation_on_reporting_view"),
            (("reporting", "lock"), "lock_escalation_on_reporting_view"),
        ]
        for tokens, guess in table:
            if all(tok in haystack for tok in tokens):
                return guess
        return "unknown"

    def _generate_fix_plan(self, root_cause: str) -> str:
        fixes = {
            "redis_connection_pool_exhausted": "increase redis pool and recycle stale connections",
            "jwt_clock_skew_mismatch": "sync clock tolerance and increase jwt leeway",
            "spf_record_misconfiguration": "fix spf record and align sending domain",
            "cache_invalidation_topic_lag": "scale invalidation consumer and replay partition 3",
            "timezone_normalization_bug": "patch timezone parser and use iana timezone map",
            "idempotency_key_regression": "restore idempotency guard and persist retry token first",
            "mtls_cert_chain_mismatch": "reissue certificate chain with full intermediate chain",
            "feature_flag_scope_misconfigured": "rollback feature flag and restrict experiment segment",
            "rate_limit_misconfigured_for_promo_segment": (
                "hotfix promo segment rate limits and enable exponential backoff"
            ),
            "schema_version_drift": "enforce schema negotiation and pin serializer to v11",
            "dedupe_rule_disabled": "restore dedupe rule and replay critical fingerprints",
            "event_ordering_race_condition": "enable sequence guards and quarantine out-of-order events",
            "lock_escalation_on_reporting_view": (
                "offload reporting to replica and schedule reporting off-peak"
            ),
        }
        return fixes.get(root_cause, "collect additional diagnostics and rollback last change")


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------


def random_action(observation: IncidentObservation) -> IncidentAction:
    action_type = random.choice(observation.available_actions or ["inspect_logs"])
    teams = observation.available_teams or [
        "triage_agent",
        "investigator_agent",
        "ops_manager_agent",
    ]
    actor = random.choice(teams)

    targets_pool: List[str] = []
    for _tool, values in (observation.investigation_targets or {}).items():
        targets_pool.extend(values)
    targets_pool.extend(
        ["payments-api", "auth-service", "dash-auth", "dash-redis", "kb-rate-limits"]
    )
    random_target = random.choice(targets_pool)

    return IncidentAction(
        actor=actor,  # type: ignore[arg-type]
        action_type=action_type,  # type: ignore[arg-type]
        target=random_target,
        root_cause="unknown",
        resolution_summary="random baseline action",
    )


# ---------------------------------------------------------------------------
# Episode driver
# ---------------------------------------------------------------------------


async def run_task(task_name: str, llm_policy=None) -> None:
    env = IncidentCommandEnvClient(base_url=ENV_URL).sync()
    if llm_policy is not None:
        policy_name = f"llm:{getattr(llm_policy, 'label', POLICY_MODEL)}"
    elif RANDOM_BASELINE:
        policy_name = "random_baseline"
    else:
        policy_name = "heuristic_coordinator"
    coordinator = HeuristicCoordinator()

    log_start(task=task_name, env=BENCHMARK, policy=policy_name)

    rewards: List[float] = []
    steps_taken = 0
    success = False
    score = 0.0

    try:
        res = env.reset(task_name=task_name)
        while not res.done:
            steps_taken += 1
            if llm_policy is not None:
                action = llm_policy.select_action(res.observation)
            elif RANDOM_BASELINE:
                action = random_action(res.observation)
            else:
                action = coordinator.select_action(res.observation)
            res = env.step(action)
            reward = float(res.reward or 0.0)
            rewards.append(reward)
            log_step(
                step=steps_taken,
                action=f"{action.actor}:{action.action_type}:{action.target or '-'}",
                reward=reward,
                done=res.done,
                components=getattr(res.observation, "reward_components", None),
            )

        score = sum(rewards) / len(rewards) if rewards else 0.0
        success = score > 0.1
    finally:
        try:
            env.close()
        except Exception:
            pass
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    llm_policy = None
    if POLICY_MODEL:
        from llm_policy import LLMPolicy

        llm_policy = LLMPolicy(POLICY_MODEL, label=POLICY_MODEL)

    for task in ["easy", "medium", "hard"]:
        asyncio.run(run_task(task, llm_policy=llm_policy))

    if llm_policy is not None:
        policy_label = f"llm:{POLICY_MODEL}"
    elif RANDOM_BASELINE:
        policy_label = "random_baseline"
    else:
        policy_label = "heuristic_coordinator"

    print(
        json.dumps(
            {
                "benchmark": BENCHMARK,
                "policy": policy_label,
                "env_url": ENV_URL,
            },
            indent=2,
        )
    )

    if llm_policy is not None:
        try:
            llm_policy.release()
        except Exception:
            pass


if __name__ == "__main__":
    main()
