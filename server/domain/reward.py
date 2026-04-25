"""Composable reward engine for the Incident Command Center environment.

The engine is intentionally *transparent*: every step produces a
`RewardBreakdown` listing the named components that contributed to the score.
This makes training curves interpretable, debugging tractable, and reward
shaping auditable — all table-stakes for enterprise use.

Design goals:

1. **Pure function** — the engine never mutates the environment; it returns
   a dataclass describing the contribution.
2. **Anti-gaming** — repeatedly querying the same evidence key yields a
   clue reward only once per incident.
3. **Business impact aware** — closure rewards and SLA penalties scale by
   customer tier and revenue impact, mirroring real SLA contracts.
4. **Composable** — you can extend this with additional components (for
   example, collaboration bonuses or cost-of-mitigation penalties) without
   touching the environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Tuple

from server.domain.incidents import Incident

# Reward component catalog --------------------------------------------------

STEP_COST_INVESTIGATION = -0.04
STEP_COST_KB = -0.03
STEP_COST_HANDOFF = -0.02
STEP_COST_APPLY_FIX = -0.02
STEP_COST_ESCALATE = -0.05
STEP_COST_ROLLBACK = -0.08
STEP_COST_POSTMORTEM = -0.01

WRONG_ACTOR_PENALTY = -0.08
REPEATED_LOOKUP_PENALTY = -0.02
INVALID_ACTION_PENALTY = -0.25

CLUE_REWARD = 0.12
CLUE_CAP_PER_INCIDENT = 3

HANDOFF_CORRECT_REWARD = 0.15
HANDOFF_WRONG_PENALTY = -0.10

MITIGATION_CORRECT_REWARD = 0.35
MITIGATION_WRONG_PENALTY = -0.30

CLOSURE_CORRECT_BASE = 0.80
CLOSURE_MITIGATION_BONUS = 0.30
CLOSURE_WRONG_PENALTY = -1.10
CLOSURE_UNDER_INVESTIGATED_PENALTY = -0.20

SPEED_BONUS_FAST = 0.20
SPEED_BONUS_OK = 0.10

POSTMORTEM_REQUIRED_BONUS = 0.12
POSTMORTEM_MISSING_PENALTY = -0.15

ESCALATION_NEEDED_REWARD = 0.10
ESCALATION_NOT_NEEDED_PENALTY = -0.10

# Business-impact multipliers for SLA / revenue-weighted penalties.
TIER_MULTIPLIER: Dict[str, float] = {
    "free": 0.6,
    "standard": 1.0,
    "premium": 1.4,
    "enterprise": 1.8,
}


@dataclass
class RewardBreakdown:
    """The structured result of scoring a single action."""

    components: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def add(self, name: str, value: float, note: str | None = None) -> None:
        if value == 0.0 and note is None:
            return
        self.components[name] = round(self.components.get(name, 0.0) + float(value), 6)
        if note is not None:
            self.notes.append(f"{name}: {note}")

    def total(self) -> float:
        return round(sum(self.components.values()), 6)

    def merge(self, other: "RewardBreakdown") -> None:
        for key, value in other.components.items():
            self.components[key] = round(self.components.get(key, 0.0) + float(value), 6)
        self.notes.extend(other.notes)

    def to_public_dict(self) -> Dict[str, float]:
        return dict(self.components)


class RewardEngine:
    """Stateless reward computations for the environment.

    Per-incident state (clues discovered, repeated lookups, mitigation flag)
    lives on the environment's `IncidentState` and is passed in explicitly.
    """

    def __init__(
        self,
        tier_multiplier: Dict[str, float] | None = None,
    ) -> None:
        self.tier_multiplier = dict(tier_multiplier or TIER_MULTIPLIER)

    # -- shared helpers ------------------------------------------------------

    def _tier_mult(self, incident: Incident) -> float:
        return self.tier_multiplier.get(incident.customer_tier, 1.0)

    def _has_matching_keyword(self, text: str, keywords: Iterable[str]) -> bool:
        text = text.lower()
        return any(k.lower() in text for k in keywords if k)

    # -- component calculators ----------------------------------------------

    def step_cost(self, action_type: str) -> RewardBreakdown:
        cost_map = {
            "inspect_logs": STEP_COST_INVESTIGATION,
            "inspect_metrics": STEP_COST_INVESTIGATION,
            "consult_kb": STEP_COST_KB,
            "negotiate_handoff": STEP_COST_HANDOFF,
            "apply_fix": STEP_COST_APPLY_FIX,
            "escalate": STEP_COST_ESCALATE,
            "rollback": STEP_COST_ROLLBACK,
            "submit_postmortem": STEP_COST_POSTMORTEM,
        }
        cost = cost_map.get(action_type, 0.0)
        br = RewardBreakdown()
        if cost:
            br.add("step_cost", cost, f"fixed step cost for {action_type}")
        return br

    def wrong_actor(self, actor: str, action_type: str, allowed: bool) -> RewardBreakdown:
        br = RewardBreakdown()
        if not allowed:
            br.add(
                "wrong_actor_penalty",
                WRONG_ACTOR_PENALTY,
                f"{actor} is not authorized for {action_type}",
            )
        return br

    def clue_reward(
        self,
        incident: Incident,
        signal_text: str,
        already_used_keys: Iterable[str],
        current_clue_count: int,
    ) -> Tuple[RewardBreakdown, bool, str | None]:
        """Award a one-time bonus when a lookup returns evidence keyed to the root cause.

        Returns `(breakdown, was_new_clue, matched_keyword)`.
        """
        br = RewardBreakdown()
        lowered = (signal_text or "").strip().lower()
        matched_keyword: str | None = None

        for keyword in incident.clue_keywords:
            if keyword.lower() in lowered:
                matched_keyword = keyword.lower()
                break

        is_new = False
        if matched_keyword is not None and matched_keyword not in already_used_keys:
            if current_clue_count < CLUE_CAP_PER_INCIDENT:
                br.add("clue_bonus", CLUE_REWARD, f"new clue: {matched_keyword}")
                is_new = True
        elif matched_keyword is not None:
            br.add(
                "repeated_lookup_penalty",
                REPEATED_LOOKUP_PENALTY,
                f"repeated clue for keyword '{matched_keyword}'",
            )
        return br, is_new, matched_keyword

    def handoff(self, incident: Incident, team: str) -> RewardBreakdown:
        br = RewardBreakdown()
        if team == incident.good_handoff:
            br.add("handoff_correct", HANDOFF_CORRECT_REWARD, f"correct handoff to {team}")
        else:
            br.add(
                "handoff_wrong",
                HANDOFF_WRONG_PENALTY,
                f"handoff to {team}; expected {incident.good_handoff}",
            )
        return br

    def mitigation(
        self,
        incident: Incident,
        resolution_summary: str,
    ) -> Tuple[RewardBreakdown, bool]:
        br = RewardBreakdown()
        text = (resolution_summary or "").lower()
        if not text:
            br.add(
                "mitigation_empty",
                MITIGATION_WRONG_PENALTY,
                "apply_fix without resolution_summary",
            )
            return br, False

        is_good = False
        for keyword_set in incident.accepted_fix_keywords:
            if all(token.lower() in text for token in keyword_set):
                is_good = True
                break

        if is_good:
            br.add("mitigation_correct", MITIGATION_CORRECT_REWARD, "accepted fix keywords matched")
        else:
            br.add("mitigation_wrong", MITIGATION_WRONG_PENALTY, "fix text did not match accepted keywords")
        return br, is_good

    def closure(
        self,
        incident: Incident,
        predicted_root_cause: str,
        mitigation_applied: bool,
        clues_count: int,
        steps_on_incident: int,
        postmortem_submitted: bool,
    ) -> Tuple[RewardBreakdown, bool]:
        br = RewardBreakdown()

        guess = (predicted_root_cause or "").strip().lower()
        candidates = [incident.root_cause.lower(), *[s.lower() for s in incident.root_cause_synonyms]]
        correct = guess in candidates or self._has_matching_keyword(guess, incident.clue_keywords)

        tier_mult = self._tier_mult(incident)

        if correct:
            base = CLOSURE_CORRECT_BASE * tier_mult
            br.add("closure_correct", base, f"root cause recognised (tier x{tier_mult})")

            if mitigation_applied:
                br.add(
                    "closure_mitigation_bonus",
                    CLOSURE_MITIGATION_BONUS,
                    "mitigation was previously applied",
                )
            elif incident.requires_mitigation:
                br.add(
                    "closure_no_mitigation",
                    -0.15,
                    "closed without applying required mitigation",
                )

            if clues_count < incident.required_investigations:
                br.add(
                    "closure_under_investigated",
                    CLOSURE_UNDER_INVESTIGATED_PENALTY,
                    f"closed with only {clues_count} clue(s); required {incident.required_investigations}",
                )

            if steps_on_incident <= 4:
                br.add("speed_bonus", SPEED_BONUS_FAST, "resolved under 4 steps")
            elif steps_on_incident <= 7:
                br.add("speed_bonus", SPEED_BONUS_OK, "resolved in 5-7 steps")

            if incident.postmortem_required:
                if postmortem_submitted:
                    br.add(
                        "postmortem_bonus",
                        POSTMORTEM_REQUIRED_BONUS,
                        "postmortem submitted for high-impact incident",
                    )
                else:
                    br.add(
                        "postmortem_missing",
                        POSTMORTEM_MISSING_PENALTY,
                        "high-impact incident closed without a postmortem",
                    )

        else:
            br.add(
                "closure_wrong",
                CLOSURE_WRONG_PENALTY * tier_mult,
                f"wrong root cause (tier x{tier_mult})",
            )

        return br, correct

    def escalation(self, incident: Incident, needed: bool) -> RewardBreakdown:
        br = RewardBreakdown()
        if needed:
            br.add(
                "escalation_needed",
                ESCALATION_NEEDED_REWARD,
                "escalation appropriate for incident scope",
            )
        else:
            br.add(
                "escalation_not_needed",
                ESCALATION_NOT_NEEDED_PENALTY,
                "escalation raised without justification",
            )
        return br

    def sla_exhaustion(self, incident: Incident) -> RewardBreakdown:
        """Penalty applied when SLA budget runs out while the incident is open."""
        br = RewardBreakdown()
        penalty = -1.2 * self._tier_mult(incident)
        br.add("sla_exhausted", penalty, "SLA budget reached zero")
        return br

    def budget_exhausted(self) -> RewardBreakdown:
        br = RewardBreakdown()
        br.add("budget_exhausted", -1.5, "investigation budget exhausted")
        return br

    def invalid_action(self, action_type: str) -> RewardBreakdown:
        br = RewardBreakdown()
        br.add(
            "invalid_action",
            INVALID_ACTION_PENALTY,
            f"unrecognised action_type '{action_type}'",
        )
        return br
