"""Incident domain model and enterprise-grade library.

Each incident template captures a realistic operational scenario:

- Partial signals the triage agent can see immediately.
- Noisy logs/metrics with **red herrings** to discourage shortcutting.
- Multiple synonymous root-cause strings and accepted-fix keywords, so the
  agent must surface the right idea rather than the exact literal string.
- Customer tier, affected users and revenue-impact metadata so the reward
  engine can scale penalties by business impact (premium tier SLA violations
  hurt more than free-tier ones).
- Playbook hints (KB articles) for the Investigator agent.

The catalog is intentionally written in plain Python so it is easy to review,
edit and extend without touching the reward logic or the HTTP layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Tuple

from server.domain.rng import SeededRNG


CustomerTier = str  # one of: "free", "standard", "premium", "enterprise"


@dataclass(frozen=True)
class IncidentTemplate:
    """Static description of an incident scenario."""

    id: str
    title: str
    description: str
    category: str
    difficulty: str

    root_cause: str
    root_cause_synonyms: Tuple[str, ...]
    clue_keywords: Tuple[str, ...]

    signals: Tuple[str, ...]
    logs: Mapping[str, str]
    metrics: Mapping[str, str]
    kb: Mapping[str, str]
    red_herring_logs: Mapping[str, str] = field(default_factory=dict)
    red_herring_metrics: Mapping[str, str] = field(default_factory=dict)

    good_handoff: str = "investigator_agent"
    accepted_fix_keywords: Tuple[Tuple[str, ...], ...] = ()
    required_investigations: int = 2

    customer_tier: CustomerTier = "standard"
    affected_users_estimate: int = 1_000
    revenue_impact_usd_per_min: int = 50
    requires_mitigation: bool = True
    postmortem_required: bool = False


@dataclass
class Incident:
    """Runtime instance of an incident derived from a template.

    A runtime Incident captures the seeded, per-episode dynamic state that
    templates do not carry (such as which red herrings were rolled in, and the
    injected noise). The environment never mutates the template directly.
    """

    template: IncidentTemplate
    logs: Dict[str, str]
    metrics: Dict[str, str]
    kb: Dict[str, str]
    clue_keywords: Tuple[str, ...]
    accepted_fix_keywords: Tuple[Tuple[str, ...], ...]
    good_handoff: str
    postmortem_note_hint: Optional[str] = None

    @property
    def id(self) -> str:
        return self.template.id

    @property
    def title(self) -> str:
        return self.template.title

    @property
    def description(self) -> str:
        return self.template.description

    @property
    def root_cause(self) -> str:
        return self.template.root_cause

    @property
    def root_cause_synonyms(self) -> Tuple[str, ...]:
        return self.template.root_cause_synonyms

    @property
    def signals(self) -> Tuple[str, ...]:
        return self.template.signals

    @property
    def customer_tier(self) -> CustomerTier:
        return self.template.customer_tier

    @property
    def affected_users_estimate(self) -> int:
        return self.template.affected_users_estimate

    @property
    def revenue_impact_usd_per_min(self) -> int:
        return self.template.revenue_impact_usd_per_min

    @property
    def requires_mitigation(self) -> bool:
        return self.template.requires_mitigation

    @property
    def postmortem_required(self) -> bool:
        return self.template.postmortem_required

    @property
    def required_investigations(self) -> int:
        return self.template.required_investigations

    @property
    def playbook_hints(self) -> Tuple[str, ...]:
        return tuple(self.kb.keys())


class IncidentLibrary:
    """Collection of incident templates grouped by task name."""

    def __init__(self, templates_by_task: Mapping[str, List[IncidentTemplate]]):
        self._templates = {
            task: list(incidents) for task, incidents in templates_by_task.items()
        }

    def tasks(self) -> List[str]:
        return list(self._templates.keys())

    def templates_for(self, task_name: str) -> List[IncidentTemplate]:
        if task_name not in self._templates:
            task_name = next(iter(self._templates))
        return list(self._templates[task_name])

    def total_incidents(self) -> int:
        return sum(len(v) for v in self._templates.values())


def instantiate_incident(template: IncidentTemplate, rng: SeededRNG) -> Incident:
    """Build a runtime Incident by merging template data with seeded noise.

    Red herrings are always included deterministically so the agent cannot
    cheat by caching a "magic" investigation target; the order of extra
    targets is shuffled per episode to discourage positional memorization.
    """
    child = rng.child(template.id)

    combined_logs: Dict[str, str] = {**dict(template.logs), **dict(template.red_herring_logs)}
    combined_metrics: Dict[str, str] = {
        **dict(template.metrics),
        **dict(template.red_herring_metrics),
    }

    ordered_logs = dict(child.shuffled(combined_logs.items()))
    ordered_metrics = dict(child.shuffled(combined_metrics.items()))
    ordered_kb = dict(child.shuffled(template.kb.items()))

    return Incident(
        template=template,
        logs=ordered_logs,
        metrics=ordered_metrics,
        kb=ordered_kb,
        clue_keywords=template.clue_keywords,
        accepted_fix_keywords=template.accepted_fix_keywords,
        good_handoff=template.good_handoff,
    )


# ---------------------------------------------------------------------------
# Incident catalog
# ---------------------------------------------------------------------------


def _redis_pool() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E1",
        title="Checkout timeouts for premium users",
        description=(
            "Premium tier users are seeing intermittent checkout failures "
            "and elevated p99 latency on the payment path."
        ),
        category="payments",
        difficulty="easy",
        root_cause="redis_connection_pool_exhausted",
        root_cause_synonyms=(
            "redis connection pool exhausted",
            "redis pool saturated",
            "redis connection saturation",
        ),
        clue_keywords=("redis", "pool", "connection"),
        signals=(
            "Spike in checkout latency concentrated on premium cohort",
            "Error budget dropped from 99.9% to 99.2% in 15 minutes",
            "Payments sidecar reporting elevated retry counters",
        ),
        logs={
            "payments-api": "Timeout waiting for redis write lock (pool saturated)",
            "checkout-worker": "Queue delay exceeds 12s under load; retries amplifying",
            "redis-cluster": "Connection pool exhausted at 512/512, slow replies",
        },
        red_herring_logs={
            "cdn-edge": "cache HIT ratio normal, no edge anomalies",
            "email-service": "outbound smtp latency within baseline",
        },
        metrics={
            "dash-checkout": "p99 latency 4.1s (baseline 450ms), error-rate 6.2%",
            "dash-redis": "connections 512/512 (saturated), evictions low, cpu 74%",
            "dash-worker": "queue_depth 440, consumer_lag 380",
        },
        red_herring_metrics={
            "dash-cdn": "hit_ratio 97%, bandwidth steady",
        },
        kb={
            "kb-redis-pool": "Raise redis pool size and recycle stale handles on checkout-worker.",
            "kb-checkout-fallback": "Degrade recommendation calls when payment queue > 300.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("increase", "redis", "pool"),
            ("raise", "connection", "pool"),
            ("recycle", "stale", "connections"),
            ("enable", "checkout", "fallback"),
        ),
        required_investigations=2,
        customer_tier="premium",
        affected_users_estimate=42_000,
        revenue_impact_usd_per_min=480,
        requires_mitigation=True,
    )


def _jwt_clock_skew() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E2",
        title="Login failures right after auth deploy",
        description=(
            "Mobile users report intermittent login failures immediately "
            "after the latest auth service rollout."
        ),
        category="auth",
        difficulty="easy",
        root_cause="jwt_clock_skew_mismatch",
        root_cause_synonyms=(
            "jwt clock skew mismatch",
            "token clock skew",
            "issuer verifier clock mismatch",
        ),
        clue_keywords=("jwt", "clock", "skew", "token"),
        signals=(
            "401 error rate spikes exactly at deploy time",
            "Regional variance observed on mobile clients",
            "Some clients recover after app restart",
        ),
        logs={
            "auth-service": "Token issued-at in future; rejected by validator",
            "gateway": "401 bursts on auth-service route; upstream 2xx",
            "mobile-api": "Retrying auth flow due to invalid token state",
        },
        red_herring_logs={
            "payments-api": "steady 2xx, no anomalies",
        },
        metrics={
            "dash-auth": "401_rate 14%, token_validation_failures high",
            "dash-gateway": "auth_route_retries 3.2x baseline",
        },
        red_herring_metrics={
            "dash-cdn": "hit_ratio 96%",
        },
        kb={
            "kb-jwt-time": "Synchronize clock-skew tolerance between issuer and verifier.",
            "kb-mobile-auth": "Fallback to server timestamp for token freshness checks.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("increase", "jwt", "leeway"),
            ("sync", "clock", "tolerance"),
            ("roll", "back", "token"),
        ),
        required_investigations=2,
        customer_tier="standard",
        affected_users_estimate=15_500,
        revenue_impact_usd_per_min=120,
        requires_mitigation=True,
    )


def _email_spam_false_positive() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E3",
        title="Transactional emails marked as spam",
        description=(
            "A small but growing share of transactional receipts is being "
            "flagged as spam by downstream mailbox providers."
        ),
        category="notifications",
        difficulty="easy",
        root_cause="spf_record_misconfiguration",
        root_cause_synonyms=(
            "spf record misconfiguration",
            "spf misaligned",
            "dns spf mismatch",
        ),
        clue_keywords=("spf", "dns", "mailbox"),
        signals=(
            "Delivery success rate dropped from 99.2% to 93% in 24h",
            "Affected domains concentrate on a single provider family",
        ),
        logs={
            "email-service": "Remote MTA reports spf=softfail domain=receipts.example",
            "dns-resolver": "SPF record length 470 chars; exceeds soft limit",
        },
        red_herring_logs={
            "catalog-api": "HTTP 200 steady",
        },
        metrics={
            "dash-email": "delivery_success 93%, spam_flag_rate 4.8%",
            "dash-dns": "spf_lookup_count 12 per domain",
        },
        kb={
            "kb-spf": "Keep SPF record within 10 lookups and align domain sending IPs.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("fix", "spf", "record"),
            ("align", "sending", "domain"),
            ("shorten", "spf"),
        ),
        required_investigations=1,
        customer_tier="standard",
        affected_users_estimate=9_000,
        revenue_impact_usd_per_min=40,
        requires_mitigation=True,
    )


def _cache_invalidation_lag() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M1",
        title="Catalog stale prices during flash sale",
        description=(
            "During a scheduled flash sale, users keep seeing old prices "
            "on hot products while checkout shows the new price."
        ),
        category="catalog",
        difficulty="medium",
        root_cause="cache_invalidation_topic_lag",
        root_cause_synonyms=(
            "cache invalidation topic lag",
            "invalidation consumer lag",
            "kafka invalidation backlog",
        ),
        clue_keywords=("cache", "invalidation", "kafka", "consumer", "lag"),
        signals=(
            "Discrepancy between checkout price and catalog price",
            "Issue concentrated on top-selling SKUs and popular regions",
        ),
        logs={
            "catalog-api": "Read cache generation=188, expected=193",
            "kafka-consumer": "Lag increased on invalidation-topic partition 3",
            "pricing-service": "Published invalidation events at 2.1k/s",
        },
        red_herring_logs={
            "payments-api": "steady 2xx, no anomalies",
            "auth-service": "normal 2xx",
        },
        metrics={
            "dash-catalog": "cache_hit 98%, stale_reads elevated",
            "dash-kafka": "consumer_lag 5400 on partition 3",
        },
        red_herring_metrics={
            "dash-auth": "401_rate 0.6%",
        },
        kb={
            "kb-cache-invalidation": "Scale invalidation consumers and replay stalled partitions.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("scale", "invalidation", "consumer"),
            ("replay", "partition"),
            ("flush", "cache", "keys"),
        ),
        required_investigations=3,
        customer_tier="premium",
        affected_users_estimate=120_000,
        revenue_impact_usd_per_min=1_100,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _tz_normalization() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M2",
        title="Shipment ETA corruption in APAC",
        description=(
            "After deploying the route-planner update, shipment ETAs in APAC "
            "jump by +24h even though physical tracking is on time."
        ),
        category="logistics",
        difficulty="medium",
        root_cause="timezone_normalization_bug",
        root_cause_synonyms=(
            "timezone normalization bug",
            "locale timezone fallback",
            "iana offset mismatch",
        ),
        clue_keywords=("timezone", "locale", "iana", "offset"),
        signals=(
            "ETA anomaly concentrated in APAC region",
            "Warehouse scans are on time; only UI estimate is wrong",
        ),
        logs={
            "route-planner": "Parsed timezone fallback=UTC for locale en-IN",
            "eta-service": "Normalization mismatch for offset +05:30",
        },
        red_herring_logs={
            "auth-service": "normal 2xx",
        },
        metrics={
            "dash-eta": "eta_anomaly_rate 9.4%",
            "dash-route": "parser_warnings spike post deploy",
        },
        kb={
            "kb-timezone": "Use IANA timezone mapping and validate locale fallback path.",
        },
        good_handoff="triage_agent",
        accepted_fix_keywords=(
            ("patch", "timezone", "parser"),
            ("use", "iana", "timezone"),
            ("rollback", "route", "update"),
        ),
        required_investigations=2,
        customer_tier="standard",
        affected_users_estimate=22_000,
        revenue_impact_usd_per_min=180,
        requires_mitigation=True,
    )


def _invoice_idempotency() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M3",
        title="Duplicate invoices for merchants",
        description=(
            "A subset of merchants received duplicate invoices for the same "
            "order within the last billing cycle."
        ),
        category="billing",
        difficulty="medium",
        root_cause="idempotency_key_regression",
        root_cause_synonyms=(
            "idempotency key regression",
            "billing retry not idempotent",
            "duplicate invoice regression",
        ),
        clue_keywords=("idempotency", "retry", "dedupe", "invoice"),
        signals=(
            "Duplicate invoices share same order id",
            "Triggered after billing retry logic change",
        ),
        logs={
            "billing-worker": "Retry path ignored idempotency token for v2 flow",
            "billing-api": "POST /invoice executed twice for order O-92A",
        },
        red_herring_logs={
            "notification-gateway": "normal delivery",
        },
        metrics={
            "dash-billing": "duplicate_invoice_rate 3.7%",
            "dash-worker": "retry_attempts 2.4x baseline",
        },
        kb={
            "kb-idempotency": "Persist retry token before dispatch and enforce dedupe check.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("restore", "idempotency", "guard"),
            ("persist", "retry", "token"),
            ("dedupe", "invoice"),
        ),
        required_investigations=2,
        customer_tier="enterprise",
        affected_users_estimate=1_800,
        revenue_impact_usd_per_min=260,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _tls_expiry() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M4",
        title="Mutual TLS handshake failures",
        description=(
            "An internal service-to-service call is failing intermittently "
            "with TLS handshake errors after a certificate refresh."
        ),
        category="platform",
        difficulty="medium",
        root_cause="mtls_cert_chain_mismatch",
        root_cause_synonyms=(
            "mtls cert chain mismatch",
            "mutual tls chain mismatch",
            "intermediate certificate missing",
        ),
        clue_keywords=("tls", "certificate", "chain", "mtls"),
        signals=(
            "Handshake failures on newly issued certificates only",
            "Error rate climbs gradually as rolling restart progresses",
        ),
        logs={
            "service-mesh-proxy": "TLS handshake failure: unable to verify leaf certificate",
            "cert-manager": "Issued new certificate bundle without intermediate chain",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
        },
        metrics={
            "dash-mesh": "handshake_failure_rate 4.1%",
        },
        kb={
            "kb-mtls-chain": "Always include full intermediate chain on issued certificates.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("reissue", "certificate", "chain"),
            ("include", "intermediate", "certificate"),
            ("rollback", "cert", "refresh"),
        ),
        required_investigations=2,
        customer_tier="premium",
        affected_users_estimate=3_500,
        revenue_impact_usd_per_min=220,
        requires_mitigation=True,
    )


def _feature_flag_rollout() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M5",
        title="Search ranking broken for logged-in users",
        description=(
            "Search ranking quality collapsed for authenticated users only "
            "after a feature flag rollout to 50% of traffic."
        ),
        category="search",
        difficulty="medium",
        root_cause="feature_flag_scope_misconfigured",
        root_cause_synonyms=(
            "feature flag scope misconfigured",
            "flag targeting wrong segment",
            "experiment config wrong bucket",
        ),
        clue_keywords=("feature", "flag", "experiment", "targeting"),
        signals=(
            "Issue scoped to logged-in users only",
            "Click-through rate on top results dropped by 38%",
        ),
        logs={
            "search-api": "Feature flag 'ranking_v2_exp' reported enabled for tier=logged_in",
            "flag-service": "Rollout plan overrode segment targeting unexpectedly",
        },
        red_herring_logs={
            "payments-api": "steady 2xx",
        },
        metrics={
            "dash-search": "ctr_top3 -38%, dwell_time -21%",
            "dash-flags": "override_applied true for logged_in segment",
        },
        kb={
            "kb-feature-flag": "Use scoped rollout plans and verify segment before enabling.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("rollback", "feature", "flag"),
            ("restrict", "experiment", "segment"),
            ("disable", "ranking", "exp"),
        ),
        required_investigations=2,
        customer_tier="premium",
        affected_users_estimate=85_000,
        revenue_impact_usd_per_min=640,
        requires_mitigation=True,
    )


def _promo_rate_cascade() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H1",
        title="Cross-service saturation cascade during promo",
        description=(
            "A sudden promo launch triggers cascading failures across "
            "checkout, auth, and notifications."
        ),
        category="reliability",
        difficulty="hard",
        root_cause="rate_limit_misconfigured_for_promo_segment",
        root_cause_synonyms=(
            "rate limit misconfigured for promo segment",
            "segment rate limiter wrong",
            "promo segment overload",
        ),
        clue_keywords=("rate", "limit", "promo", "backoff"),
        signals=(
            "Failure spreads from notifications to checkout within minutes",
            "Customer segment 'promo_mega' has concentrated failures",
        ),
        logs={
            "notification-gateway": "429 flood for promo_mega segment",
            "checkout-api": "Retries amplified upstream failures from notification sidecar",
            "auth-service": "Session refresh queue saturated due to retry storm",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
            "dns-resolver": "no anomalies",
        },
        metrics={
            "dash-global": "error budget burn 3.7x",
            "dash-notify": "429_rate 38%",
            "dash-auth": "session_queue_depth 940",
        },
        kb={
            "kb-rate-limits": "Segment-specific limits must be applied with gradual rollout and backoff.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("hotfix", "promo", "rate"),
            ("enable", "exponential", "backoff"),
            ("throttle", "notification", "fanout"),
        ),
        required_investigations=3,
        customer_tier="premium",
        affected_users_estimate=410_000,
        revenue_impact_usd_per_min=2_400,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _schema_drift() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H2",
        title="Enterprise data export corruption",
        description=(
            "Enterprise customers report corrupted CSV exports from the "
            "analytics dashboard only for accounts migrated last week."
        ),
        category="analytics",
        difficulty="hard",
        root_cause="schema_version_drift",
        root_cause_synonyms=(
            "schema version drift",
            "exporter schema mismatch",
            "serializer version drift",
        ),
        clue_keywords=("schema", "version", "serializer", "drift"),
        signals=(
            "Corruption concentrated in accounts migrated last week",
            "Export job success is high but data quality is low",
        ),
        logs={
            "export-worker": "Schema mismatch: expected v11 got v10 on tenant shard",
            "analytics-api": "Fallback serializer dropped nullable columns",
        },
        red_herring_logs={
            "auth-service": "steady",
        },
        metrics={
            "dash-export": "job_success 97%, data_quality_score 61%",
            "dash-analytics": "schema_mismatch counter rising",
        },
        kb={
            "kb-schema-drift": "Force schema negotiation at read time and backfill migrated shards.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("enforce", "schema", "negotiation"),
            ("backfill", "migrated", "shards"),
            ("pin", "serializer"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=4_200,
        revenue_impact_usd_per_min=1_600,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _alert_storm() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H3",
        title="On-call alert storm masks outage",
        description=(
            "On-call rotations are overwhelmed by noisy duplicate alerts "
            "and miss the signal of a real outage forming underneath."
        ),
        category="observability",
        difficulty="hard",
        root_cause="dedupe_rule_disabled",
        root_cause_synonyms=(
            "dedupe rule disabled",
            "alert dedupe bypassed",
            "deduplication pipeline off",
        ),
        clue_keywords=("dedupe", "alert", "fingerprint"),
        signals=(
            "Alert volume 10x baseline with low incident diversity",
            "Primary outage not visible on first-page alerts",
        ),
        logs={
            "alert-router": "Deduplication pipeline bypassed after config reload",
            "pager-service": "Repeated notifications for identical fingerprint",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
        },
        metrics={
            "dash-alerts": "alerts_per_minute 1200",
            "dash-pager": "notification_duplicates 87%",
        },
        kb={
            "kb-alert-dedupe": "Restore dedupe stage and replay suppressed critical fingerprint set.",
        },
        good_handoff="triage_agent",
        accepted_fix_keywords=(
            ("restore", "dedupe", "rule"),
            ("replay", "critical", "fingerprints"),
            ("mute", "duplicate", "alert"),
        ),
        required_investigations=2,
        customer_tier="standard",
        affected_users_estimate=65_000,
        revenue_impact_usd_per_min=480,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _inventory_race() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H4",
        title="Inventory phantom stock oversells",
        description=(
            "Inventory service reports available stock that does not exist in "
            "the warehouse, causing real oversell incidents."
        ),
        category="inventory",
        difficulty="hard",
        root_cause="event_ordering_race_condition",
        root_cause_synonyms=(
            "event ordering race condition",
            "out of order reserve release",
            "event sequencing race",
        ),
        clue_keywords=("ordering", "race", "sequence", "reserve", "release"),
        signals=(
            "Negative physical stock but positive ledger entries",
            "Warehouse reconciliation jobs are delayed",
        ),
        logs={
            "inventory-ledger": "Out-of-order reserve/release events for same SKU",
            "warehouse-sync": "Late event merge exceeded ordering window",
        },
        red_herring_logs={
            "payments-api": "steady 2xx",
        },
        metrics={
            "dash-inventory": "oversell_incidents 4.2%",
            "dash-sync": "late_event_ratio 17%",
        },
        kb={
            "kb-event-ordering": "Use monotonic sequence guards and quarantine out-of-order events.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("enable", "sequence", "guards"),
            ("quarantine", "out-of-order", "events"),
            ("reconcile", "skus"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=2_500,
        revenue_impact_usd_per_min=1_250,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _deadlock_database() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H5",
        title="Recurring database deadlocks during reporting window",
        description=(
            "A heavy reporting workload is deadlocking with OLTP writes "
            "every hour causing brief customer-facing errors."
        ),
        category="data",
        difficulty="hard",
        root_cause="lock_escalation_on_reporting_view",
        root_cause_synonyms=(
            "lock escalation on reporting view",
            "reporting lock escalation",
            "database lock escalation",
        ),
        clue_keywords=("deadlock", "lock", "escalation", "reporting"),
        signals=(
            "Periodic spikes of 5xx errors exactly on the hour",
            "Reporting queries start at the same cadence",
        ),
        logs={
            "db-primary": "Deadlock detected between reporting-view-refresh and oltp-writer",
            "reporting-service": "Long-running view refresh initiated hourly",
        },
        red_herring_logs={
            "email-service": "no anomalies",
        },
        metrics={
            "dash-db": "deadlock_count 6 per hour",
            "dash-reports": "report_refresh_duration_s 52",
        },
        kb={
            "kb-lock-escalation": "Offload reporting to a read replica and lower isolation for view refresh.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("offload", "reporting", "replica"),
            ("reduce", "isolation", "view"),
            ("schedule", "reporting", "off-peak"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=12_000,
        revenue_impact_usd_per_min=980,
        requires_mitigation=True,
        postmortem_required=True,
    )


def build_incident_library() -> IncidentLibrary:
    """Return the built-in enterprise incident library."""
    return IncidentLibrary(
        templates_by_task={
            "easy": [_redis_pool(), _jwt_clock_skew(), _email_spam_false_positive()],
            "medium": [
                _cache_invalidation_lag(),
                _tz_normalization(),
                _invoice_idempotency(),
                _tls_expiry(),
                _feature_flag_rollout(),
            ],
            "hard": [
                _promo_rate_cascade(),
                _schema_drift(),
                _alert_storm(),
                _inventory_race(),
                _deadlock_database(),
            ],
        }
    )
