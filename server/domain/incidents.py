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


# ---------------------------------------------------------------------------
# Extended catalog (round-2 polish)
#
# 17 additional templates balance the tier mix (free / standard / premium /
# enterprise), add new service dimensions (DNS, CDN, ML inference, storage,
# message queue, config distribution) and new failure modes (GPU memory leaks,
# replication saturation, cache key collisions, firmware regressions, DST
# bugs). Each template follows the same pattern as INC-E1..H5 so the reward
# rubric, environment plumbing and training scripts require no changes.
# ---------------------------------------------------------------------------


def _dns_ttl_stale() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E4",
        title="Stale DNS routes free-tier API traffic to drained region",
        description=(
            "Free-tier API callers keep hitting a drained region even after "
            "a planned failover because DNS TTLs have not expired."
        ),
        category="networking",
        difficulty="easy",
        root_cause="dns_ttl_stale_after_failover",
        root_cause_synonyms=(
            "dns ttl stale after failover",
            "stale dns record",
            "long ttl blocking failover",
        ),
        clue_keywords=("dns", "ttl", "failover", "drain"),
        signals=(
            "Traffic ratio to drained region stays above 30% 30 minutes post-failover",
            "Only free-tier resolvers (no Anycast) are affected",
        ),
        logs={
            "dns-edge": "A record TTL=3600s still cached at regional resolvers",
            "traffic-router": "Residual traffic observed on drained region us-west-2b",
        },
        red_herring_logs={
            "payments-api": "steady 2xx",
        },
        metrics={
            "dash-dns": "ttl_expired_ratio 0.71 (expected >0.95)",
            "dash-router": "drained_region_share 34%",
        },
        red_herring_metrics={
            "dash-cdn": "hit_ratio 95%",
        },
        kb={
            "kb-dns-ttl": "Pre-lower TTL to 60s at least 2 TTLs before planned failovers.",
        },
        good_handoff="triage_agent",
        accepted_fix_keywords=(
            ("shorten", "dns", "ttl"),
            ("force", "resolver", "refresh"),
            ("rollback", "region", "drain"),
        ),
        required_investigations=1,
        customer_tier="free",
        affected_users_estimate=2_500,
        revenue_impact_usd_per_min=15,
        requires_mitigation=True,
    )


def _cdn_purge_scope() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E5",
        title="CDN purge missed a hot asset after release",
        description=(
            "A marketing banner refresh missed a subset of CDN edges, so a "
            "fraction of standard-tier users see the old creative."
        ),
        category="cdn",
        difficulty="easy",
        root_cause="cdn_purge_scope_mismatch",
        root_cause_synonyms=(
            "cdn purge scope mismatch",
            "edge purge partial",
            "shield purge missed",
        ),
        clue_keywords=("cdn", "purge", "edge", "shield"),
        signals=(
            "Small but persistent share of stale banner impressions",
            "Affected edges cluster on a single PoP provider",
        ),
        logs={
            "cdn-control-plane": "Purge job completed with 14 edges skipped (policy=legacy)",
            "edge-pop-bom-1": "Serving banner_v12 while origin is on banner_v13",
        },
        metrics={
            "dash-cdn": "stale_object_rate 1.4%, edge_sync_lag_s 312",
        },
        red_herring_metrics={
            "dash-auth": "401_rate 0.2%",
        },
        kb={
            "kb-cdn-purge": "Always use wildcard purge with full edge fanout for visual assets.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("reissue", "cdn", "purge"),
            ("fanout", "edge", "invalidation"),
            ("rotate", "asset", "hash"),
        ),
        required_investigations=1,
        customer_tier="standard",
        affected_users_estimate=11_000,
        revenue_impact_usd_per_min=60,
        requires_mitigation=True,
    )


def _autocomplete_stale() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E6",
        title="Search autocomplete missing this week's products",
        description=(
            "Free-tier shoppers see a stale autocomplete list that does not "
            "surface new SKUs released this Monday."
        ),
        category="search",
        difficulty="easy",
        root_cause="autocomplete_index_rebuild_skipped",
        root_cause_synonyms=(
            "autocomplete index rebuild skipped",
            "suggestion index stale",
            "nightly reindex missed",
        ),
        clue_keywords=("autocomplete", "index", "reindex", "suggestion"),
        signals=(
            "New SKUs launched Monday never appear in suggest responses",
            "Full text search returns them correctly",
        ),
        logs={
            "suggest-indexer": "Scheduled rebuild skipped (upstream lock held)",
            "suggest-api": "Serving snapshot v88 (expected v91)",
        },
        red_herring_logs={
            "payments-api": "steady 2xx",
        },
        metrics={
            "dash-suggest": "index_version 88, target_version 91",
            "dash-search": "full_text_recall 99%, autocomplete_recall 71%",
        },
        kb={
            "kb-autocomplete": "Reindex lock must release on job exit and alert on missed window.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("force", "index", "rebuild"),
            ("release", "reindex", "lock"),
            ("promote", "suggestion", "snapshot"),
        ),
        required_investigations=1,
        customer_tier="free",
        affected_users_estimate=18_000,
        revenue_impact_usd_per_min=30,
        requires_mitigation=True,
    )


def _webhook_retry_budget() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E7",
        title="Partner webhooks silently dropping",
        description=(
            "A handful of partner integrations stopped receiving webhook "
            "deliveries after a downstream 429 spike."
        ),
        category="integrations",
        difficulty="easy",
        root_cause="webhook_retry_budget_exhausted",
        root_cause_synonyms=(
            "webhook retry budget exhausted",
            "partner webhook giving up",
            "429 retry exhaustion",
        ),
        clue_keywords=("webhook", "retry", "429", "budget"),
        signals=(
            "Deliveries succeed for some partners and silently fail for others",
            "Affected partners all share a single rate-limit bucket",
        ),
        logs={
            "webhook-dispatcher": "Retry budget exhausted for partner_bucket=bucket-7",
            "partner-gateway": "HTTP 429 for 22 consecutive attempts on bucket-7",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
        },
        metrics={
            "dash-webhooks": "delivery_success_bucket7 34%, retry_budget_remaining 0",
        },
        kb={
            "kb-webhook-retry": "Split rate-limit buckets per partner and reset retry budgets on recovery.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("split", "retry", "bucket"),
            ("reset", "retry", "budget"),
            ("pause", "partner", "bucket"),
        ),
        required_investigations=2,
        customer_tier="standard",
        affected_users_estimate=1_400,
        revenue_impact_usd_per_min=80,
        requires_mitigation=True,
    )


def _thumbnail_worker_oom() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-E8",
        title="User profile thumbnails render blank on mobile",
        description=(
            "Free-tier mobile users see empty circles where their profile "
            "photo should appear, intermittently."
        ),
        category="media",
        difficulty="easy",
        root_cause="thumbnail_worker_oom_killed",
        root_cause_synonyms=(
            "thumbnail worker oom killed",
            "image worker out of memory",
            "thumbnailer oom loop",
        ),
        clue_keywords=("thumbnail", "oom", "memory", "worker"),
        signals=(
            "Missing thumbnails correlate with HEIC uploads from newer devices",
            "CPU is normal but worker restart count is spiking",
        ),
        logs={
            "thumbnail-worker": "SIGKILL received (oom_score_adj=500)",
            "image-pipeline": "HEIC decoder peak rss 1.9GB on large uploads",
        },
        metrics={
            "dash-thumbnails": "render_success 82%, worker_restarts 240/hr",
            "dash-k8s": "pod_oom_kill_count 42",
        },
        kb={
            "kb-thumbnail": "Cap HEIC decode memory or reject above 30MP at the edge.",
        },
        good_handoff="triage_agent",
        accepted_fix_keywords=(
            ("raise", "memory", "limit"),
            ("reject", "oversized", "heic"),
            ("downscale", "before", "decode"),
        ),
        required_investigations=2,
        customer_tier="free",
        affected_users_estimate=55_000,
        revenue_impact_usd_per_min=20,
        requires_mitigation=True,
    )


def _recommender_heap_leak() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M6",
        title="Recommender latency drifts up after model swap",
        description=(
            "Homepage recommendation latency is drifting up over six hours "
            "since this morning's model swap. p99 is now 2.1s."
        ),
        category="recommendations",
        difficulty="medium",
        root_cause="recommender_heap_leak_after_model_swap",
        root_cause_synonyms=(
            "recommender heap leak after model swap",
            "embedding cache not released",
            "old model tensors pinned",
        ),
        clue_keywords=("heap", "leak", "embedding", "model", "swap"),
        signals=(
            "Heap utilisation climbs 2% / hour since deploy",
            "Full GC frequency doubled but does not recover memory",
        ),
        logs={
            "recommender-service": "Loaded model v42; previous tensors not released",
            "jvm-gc": "Old gen occupancy 88% after full GC",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
        },
        metrics={
            "dash-recommender": "p99_latency_ms 2100, heap_used_pct 88",
            "dash-jvm": "full_gc_per_min 4, reclaimed_bytes_low",
        },
        red_herring_metrics={
            "dash-search": "ctr steady",
        },
        kb={
            "kb-model-swap": "Release previous model tensors explicitly before binding the new one.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("release", "previous", "model"),
            ("unload", "embedding", "cache"),
            ("rollback", "model", "swap"),
        ),
        required_investigations=2,
        customer_tier="premium",
        affected_users_estimate=95_000,
        revenue_impact_usd_per_min=410,
        requires_mitigation=True,
    )


def _consumer_group_rebalance() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M7",
        title="Order events stuck behind consumer rebalance storm",
        description=(
            "Order processing lag spiked after a rolling restart and has not "
            "recovered; fresh orders are 90s behind real time."
        ),
        category="messaging",
        difficulty="medium",
        root_cause="consumer_group_rebalance_storm",
        root_cause_synonyms=(
            "consumer group rebalance storm",
            "kafka consumer thrashing",
            "repeated partition reassignment",
        ),
        clue_keywords=("kafka", "consumer", "rebalance", "partition"),
        signals=(
            "Consumer group rebalanced 11 times in 5 minutes",
            "Lag stuck even though CPU is at 30%",
        ),
        logs={
            "order-consumer": "Rebalance triggered: member id rotated, session timeout=10s",
            "kafka-coordinator": "Generation 412 -> 423 in 5m, partitions churning",
        },
        red_herring_logs={
            "auth-service": "normal 2xx",
        },
        metrics={
            "dash-orders": "consumer_lag 90s, rebalance_count_5m 11",
            "dash-kafka": "generation_rotations 2.2/min",
        },
        kb={
            "kb-consumer-tuning": "Raise session.timeout.ms and heartbeat.interval.ms to avoid false expulsion.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("raise", "session", "timeout"),
            ("pin", "static", "membership"),
            ("stabilise", "consumer", "group"),
        ),
        required_investigations=2,
        customer_tier="premium",
        affected_users_estimate=48_000,
        revenue_impact_usd_per_min=520,
        requires_mitigation=True,
    )


def _config_push_skipped_canary() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M8",
        title="Enterprise tenants hit TLS verify failures after config push",
        description=(
            "A global config change flipped a TLS verification flag in "
            "production without going through canary."
        ),
        category="platform",
        difficulty="medium",
        root_cause="config_push_skipped_canary",
        root_cause_synonyms=(
            "config push skipped canary",
            "global config bypassed stage",
            "bulk config rollout regression",
        ),
        clue_keywords=("config", "canary", "push", "rollout"),
        signals=(
            "Enterprise tenants see TLS verify errors 3 minutes after deploy",
            "Canary stage shows zero traffic for this change",
        ),
        logs={
            "config-service": "Changeset CR-8812 applied globally (stages=[])",
            "api-gateway": "TLS verify flag=strict caused downstream handshake failures",
        },
        red_herring_logs={
            "email-service": "no anomalies",
        },
        metrics={
            "dash-config": "canary_coverage 0%, rollout_surface 100%",
            "dash-gateway": "tls_verify_failures 8.3%",
        },
        kb={
            "kb-config-rollout": "Require canary + 15 minutes bake before promoting config changes.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("rollback", "config", "change"),
            ("re-enable", "canary", "stage"),
            ("revert", "tls", "flag"),
        ),
        required_investigations=2,
        customer_tier="enterprise",
        affected_users_estimate=2_100,
        revenue_impact_usd_per_min=640,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _health_check_flapping() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M9",
        title="Autoscaler thrashing under brief latency blips",
        description=(
            "Autoscaler is adding and removing pods every 2 minutes in "
            "response to very short latency blips."
        ),
        category="platform",
        difficulty="medium",
        root_cause="health_check_timeout_too_aggressive",
        root_cause_synonyms=(
            "health check timeout too aggressive",
            "liveness probe too tight",
            "autoscaler oscillating",
        ),
        clue_keywords=("health", "check", "liveness", "autoscaler"),
        signals=(
            "Pod churn 6x baseline with no underlying load change",
            "Brief p99 blips align with scale events, not incidents",
        ),
        logs={
            "kubelet": "Liveness probe failed: HTTP 500 after 800ms",
            "autoscaler": "Scale up triggered; 3 pods added, 2 removed within 2m",
        },
        red_herring_logs={
            "payments-api": "steady 2xx",
        },
        metrics={
            "dash-k8s": "pod_churn_per_min 9, cpu_avg 42%",
            "dash-slo": "p99_latency_ms spikes tied to scale events",
        },
        kb={
            "kb-health-probe": "Raise liveness timeout and stagger readiness to avoid flap-driven scale events.",
        },
        good_handoff="triage_agent",
        accepted_fix_keywords=(
            ("raise", "probe", "timeout"),
            ("dampen", "autoscaler", "cooldown"),
            ("relax", "liveness", "threshold"),
        ),
        required_investigations=2,
        customer_tier="standard",
        affected_users_estimate=31_000,
        revenue_impact_usd_per_min=210,
        requires_mitigation=True,
    )


def _payment_webhook_dedupe() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M10",
        title="Payment confirmations delivered twice to enterprise partners",
        description=(
            "Two enterprise payment partners received the same confirmation "
            "webhook twice for a subset of transactions."
        ),
        category="payments",
        difficulty="medium",
        root_cause="webhook_dedupe_window_too_narrow",
        root_cause_synonyms=(
            "webhook dedupe window too narrow",
            "payment webhook duplicate delivery",
            "idempotency window clock drift",
        ),
        clue_keywords=("webhook", "dedupe", "idempotency", "window"),
        signals=(
            "Duplicates concentrated on retries across failover boundary",
            "Dedupe cache TTL is shorter than retry backoff",
        ),
        logs={
            "payments-webhook": "Duplicate delivery for txn T-332a after dedupe cache eviction",
            "scheduler": "Retry backoff 90s; dedupe ttl=60s",
        },
        red_herring_logs={
            "email-service": "steady",
        },
        metrics={
            "dash-payments": "duplicate_webhook_rate 0.9%, dedupe_hit_rate 88%",
        },
        kb={
            "kb-webhook-dedupe": "Dedupe TTL must exceed the maximum retry backoff window.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("extend", "dedupe", "ttl"),
            ("shrink", "retry", "backoff"),
            ("persist", "dedupe", "store"),
        ),
        required_investigations=2,
        customer_tier="enterprise",
        affected_users_estimate=620,
        revenue_impact_usd_per_min=480,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _origin_shield_bypass() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-M11",
        title="Origin overloaded after CDN policy change",
        description=(
            "Origin servers are seeing 5x normal traffic because a CDN "
            "policy change disabled origin shield for a large segment."
        ),
        category="cdn",
        difficulty="medium",
        root_cause="origin_shield_bypass_after_policy_change",
        root_cause_synonyms=(
            "origin shield bypass after policy change",
            "shield disabled for segment",
            "cache hierarchy collapsed",
        ),
        clue_keywords=("origin", "shield", "cdn", "policy"),
        signals=(
            "Origin 5xx rate climbs as CDN hit ratio collapses",
            "New CDN policy rolled out exactly at fault onset",
        ),
        logs={
            "cdn-policy": "Policy v5 removed shield targeting for premium segment",
            "origin-lb": "Connection queue depth spiking 5x baseline",
        },
        red_herring_logs={
            "dns-resolver": "no anomalies",
        },
        metrics={
            "dash-cdn": "hit_ratio 67% (baseline 94%)",
            "dash-origin": "rps 5.2x baseline, 5xx_rate 7.1%",
        },
        kb={
            "kb-origin-shield": "Changes to shield routing must go through shadow traffic before promotion.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("rollback", "cdn", "policy"),
            ("re-enable", "origin", "shield"),
            ("route", "through", "shield"),
        ),
        required_investigations=3,
        customer_tier="premium",
        affected_users_estimate=240_000,
        revenue_impact_usd_per_min=1_300,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _gpu_memory_fragmentation() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H6",
        title="LLM inference latency drifts up on production A100 pool",
        description=(
            "Enterprise API latency for the inference gateway has drifted "
            "from 420ms to 1.4s over 36 hours, with OOMs on larger prompts."
        ),
        category="ml_inference",
        difficulty="hard",
        root_cause="gpu_memory_fragmentation_after_prompt_schema_change",
        root_cause_synonyms=(
            "gpu memory fragmentation after prompt schema change",
            "kv cache fragmentation",
            "inference pool memory fragmentation",
        ),
        clue_keywords=("gpu", "memory", "fragmentation", "kv", "cache"),
        signals=(
            "Free VRAM fragmented into small blocks even though total free > 18GB",
            "OOM errors concentrate on prompts >2k tokens",
        ),
        logs={
            "inference-gateway": "CUDA OOM despite torch reports 18GB free; fragmentation detected",
            "model-runner": "Prompt schema v3 increased variable sequence lengths",
        },
        red_herring_logs={
            "auth-service": "steady",
        },
        metrics={
            "dash-inference": "p99_latency_ms 1400, oom_rate 3.2%",
            "dash-gpu": "vram_fragmentation_score 0.74",
        },
        kb={
            "kb-vram": "Recycle inference workers daily and pad sequences to bucketed lengths.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("recycle", "inference", "workers"),
            ("bucket", "prompt", "lengths"),
            ("rollback", "prompt", "schema"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=5_200,
        revenue_impact_usd_per_min=1_850,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _replication_saturation() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H7",
        title="Cross-region replication lag blocks disaster-recovery RPO",
        description=(
            "Replication lag from the primary region to DR has exceeded "
            "five minutes for the last hour, violating RPO=60s."
        ),
        category="data",
        difficulty="hard",
        root_cause="replication_saturation_during_backup_window",
        root_cause_synonyms=(
            "replication saturation during backup window",
            "wal shipping backpressure",
            "replica network saturation",
        ),
        clue_keywords=("replication", "lag", "wal", "rpo", "backup"),
        signals=(
            "Lag correlates exactly with nightly backup window",
            "Network egress saturated on primary -> DR link",
        ),
        logs={
            "db-primary": "WAL shipping backpressure; replica slot lagging 6.2m",
            "backup-job": "Base backup in progress; 4.1 GB/s read rate",
        },
        red_herring_logs={
            "notification-gateway": "steady delivery",
        },
        metrics={
            "dash-replication": "lag_seconds 372 (rpo=60)",
            "dash-network": "egress_primary_to_dr 9.8 Gbps (cap=10)",
        },
        kb={
            "kb-replication-backup": "Throttle backup or move it off hours of peak replication traffic.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("throttle", "backup", "rate"),
            ("shift", "backup", "window"),
            ("raise", "replication", "bandwidth"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=8_900,
        revenue_impact_usd_per_min=1_400,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _cache_key_collision() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H8",
        title="Cross-tenant data bleed from cache key collision",
        description=(
            "A rare cache key collision is briefly returning one enterprise "
            "tenant's data to another. This is a data-isolation incident."
        ),
        category="security",
        difficulty="hard",
        root_cause="cache_key_collision_across_tenants",
        root_cause_synonyms=(
            "cache key collision across tenants",
            "shared cache tenant bleed",
            "tenant id missing from cache key",
        ),
        clue_keywords=("cache", "key", "collision", "tenant"),
        signals=(
            "Two enterprise tenants report seeing each other's dashboard metadata",
            "Cache key construction omits tenant-id under a specific code path",
        ),
        logs={
            "api-gateway": "Cache HIT for key=/v2/workspace/42 served to tenant=91",
            "cache-layer": "Collision detected between tenants 42 and 91 on key prefix /v2/workspace",
        },
        red_herring_logs={
            "email-service": "steady",
        },
        metrics={
            "dash-cache": "collision_count 14 in last 2h",
            "dash-security": "isolation_violations 2",
        },
        kb={
            "kb-cache-tenant": "Prefix every cache key with tenant_id and enforce via lint check.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("prefix", "tenant", "cache"),
            ("invalidate", "shared", "cache"),
            ("quarantine", "cache", "segment"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=320,
        revenue_impact_usd_per_min=2_100,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _cron_dst_double_trigger() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H9",
        title="Scheduled jobs fire twice at DST rollover",
        description=(
            "Key premium billing jobs executed twice at the daylight-saving "
            "transition, causing premium charge duplicates."
        ),
        category="scheduling",
        difficulty="hard",
        root_cause="cron_dst_transition_double_trigger",
        root_cause_synonyms=(
            "cron dst transition double trigger",
            "scheduler timezone ambiguity",
            "dst fallback replay",
        ),
        clue_keywords=("cron", "dst", "timezone", "scheduler"),
        signals=(
            "Job history shows two runs at 01:00 and 01:00 local time",
            "Billing duplicates concentrate on a single geographic region",
        ),
        logs={
            "scheduler": "Fired job billing.nightly at 2026-03-29 01:00 (GMT+1 and GMT+0)",
            "billing-worker": "Second invocation completed 12 minutes after first",
        },
        red_herring_logs={
            "catalog-api": "steady 2xx",
        },
        metrics={
            "dash-scheduler": "double_fire_count 3 (expected 0)",
            "dash-billing": "duplicate_charge_rate 2.1%",
        },
        kb={
            "kb-dst-schedule": "Anchor scheduled jobs on UTC and convert to local time at display only.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("anchor", "schedule", "utc"),
            ("deduplicate", "scheduled", "runs"),
            ("reconcile", "duplicate", "charges"),
        ),
        required_investigations=3,
        customer_tier="premium",
        affected_users_estimate=6_400,
        revenue_impact_usd_per_min=1_100,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _partial_publish_feed() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H10",
        title="Real-time feed gaps during partial publish",
        description=(
            "Premium trading-floor customers see gaps in the realtime price "
            "feed after a publisher restart; some updates never arrived."
        ),
        category="realtime",
        difficulty="hard",
        root_cause="partial_publish_without_transaction_boundary",
        root_cause_synonyms=(
            "partial publish without transaction boundary",
            "publisher crash mid batch",
            "realtime feed gap",
        ),
        clue_keywords=("publish", "transaction", "feed", "partial"),
        signals=(
            "Sequence numbers skip in a bounded window around the publisher restart",
            "Replay API can fill the gap but live subscribers missed it",
        ),
        logs={
            "price-publisher": "Process restarted mid-batch, seq=88230 not flushed",
            "realtime-bus": "Detected sequence gap 88230-88236 on channel=prices.us",
        },
        red_herring_logs={
            "auth-service": "steady",
        },
        metrics={
            "dash-realtime": "gap_count 6 in 30s, subscriber_reconcile_lag_s 48",
        },
        kb={
            "kb-publish-txn": "Wrap each batch in a transactional publish so crashes never leave gaps.",
        },
        good_handoff="investigator_agent",
        accepted_fix_keywords=(
            ("enable", "transactional", "publish"),
            ("replay", "sequence", "gap"),
            ("force", "subscriber", "reconcile"),
        ),
        required_investigations=3,
        customer_tier="premium",
        affected_users_estimate=3_900,
        revenue_impact_usd_per_min=1_750,
        requires_mitigation=True,
        postmortem_required=True,
    )


def _ssd_firmware_regression() -> IncidentTemplate:
    return IncidentTemplate(
        id="INC-H11",
        title="Storage checksum failures on upgraded SSD fleet",
        description=(
            "Enterprise object storage is returning checksum-mismatch errors "
            "on a subset of volumes after a firmware roll-forward."
        ),
        category="storage",
        difficulty="hard",
        root_cause="ssd_firmware_checksum_regression",
        root_cause_synonyms=(
            "ssd firmware checksum regression",
            "storage firmware corruption",
            "nvme firmware crc bug",
        ),
        clue_keywords=("firmware", "ssd", "checksum", "storage"),
        signals=(
            "Checksum failures concentrate on volumes upgraded in the last 72 hours",
            "Vendor advisory mentions similar symptoms after firmware F2.14",
        ),
        logs={
            "storage-agent": "CRC mismatch on volume vol-221 firmware=F2.14",
            "fleet-manager": "Upgrade batch included F2.14 for 18 volumes",
        },
        red_herring_logs={
            "email-service": "steady",
        },
        metrics={
            "dash-storage": "checksum_error_rate 0.8%",
            "dash-fleet": "volumes_on_F2.14 18, volumes_healthy 402",
        },
        kb={
            "kb-ssd-firmware": "Quarantine affected firmware and roll back to the last known-good version.",
        },
        good_handoff="ops_manager_agent",
        accepted_fix_keywords=(
            ("rollback", "ssd", "firmware"),
            ("quarantine", "affected", "volumes"),
            ("reseed", "checksum", "index"),
        ),
        required_investigations=3,
        customer_tier="enterprise",
        affected_users_estimate=1_800,
        revenue_impact_usd_per_min=1_950,
        requires_mitigation=True,
        postmortem_required=True,
    )


def build_incident_library() -> IncidentLibrary:
    """Return the built-in enterprise incident library (30 templates)."""
    return IncidentLibrary(
        templates_by_task={
            "easy": [
                _redis_pool(),
                _jwt_clock_skew(),
                _email_spam_false_positive(),
                _dns_ttl_stale(),
                _cdn_purge_scope(),
                _autocomplete_stale(),
                _webhook_retry_budget(),
                _thumbnail_worker_oom(),
            ],
            "medium": [
                _cache_invalidation_lag(),
                _tz_normalization(),
                _invoice_idempotency(),
                _tls_expiry(),
                _feature_flag_rollout(),
                _recommender_heap_leak(),
                _consumer_group_rebalance(),
                _config_push_skipped_canary(),
                _health_check_flapping(),
                _payment_webhook_dedupe(),
                _origin_shield_bypass(),
            ],
            "hard": [
                _promo_rate_cascade(),
                _schema_drift(),
                _alert_storm(),
                _inventory_race(),
                _deadlock_database(),
                _gpu_memory_fragmentation(),
                _replication_saturation(),
                _cache_key_collision(),
                _cron_dst_double_trigger(),
                _partial_publish_feed(),
                _ssd_firmware_regression(),
            ],
        }
    )
