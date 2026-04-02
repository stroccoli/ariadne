from __future__ import annotations

import random as _random
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Minimal data models (previously in benchmark_models / rubric_scoring)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ConceptCriterion:
    label: str
    keywords: tuple[str, ...]


def criterion(label: str, *keywords: str) -> ConceptCriterion:
    """Build a ConceptCriterion with normalised lowercase keywords."""
    return ConceptCriterion(label=label, keywords=tuple(k.lower() for k in keywords))


@dataclass(frozen=True)
class RootCauseRubric:
    required_concepts: tuple[ConceptCriterion, ...]
    forbidden_terms: tuple[str, ...] = ()
    require_uncertainty: bool = False


@dataclass(frozen=True)
class ActionRubric:
    required_concepts: tuple[ConceptCriterion, ...]
    minimum_actions: int = 1
    discouraged_phrases: tuple[str, ...] = ()


@dataclass(frozen=True)
class IncidentSample:
    sample_id: str
    description: str
    logs: str
    expected_incident_type: str
    expectation_note: str
    root_cause_rubric: RootCauseRubric
    action_rubric: ActionRubric


def get_sample_by_id(sample_id: str) -> IncidentSample:
    for sample in DEFAULT_AB_TEST_SAMPLES:
        if sample.sample_id == sample_id:
            return sample

    raise ValueError(f"Unknown sample id: {sample_id}")


def list_sample_ids() -> tuple[str, ...]:
    return tuple(sample.sample_id for sample in DEFAULT_AB_TEST_SAMPLES)


def _build_timeout_sample(
    sample_id: str,
    description: str,
    service: str,
    dependency: str,
    endpoint: str,
    deadline_ms: int,
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=(
            f"2026-03-19T09:14:21Z {service} WARN request to {dependency} exceeded {deadline_ms}ms deadline\n"
            f"2026-03-19T09:14:22Z {service} ERROR upstream call to {dependency} timed out after {deadline_ms + 1}ms\n"
            f"2026-03-19T09:14:23Z edge-proxy WARN returned HTTP 504 for {endpoint} after {deadline_ms / 1000:.1f}s\n"
            f"2026-03-19T09:14:24Z {service} INFO retry budget exhausted for {dependency}"
        ),
        expected_incident_type="timeout",
        expectation_note=(
            "This case should land on timeout because the strongest repeated evidence is deadline exhaustion, "
            "HTTP 504s, and retry exhaustion while calling a downstream dependency."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(
                criterion("dependency named", dependency),
                criterion("timeout evidence", "timed out", "deadline", "504", "exceeded"),
                criterion("retry or exhaustion signal", "retry", "exhausted", "latency"),
            )
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("inspect upstream latency", dependency, "latency", "upstream"),
                criterion("review timeout or retry policy", "timeout", "deadline", "retry", "backoff"),
            ),
            minimum_actions=2,
        ),
    )


def _build_dependency_sample(
    sample_id: str,
    description: str,
    service: str,
    dependency: str,
    endpoint: str,
    signal_a: str,
    signal_b: str,
    signal_c: str,
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=(
            f"2026-03-19T11:27:41Z {service} ERROR {dependency} {signal_a}\n"
            f"2026-03-19T11:27:42Z {service} WARN {signal_b}\n"
            f"2026-03-19T11:27:43Z {service} ERROR request for {endpoint} failed because {signal_c}\n"
            f"2026-03-19T11:27:44Z {service} INFO local process memory stable at 41 percent"
        ),
        expected_incident_type="dependency_failure",
        expectation_note=(
            "This case should land on dependency_failure because the failure is concentrated in a downstream service "
            "or external provider rather than the caller itself."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(
                criterion("dependency named", dependency),
                criterion(
                    "dependency outage signal",
                    "connection refused",
                    "503",
                    "unavailable",
                    "dns",
                    "reset",
                    "circuit breaker",
                    "broker unavailable",
                    "502",
                ),
                criterion("downstream framing", "upstream", "dependency", "downstream", "provider"),
            )
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("validate dependency health", dependency, "connectivity", "health", "availability", "dns"),
                criterion("mitigate dependency blast radius", "circuit breaker", "failover", "backoff", "degrade", "disable"),
            ),
            minimum_actions=2,
        ),
    )


def _build_database_sample(
    sample_id: str,
    description: str,
    service: str,
    database: str,
    signal_a: str,
    signal_b: str,
    signal_c: str,
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=(
            f"2026-03-19T13:40:01Z {service} ERROR {database} {signal_a}\n"
            f"2026-03-19T13:40:02Z {service} WARN transaction failed because {signal_b}\n"
            f"2026-03-19T13:40:03Z db-proxy ERROR {signal_c}\n"
            f"2026-03-19T13:40:04Z {service} INFO application workers healthy but writes blocked"
        ),
        expected_incident_type="database_issue",
        expectation_note=(
            "This case should land on database_issue because the logs explicitly name a database failure mode rather "
            "than a generic dependency problem."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(
                criterion("database named", database, "postgres", "mysql", "mongo", "database"),
                criterion(
                    "database failure signal",
                    "connection pool",
                    "deadlock",
                    "read only",
                    "statement timeout",
                    "authentication failed",
                    "server selection timeout",
                    "no space left on device",
                    "connection refused",
                    "missing column",
                ),
                criterion("database mechanism", "query", "transaction", "pool", "replica", "schema", "database"),
            )
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("inspect database health", database, "database", "query", "pool", "replica", "schema"),
                criterion("mitigate database load or failure", "reduce", "rollback", "restart", "pool", "lock", "failover", "index"),
            ),
            minimum_actions=2,
        ),
    )


def _build_memory_sample(
    sample_id: str,
    description: str,
    service: str,
    signal_a: str,
    signal_b: str,
    signal_c: str,
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=(
            f"2026-03-19T15:10:11Z {service} ERROR {signal_a}\n"
            f"2026-03-19T15:10:12Z kubelet WARN {signal_b}\n"
            f"2026-03-19T15:10:13Z {service} ERROR {signal_c}\n"
            f"2026-03-19T15:10:14Z deployment-controller INFO restarted {service} after memory failure"
        ),
        expected_incident_type="memory_issue",
        expectation_note=(
            "This case should land on memory_issue because the logs explicitly point to out-of-memory, heap, or "
            "allocation exhaustion signals."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(
                criterion("memory exhaustion signal", "out of memory", "oom", "heap", "allocation", "memory limit", "gc overhead"),
                criterion("service named", service),
                criterion("restart or pressure consequence", "restarted", "killed", "oomkill", "cgroup", "pressure"),
            )
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("inspect memory growth", "memory", "heap", "allocation", "rss", "profile"),
                criterion("mitigate memory pressure", "scale", "restart", "limit", "leak", "raise", "reduce"),
            ),
            minimum_actions=2,
        ),
    )


def _build_unknown_ambiguous_sample(
    sample_id: str,
    description: str,
    logs: str,
    observed_terms: tuple[str, ...],
    action_terms: tuple[str, ...],
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=logs,
        expected_incident_type="unknown",
        expectation_note=(
            "This case should stay unknown because the evidence supports multiple plausible explanations and the logs "
            "do not clearly separate them."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(criterion("mixed observed signals", *observed_terms),),
            require_uncertainty=True,
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("validate competing hypotheses", *action_terms),
                criterion("collect targeted evidence", "correlate", "trace", "confirm", "metrics", "telemetry"),
            ),
            minimum_actions=2,
        ),
    )


def _build_unknown_specific_sample(
    sample_id: str,
    description: str,
    logs: str,
    root_terms: tuple[str, ...],
    action_terms: tuple[str, ...],
) -> IncidentSample:
    return IncidentSample(
        sample_id=sample_id,
        description=description,
        logs=logs,
        expected_incident_type="unknown",
        expectation_note=(
            "This case remains unknown because the root cause is outside the current Day 1 taxonomy even though the "
            "logs still point to a concrete operational issue."
        ),
        root_cause_rubric=RootCauseRubric(
            required_concepts=(criterion("specific unsupported cause", *root_terms),)
        ),
        action_rubric=ActionRubric(
            required_concepts=(
                criterion("issue-specific mitigation", *action_terms),
                criterion("validation step", "validate", "confirm", "inspect", "verify"),
            ),
            minimum_actions=2,
        ),
    )


def _build_default_samples() -> tuple[IncidentSample, ...]:
    timeout_samples = (
        _build_timeout_sample("timeout_checkout_pricing", "Checkout waits on pricing dependency", "checkout-api", "pricing-service", "POST /checkout", 3000),
        _build_timeout_sample("timeout_search_catalog", "Search page stalls on catalog lookups", "search-api", "catalog-service", "GET /search?q=shoes", 2500),
        _build_timeout_sample("timeout_billing_invoice", "Billing worker times out while posting invoices", "billing-worker", "invoice-service", "POST /invoices", 5000),
        _build_timeout_sample("timeout_mobile_profile", "Mobile BFF hits profile deadline", "mobile-bff", "profile-service", "GET /me", 2000),
        _build_timeout_sample("timeout_orders_shipping", "Order orchestration blocks on shipping quote", "order-orchestrator", "shipping-quote-service", "POST /quotes", 3500),
        _build_timeout_sample("timeout_notifications_webhook", "Notification dispatcher exceeds webhook deadline", "notification-dispatcher", "webhook-delivery", "POST /deliveries", 4000),
        _build_timeout_sample("timeout_media_renderer", "Image rendering calls stall", "media-api", "image-renderer-service", "POST /transform", 4500),
        _build_timeout_sample("timeout_reporting_export", "Export creation exceeds worker deadline", "reporting-api", "export-worker", "POST /exports", 6000),
        _build_timeout_sample("timeout_recommendations_home", "Homepage recommendations expire under load", "homepage-gateway", "recommendations-service", "GET /home", 1800),
        _build_timeout_sample("timeout_audit_archive", "Audit archival requests hit the deadline", "audit-scheduler", "archive-service", "POST /archive", 7000),
    )

    dependency_samples = (
        _build_dependency_sample("dependency_auth_gateway", "Gateway cannot reach authentication service", "api-gateway", "auth-service", "POST /login", "connection refused at 10.2.4.18:8443", "circuit breaker opened for auth-service", "upstream auth-service returned HTTP 503"),
        _build_dependency_sample("dependency_tax_calculator", "Checkout depends on failing tax provider", "checkout-api", "tax-api", "POST /taxes", "returned HTTP 503 for request batch 991", "retry budget shrinking because tax-api is unavailable", "upstream tax-api returned HTTP 503"),
        _build_dependency_sample("dependency_email_relay", "Email sender loses SMTP dependency", "email-sender", "smtp-relay", "POST /send", "connection reset by peer during message delivery", "circuit breaker opened after smtp-relay failures", "smtp-relay returned temporary upstream failure"),
        _build_dependency_sample("dependency_flags_web", "Frontend cannot load feature flags", "web-frontend", "feature-flag-service", "GET /flags", "dns resolution failed for feature-flag-service", "fallback cache expired while flag dependency remained unavailable", "downstream feature-flag-service could not be resolved"),
        _build_dependency_sample("dependency_fraud_scoring", "Payment scoring service is down", "payments-api", "fraud-score-service", "POST /score", "returned HTTP 502 from upstream proxy", "circuit breaker opened for fraud-score-service", "downstream fraud-score-service is unavailable"),
        _build_dependency_sample("dependency_object_storage", "Media uploads fail on object storage outage", "media-uploader", "object-storage", "PUT /assets", "returned HTTP 503 slow down from provider", "upload workers degraded because object-storage remained unavailable", "provider object-storage returned HTTP 503"),
        _build_dependency_sample("dependency_secrets_sync", "Deployment controller loses secrets backend", "deployment-controller", "secrets-manager", "POST /refresh", "returned HTTP 503 during credential refresh", "retrying because secrets-manager is unavailable", "upstream secrets-manager returned HTTP 503"),
        _build_dependency_sample("dependency_cache_reader", "Session API cannot reach redis cache", "session-api", "redis-cache", "GET /sessions", "connection refused on 6379", "dependency reconnect loop is backing off", "downstream redis-cache remained unavailable"),
        _build_dependency_sample("dependency_broker_consumer", "Consumer group loses Kafka broker", "event-consumer", "kafka-broker", "READ payments", "broker unavailable during fetch request", "rebalancing because kafka-broker is unavailable", "dependency kafka-broker remained unavailable"),
        _build_dependency_sample("dependency_maps_eta", "Routing API loses maps provider", "routing-api", "maps-provider", "GET /eta", "returned HTTP 502 bad gateway", "fallback route estimator engaged after provider failures", "upstream maps-provider returned HTTP 502"),
    )

    database_samples = (
        _build_database_sample("database_pool_checkout", "Checkout exhausts the postgres pool", "checkout-api", "postgres-primary", "connection pool exhausted after 120 waiting clients", "too many clients already", "postgres-primary connection pool exhausted"),
        _build_database_sample("database_deadlock_inventory", "Inventory worker hits a MySQL deadlock", "inventory-worker", "mysql-orders", "deadlock found when trying to get lock", "transaction rolled back due to deadlock", "mysql-orders deadlock detected on stock_update"),
        _build_database_sample("database_missing_column_billing", "Billing deploy references a missing column", "billing-api", "postgres-billing", "column invoice_status does not exist", "statement failed during schema mismatch", "postgres-billing rejected query because column invoice_status does not exist"),
        _build_database_sample("database_read_only_orders", "Order writes routed to a read-only replica", "order-service", "postgres-replica", "cannot execute insert in a read-only transaction", "write attempted against read-only replica", "postgres-replica refused write transaction"),
        _build_database_sample("database_statement_timeout_reporting", "Reporting queries exceed statement timeout", "reporting-api", "analytics-db", "canceling statement due to statement timeout", "query exceeded statement timeout while aggregating reports", "analytics-db canceled slow query after statement timeout"),
        _build_database_sample("database_replica_lag_search", "Search sync falls behind due to replica lag", "search-sync", "postgres-search", "replica replay lag exceeded 45 seconds", "stale reads observed while replica lag persisted", "postgres-search replica lag remained above threshold"),
        _build_database_sample("database_auth_failure_audit", "Audit writer loses database credentials", "audit-api", "postgres-audit", "password authentication failed for user writer", "database login failed for write transaction", "postgres-audit rejected authentication for writer"),
        _build_database_sample("database_disk_full_events", "Event writer hits database disk exhaustion", "event-writer", "postgres-events", "could not extend file because no space left on device", "write failed after database volume reached capacity", "postgres-events could not allocate disk for relation file"),
        _build_database_sample("database_server_selection_catalog", "Catalog API loses Mongo primary", "catalog-api", "mongo-catalog", "server selection timeout after 30000 ms", "database client could not select writable server", "mongo-catalog server selection timeout persisted"),
        _build_database_sample("database_connection_refused_payments", "Payments API cannot open DB connections", "payments-api", "postgres-payments", "connection refused on 5432", "database client reconnects exhausted", "postgres-payments refused connection attempts"),
    )

    memory_samples = (
        _build_memory_sample("memory_oom_report_worker", "Report worker is OOM-killed", "report-worker", "process terminated: out of memory while rendering batch 44", "container report-worker hit memory limit 512Mi", "oomkill recorded after resident set size spiked"),
        _build_memory_sample("memory_java_heap_auth", "Auth service exhausts Java heap", "auth-service", "java.lang.outofmemoryerror: java heap space", "auth-service exceeded cgroup memory limit", "heap allocation failed during token cache refresh"),
        _build_memory_sample("memory_gc_overhead_indexer", "Indexer stalls in GC overhead", "search-indexer", "java.lang.outofmemoryerror: gc overhead limit exceeded", "search-indexer restarted after memory pressure", "heap exhausted during segment merge"),
        _build_memory_sample("memory_node_heap_web", "Web renderer exhausts Node heap", "web-renderer", "fatal error: reached heap limit allocation failed - javascript heap out of memory", "web-renderer container restarted after oomkill", "allocation failed while rendering homepage bundle"),
        _build_memory_sample("memory_cgroup_ingest", "Ingest service hits container memory ceiling", "ingest-service", "killed process 4121 because memory cgroup out of memory", "pod entered crashloopbackoff after memory limit exceeded", "out of memory detected while decoding batch payload"),
        _build_memory_sample("memory_allocator_recommendations", "ML recommendations service cannot allocate memory", "recommendation-ml", "std::bad_alloc raised during candidate ranking", "recommendation-ml restarted after memory pressure", "allocator failed because available memory was exhausted"),
        _build_memory_sample("memory_restart_gateway", "Gateway restarts after memory saturation", "api-gateway", "out of memory while buffering large request body", "api-gateway pod exceeded memory limit and restarted", "oomkill observed after request queue expanded"),
        _build_memory_sample("memory_leak_scheduler", "Scheduler leaks memory over time", "job-scheduler", "rss grew to 2.1gib before out of memory kill", "job-scheduler restarted because memory limit was exceeded", "memory leak suspected after steady heap growth"),
        _build_memory_sample("memory_pressure_notifications", "Notifications API enters heavy memory pressure", "notifications-api", "allocator failed under sustained memory pressure", "oom reaper killed notifications-api worker", "out of memory after queue fan-out increased"),
        _build_memory_sample("memory_fragmentation_image_processor", "Image processor fails on allocator exhaustion", "image-processor", "unable to allocate memory for resize buffer", "image-processor restarted after oomkill", "heap fragmentation left no contiguous memory for processing"),
    )

    unknown_samples = (
        _build_unknown_ambiguous_sample(
            "unknown_inventory_mixed",
            "Inventory worker shows mixed timeout and transient database signals",
            (
                "2026-03-19T10:03:10Z inventory-worker WARN sync job exceeded expected runtime by 18s\n"
                "2026-03-19T10:03:11Z inventory-worker ERROR partner feed returned HTTP 503 for batch 1882\n"
                "2026-03-19T10:03:12Z inventory-worker WARN retrying batch 1882 after read timeout from partner feed\n"
                "2026-03-19T10:03:13Z inventory-worker WARN postgres serialization failure while updating stock snapshot\n"
                "2026-03-19T10:03:14Z inventory-worker INFO retry succeeded for database write on attempt 2"
            ),
            observed_terms=("timeout", "503", "serialization", "partner feed", "postgres"),
            action_terms=("partner feed", "postgres", "separate", "isolate", "validate"),
        ),
        _build_unknown_ambiguous_sample(
            "unknown_checkout_latency_vs_cpu",
            "Checkout shows both deadline misses and CPU saturation hints",
            (
                "2026-03-19T12:11:00Z checkout-api WARN request latency reached 4.1s for POST /checkout\n"
                "2026-03-19T12:11:01Z checkout-api ERROR upstream call timed out after 4000ms\n"
                "2026-03-19T12:11:02Z checkout-api WARN worker cpu usage sustained at 98 percent\n"
                "2026-03-19T12:11:03Z checkout-api INFO no database or dependency error was emitted"
            ),
            observed_terms=("timeout", "cpu", "latency"),
            action_terms=("cpu", "timeout", "profile", "trace"),
        ),
        _build_unknown_ambiguous_sample(
            "unknown_gateway_tls_vs_auth",
            "Gateway failures mix TLS and auth denial signals",
            (
                "2026-03-19T12:21:00Z api-gateway ERROR tls handshake failed for auth edge\n"
                "2026-03-19T12:21:01Z api-gateway WARN login attempts returning HTTP 401\n"
                "2026-03-19T12:21:02Z api-gateway WARN edge clocks differ by 7 minutes across zones\n"
                "2026-03-19T12:21:03Z api-gateway INFO downstream service health check still passing"
            ),
            observed_terms=("tls", "401", "clock"),
            action_terms=("certificate", "clock", "auth", "verify"),
        ),
        _build_unknown_ambiguous_sample(
            "unknown_worker_dependency_vs_disk",
            "Background worker mixes partner errors with local disk pressure",
            (
                "2026-03-19T12:31:00Z sync-worker ERROR partner upload returned HTTP 503\n"
                "2026-03-19T12:31:01Z sync-worker WARN local spool directory at 98 percent capacity\n"
                "2026-03-19T12:31:02Z sync-worker ERROR write failed with no space left on device\n"
                "2026-03-19T12:31:03Z sync-worker INFO retry to partner still pending"
            ),
            observed_terms=("503", "no space left on device", "disk"),
            action_terms=("disk", "partner", "confirm", "isolate"),
        ),
        _build_unknown_ambiguous_sample(
            "unknown_session_cache_vs_network",
            "Session API sees both cache refusal and network packet loss hints",
            (
                "2026-03-19T12:41:00Z session-api ERROR redis-cache connection refused\n"
                "2026-03-19T12:41:01Z session-api WARN packet loss reached 18 percent between nodes\n"
                "2026-03-19T12:41:02Z session-api WARN reconnect succeeded once network jitter dropped\n"
                "2026-03-19T12:41:03Z session-api INFO cache health endpoint fluctuating"
            ),
            observed_terms=("connection refused", "packet loss", "jitter"),
            action_terms=("network", "cache", "confirm", "correlate"),
        ),
        _build_unknown_specific_sample(
            "unknown_disk_full_checkout",
            "Checkout host runs out of disk space",
            (
                "2026-03-19T14:00:00Z checkout-api ERROR failed to append order event: no space left on device\n"
                "2026-03-19T14:00:01Z checkout-api WARN local volume usage at 100 percent\n"
                "2026-03-19T14:00:02Z checkout-api INFO downstream dependencies healthy"
            ),
            root_terms=("disk", "no space left on device", "volume"),
            action_terms=("free disk", "volume", "cleanup", "expand"),
        ),
        _build_unknown_specific_sample(
            "unknown_cert_expiry_mobile",
            "Mobile gateway fails on certificate expiry",
            (
                "2026-03-19T14:10:00Z mobile-gateway ERROR tls certificate expired for edge listener\n"
                "2026-03-19T14:10:01Z mobile-gateway WARN handshake rejected before request routing\n"
                "2026-03-19T14:10:02Z mobile-gateway INFO upstream services healthy"
            ),
            root_terms=("certificate expired", "tls", "handshake"),
            action_terms=("renew certificate", "rotate", "listener", "validate"),
        ),
        _build_unknown_specific_sample(
            "unknown_config_regression_billing",
            "Billing release introduces bad configuration",
            (
                "2026-03-19T14:20:00Z billing-api ERROR invalid configuration: missing VAT_REGION_MAP\n"
                "2026-03-19T14:20:01Z billing-api WARN deployment 2026.03.19-7 rolled out 2 minutes ago\n"
                "2026-03-19T14:20:02Z billing-api INFO rollback not yet started"
            ),
            root_terms=("configuration", "missing", "deployment"),
            action_terms=("rollback", "config", "validate", "restore"),
        ),
        _build_unknown_specific_sample(
            "unknown_permission_media",
            "Media uploader fails on permission denial",
            (
                "2026-03-19T14:30:00Z media-uploader ERROR permission denied while writing to /mnt/assets\n"
                "2026-03-19T14:30:01Z media-uploader WARN service account lost write access after policy change\n"
                "2026-03-19T14:30:02Z media-uploader INFO object storage health checks passing"
            ),
            root_terms=("permission denied", "policy", "write access"),
            action_terms=("permissions", "policy", "restore", "verify"),
        ),
        _build_unknown_specific_sample(
            "unknown_clock_skew_tokens",
            "Token validation fails because nodes disagree on time",
            (
                "2026-03-19T14:40:00Z auth-service ERROR jwt token not yet valid by 420 seconds\n"
                "2026-03-19T14:40:01Z auth-service WARN ntp drift detected across worker nodes\n"
                "2026-03-19T14:40:02Z auth-service INFO database and downstream dependency health checks passing"
            ),
            root_terms=("clock skew", "ntp drift", "token not yet valid"),
            action_terms=("sync time", "ntp", "clock", "verify"),
        ),
    )

    samples = timeout_samples + dependency_samples + database_samples + memory_samples + unknown_samples
    if len(samples) != 50:
        raise ValueError(f"Expected 50 A/B test samples, found {len(samples)}")

    return samples


DEFAULT_AB_TEST_SAMPLES = _build_default_samples()


def get_random_samples(n: int = 10, seed: int | None = None) -> tuple[IncidentSample, ...]:
    """Return *n* samples drawn without replacement from the default library.

    When *seed* is provided the selection is reproducible.
    """
    total = len(DEFAULT_AB_TEST_SAMPLES)
    if n < 1 or n > total:
        raise ValueError(f"n must be between 1 and {total}, got {n}")

    rng = _random.Random(seed)
    selected = rng.sample(DEFAULT_AB_TEST_SAMPLES, n)
    return tuple(selected)