# Day 1: Problem Definition

## 1. Problem Definition

AI Incident Analyzer is a system that reads operational evidence from an incident, interprets what kind of failure is most likely happening, proposes a likely cause, and suggests what responders should do next.

At a functional level, the system exists to help a human move from raw evidence to a usable incident assessment. The key point is not log storage, dashboarding, or automation. The key point is decision support during incident triage.

This problem is worth defining carefully because incident response usually fails first at the interpretation stage. Teams often have plenty of raw data, but not enough time or attention to turn that data into a clear hypothesis. A useful Day 1 system therefore should not try to solve everything. It should do one thing well: transform noisy incident evidence into a reviewable analysis.

The system does not need to prove the truth with certainty. It needs to provide a structured, inspectable starting point that reduces confusion and shortens the path to the next good engineering action.

### Why this scope is appropriate

This scope is intentionally narrow.

- It focuses on incident understanding, which is the first bottleneck during triage.
- It avoids implementation commitments too early, which keeps the design flexible.
- It preserves the role of the engineer as the decision-maker.
- It fits the project overview, which defines Ariadne as an analysis assistant rather than an observability platform or autonomous responder.

If the problem definition were broader, the project would become vague. If it were narrower, it would risk becoming a pattern-matching toy. This definition is a useful middle ground.

## 2. Input Specification

The input should represent a bounded incident investigation package. That means the system should receive not just raw log lines, but also enough surrounding context to interpret them responsibly.

At minimum, the input should contain four kinds of information:

1. Incident context

This identifies what service or workflow is under investigation, along with the time window and environment. Without this, the same error message may be interpreted incorrectly.

2. Log evidence

This is the primary signal. Logs may be plain text or structured records, but conceptually they are event observations tied to time, severity, component, and message content.

3. Metadata

Metadata gives the model important interpretation hints, such as service name, deployment version, region, request path, dependency name, or trace identifier.

4. Optional annotations

These are human-provided notes such as "incident started after deploy" or "database CPU is elevated." They should remain optional because Day 1 should not depend on perfect incident hygiene.

### Proposed input structure

- `incident_id`: unique identifier for the incident analysis request
- `service`: service or application under investigation
- `environment`: production, staging, or another deployment environment
- `time_window`: start and end of the relevant observation period
- `logs`: collection of observed log events
- `metadata`: contextual fields that help interpretation
- `annotations`: optional human notes or investigation hints

### Why this input design makes sense

- It keeps logs as the primary source of truth.
- It includes just enough context to avoid blind pattern matching.
- It allows both machine-generated and human-provided evidence.
- It stays general enough to support different logging styles later.

### JSON Example

```json
{
  "incident_id": "inc-2026-03-18-001",
  "service": "checkout-api",
  "environment": "production",
  "time_window": {
    "start": "2026-03-18T09:10:00Z",
    "end": "2026-03-18T09:20:00Z"
  },
  "logs": [
    {
      "timestamp": "2026-03-18T09:12:14Z",
      "level": "ERROR",
      "component": "payment-client",
      "message": "request to payment-service timed out after 3000ms",
      "request_id": "req-7812"
    },
    {
      "timestamp": "2026-03-18T09:12:15Z",
      "level": "WARN",
      "component": "checkout-handler",
      "message": "retrying payment authorization",
      "request_id": "req-7812"
    }
  ],
  "metadata": {
    "region": "eu-west-1",
    "version": "2026.03.18.4",
    "team": "commerce-platform"
  },
  "annotations": [
    "Spike in checkout failures started shortly after the latest deploy"
  ]
}
```

## 3. Output Specification

The output should be a structured incident assessment rather than a free-form paragraph.

The Day 1 output must include:

- `incident_type`: the best-fit category for the failure pattern
- `root_cause`: a concise explanation of the most likely cause
- `confidence`: a normalized indication of how strongly the evidence supports the conclusion
- `recommended_actions`: a list of next steps for the responder

### Proposed structured output

- `incident_type`: string from the defined incident taxonomy
- `root_cause`: short natural-language statement describing the likely cause
- `confidence`: numeric score from 0.0 to 1.0
- `recommended_actions`: ordered list of practical next actions

### Why this output stays minimal

This is the smallest output that still supports real operational use.

- It tells the responder what is probably happening.
- It states why the system thinks it is happening.
- It communicates uncertainty.
- It converts analysis into immediate action.

Anything less would be too vague to help. Anything much more, on Day 1, risks inventing precision before the product has earned it.

### JSON Example

```json
{
  "incident_type": "dependency_failure",
  "root_cause": "The checkout service is failing because calls to the downstream payment service are repeatedly timing out.",
  "confidence": 0.87,
  "recommended_actions": [
    "Check payment-service latency and availability in the affected region.",
    "Review whether the latest deploy changed timeout or retry behavior.",
    "Inspect error rates and saturation for the payment client connection pool."
  ]
}
```

## 4. Explanation of Output Design

### Why structured output?

Structured output is important because incident response is not only a reading task. It is also a coordination task.

In an incident, responders may need to:

- scan conclusions quickly
- compare one analysis with another
- feed results into downstream tools
- track patterns across incidents
- evaluate whether the system is improving over time

Free-form text is flexible, but difficult to compare and harder to validate consistently. A structured format forces the system to make explicit decisions. That is useful both for engineers and for future evaluation.

For this project specifically, structured output gives Ariadne a stable contract. Even if the internal analysis changes later, the system can still produce the same kind of operational artifact.

### Why these specific fields?

`incident_type` exists because responders need a fast mental frame. Classification is not the full answer, but it narrows the search space.

`root_cause` exists because classification alone is too abstract. Saying "database_issue" is useful, but saying "connection pool exhaustion is causing query failures" is much more actionable.

`recommended_actions` exists because diagnosis without next steps leaves the engineer with extra translation work. The system should reduce cognitive load, not shift it.

These three fields together create a useful progression:

1. What kind of problem is this?
2. What is the likely explanation?
3. What should I do next?

That sequence matches how real triage usually works.

### Why confidence matters

Confidence matters because incident evidence is often incomplete, conflicting, or misleading.

If the system presents every answer with the same tone, users will either over-trust it or ignore it. Neither outcome is acceptable. Confidence gives the responder a signal about how cautiously to interpret the conclusion.

For this project, confidence is useful in three ways:

- It reminds users that the system is probabilistic, not authoritative.
- It helps prioritize human review when evidence is weak.
- It creates a measurable output that can later be calibrated and improved.

Confidence does not mean the system knows the truth mathematically. It means the system is expressing how strongly the available evidence supports its current hypothesis. That distinction is important.

## 5. Incident Type Taxonomy

The Day 1 taxonomy should be small, comprehensible, and operationally useful. Too many categories would create ambiguity and unstable labels. Too few categories would hide meaningful differences.

### Initial taxonomy

1. `timeout`

Use when the dominant pattern is requests or operations exceeding expected time limits.

2. `dependency_failure`

Use when the service under investigation appears healthy enough to run, but a downstream dependency is unavailable, degraded, or returning failures.

3. `database_issue`

Use when the evidence points specifically to database-related failures such as connection problems, query errors, deadlocks, schema mismatches, or pool exhaustion.

4. `memory_issue`

Use when the dominant pattern involves out-of-memory conditions, heap pressure, memory leaks, repeated restarts due to memory exhaustion, or abnormal allocation behavior.

5. `unknown`

Use when the available evidence is insufficient, contradictory, or does not clearly fit the known categories.

### Why this classification is useful

This classification is useful because it provides a shared language for triage.

- It makes outputs easier to compare across incidents.
- It supports routing, dashboards, and future evaluation.
- It gives the user a fast orientation before reading the explanation.
- It prevents the system from pretending to know more than it does by preserving an explicit `unknown` category.

The taxonomy also balances generality and specificity. For example, `dependency_failure` and `database_issue` may overlap in practice, but separating them is valuable because database problems are common, operationally distinct, and often need different follow-up actions.

## 6. Example Cases

The examples below are intentionally realistic rather than perfectly clean. Real incidents rarely arrive as textbook patterns.

### Example 1: API request timeout

**Input logs**

- `2026-03-18T10:01:14Z ERROR api-gateway request to order-service exceeded timeout of 2000ms`
- `2026-03-18T10:01:14Z WARN api-gateway returning 504 for POST /orders`
- `2026-03-18T10:01:16Z ERROR api-gateway request to order-service exceeded timeout of 2000ms`

**Expected output**

```json
{
  "incident_type": "timeout",
  "root_cause": "Requests from the API gateway to the order service are exceeding the configured timeout threshold.",
  "confidence": 0.91,
  "recommended_actions": [
    "Check latency and saturation for the order service.",
    "Review recent changes affecting request duration.",
    "Inspect whether timeout thresholds are too aggressive for current load."
  ]
}
```

### Example 2: Downstream dependency failure

**Input logs**

- `2026-03-18T11:20:03Z ERROR checkout-api payment client received HTTP 503 from payment-service`
- `2026-03-18T11:20:03Z WARN checkout-api payment authorization failed, scheduling retry`
- `2026-03-18T11:20:05Z ERROR checkout-api payment client received HTTP 503 from payment-service`

**Expected output**

```json
{
  "incident_type": "dependency_failure",
  "root_cause": "Checkout failures are driven by repeated 503 responses from the downstream payment service.",
  "confidence": 0.89,
  "recommended_actions": [
    "Validate the health of the payment service.",
    "Check whether the failure is regional or global.",
    "Assess whether retries are increasing pressure on the dependency."
  ]
}
```

### Example 3: Database issue

**Input logs**

- `2026-03-18T12:42:18Z ERROR user-service database connection pool exhausted`
- `2026-03-18T12:42:18Z ERROR user-service failed to execute query: timeout acquiring connection`
- `2026-03-18T12:42:19Z WARN user-service request latency elevated for GET /profile`

**Expected output**

```json
{
  "incident_type": "database_issue",
  "root_cause": "The user service cannot obtain database connections, indicating connection pool exhaustion or database-side saturation.",
  "confidence": 0.94,
  "recommended_actions": [
    "Inspect database connection usage and pool limits.",
    "Check for slow queries or locking behavior.",
    "Review whether application traffic recently increased beyond expected capacity."
  ]
}
```

### Example 4: Memory issue

**Input logs**

- `2026-03-18T13:07:44Z ERROR recommendation-worker java.lang.OutOfMemoryError: Java heap space`
- `2026-03-18T13:07:45Z WARN recommendation-worker process exiting unexpectedly`
- `2026-03-18T13:08:02Z INFO recommendation-worker container restarted by orchestrator`

**Expected output**

```json
{
  "incident_type": "memory_issue",
  "root_cause": "The recommendation worker is crashing because it exhausted available heap memory.",
  "confidence": 0.97,
  "recommended_actions": [
    "Check heap utilization and recent changes in workload size.",
    "Review recent code or configuration changes affecting memory usage.",
    "Inspect restart frequency and container memory limits."
  ]
}
```

### Example 5: Unknown or mixed evidence

**Input logs**

- `2026-03-18T14:15:21Z ERROR notification-service request failed`
- `2026-03-18T14:15:23Z WARN notification-service retrying request`
- `2026-03-18T14:15:28Z ERROR notification-service unexpected exception in handler`

**Expected output**

```json
{
  "incident_type": "unknown",
  "root_cause": "The available logs show repeated failures, but the evidence is too generic to distinguish between an internal bug, dependency problem, or resource issue.",
  "confidence": 0.34,
  "recommended_actions": [
    "Collect more detailed logs or stack traces for the failing handler.",
    "Correlate the failures with recent deploys and dependency health.",
    "Check whether the issue is isolated to one code path or one environment."
  ]
}
```

## 7. Design Decisions and Tradeoffs

### Assumptions being made

The Day 1 design makes several assumptions:

- Relevant incident logs can be collected into a bounded analysis window.
- Logs contain enough signal to support at least coarse incident classification.
- Users prefer a concise structured assessment over a long narrative.
- Human responders will validate the output rather than execute it blindly.

These assumptions are practical, but not universally true. They are acceptable for an initial version because they keep the project focused on a tractable problem.

### What is simplified for now

Several things are intentionally simplified.

First, the taxonomy is small. Real incidents are messier than five labels, but a small taxonomy is easier to understand, test, and improve.

Second, the output schema is minimal. We are not yet including evidence citations, ranked alternatives, impact estimation, or remediation plans.

Third, the input centers on logs. In real operations, metrics, traces, deploy events, and infrastructure signals matter a lot. They are excluded at Day 1 to avoid premature system sprawl.

Fourth, `root_cause` is treated as a likely explanation, not a formally proven causal conclusion. That wording matters because true root cause analysis often requires broader evidence and more time.

### What should improve later

Later versions should improve in three broad directions.

1. Better evidence grounding

Outputs should eventually reference the specific evidence that supports each conclusion.

2. Richer taxonomy and multi-label reasoning

Some incidents should be allowed to express primary and secondary categories, such as a database issue that manifests first as timeouts.

3. Better uncertainty handling

Confidence should eventually be calibrated using evaluation data rather than treated only as a qualitative score.

### Why these tradeoffs are reasonable

The central tradeoff is between usefulness and over-design.

If we try to model the full complexity of production incidents on Day 1, the project becomes abstract and hard to build. If we oversimplify too much, the system becomes operationally unhelpful.

The current design chooses a disciplined middle path:

- small enough to implement and evaluate
- structured enough to be operationally useful
- explicit enough to teach good thinking habits
- limited enough to evolve cleanly later

That is the right shape for a Day 1 problem definition.