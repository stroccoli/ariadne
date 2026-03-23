# Structured Output

Structured output means the system returns information in a fixed shape instead of an unconstrained paragraph.

For AI Incident Analyzer, this matters because incident response depends on speed, consistency, and comparability. If every answer is written differently, the user has to interpret both the incident and the format at the same time. That increases cognitive load at exactly the wrong moment.

In this project, structured output creates a stable response contract with four key fields: incident type, likely root cause, confidence, and recommended actions. This is useful because each field answers a different triage question.

- `incident_type` gives a quick frame for what kind of failure is happening.
- `root_cause` states the most likely explanation.
- `confidence` signals how cautiously the result should be interpreted.
- `recommended_actions` turns analysis into next steps.

The deeper idea is that structure forces clarity. The system cannot hide behind vague language as easily. It must commit to a category, a hypothesis, a level of certainty, and concrete actions.

That is especially important in this project because the system is meant to support engineers during incidents, not impress them with fluent prose. Good output should be easy to scan, easy to compare, and easy to evaluate later.