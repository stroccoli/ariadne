# Incident Classification

Incident classification means grouping observed failures into a small set of operationally meaningful categories.

For AI Incident Analyzer, classification is useful because raw logs are too detailed to reason about quickly under pressure. A label such as `timeout` or `database_issue` gives the responder an immediate mental frame. It does not solve the whole problem, but it narrows the search space.

This project starts with a deliberately small taxonomy:

- `timeout`
- `dependency_failure`
- `database_issue`
- `memory_issue`
- `unknown`

This is a good Day 1 choice because the categories are broad enough to cover many common incidents, but specific enough to guide action. It also includes `unknown`, which is important. A good incident analysis system must be able to admit uncertainty when evidence is weak or mixed.

The broader lesson is that classification is not just about labeling data. It is about creating a useful decision shortcut for humans. In this project, the label helps the user orient quickly, compare incidents more consistently, and choose more relevant investigation steps.

Later, the classification system can become richer. For now, keeping it small makes the problem easier to define, easier to test, and easier to explain.