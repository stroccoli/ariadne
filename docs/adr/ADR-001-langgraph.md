# ADR-001: LangGraph over simple LangChain chains

## Status: Accepted

## Date: 2026-03-24

---

## Context

The incident analysis pipeline started on Day 1 as a single prompt call and grew, through Day 4, into a multi-agent architecture with a manually written `Orchestrator` class. That class managed:

- Sequencing three agents: `ClassifierAgent` → `RAGAgent` → `AnalyzerAgent`
- A shared `IncidentState` object that all agents read and mutated
- Conditional retry logic: if `analysis.confidence < 0.7` and `retrieval_attempts < MAX_RETRIES`, re-run retrieval with an enriched query and re-run the analyzer
- Token usage and timing bookkeeping accumulated across agents
- LangSmith tracing wired manually

The Day 4 custom orchestrator worked, but encoding the control flow inside a hand-rolled class made the graph implicit. The retry condition (`confidence < 0.7`, max one retry) was a nested `if/else` block buried inside `Orchestrator.analyze_incident()`. Adding a second conditional branch would have required modifying orchestration logic that mixed sequencing, state mutation, and business rules.

The decision was evaluated at Day 5 refactor. The pipeline has four nodes, one conditional edge, and shared typed state. These are exactly the primitives that graph-based orchestrators are designed for.

---

## Decision

Replace the custom `Orchestrator` class with a **LangGraph `StateGraph`** defined in `ariadne/core/graph.py`.

Each pipeline stage is a pure function (`classify_node`, `retrieve_node`, `analyze_node`, `build_output_node`) that receives the current `IncidentState` and returns a dict of fields to update. LangGraph handles state merging, edge traversal, and conditional routing. The retry condition is:

```python
def should_retry(state: IncidentState) -> str:
    confidence = state.analysis.confidence if state.analysis else 0.0
    if confidence < CONFIDENCE_THRESHOLD and state.retrieval_attempts < MAX_RETRIEVAL_ATTEMPTS:
        return "retry"
    return "done"

graph.add_conditional_edges("analyze", should_retry, {"retry": "retrieve", "done": "build_output"})
```

`CONFIDENCE_THRESHOLD = 0.7`, `MAX_RETRIEVAL_ATTEMPTS = 2`.

LangSmith tracing is native: passing `run_name` and `tags` in the `config` argument to `graph.invoke()` is sufficient — no manual span management needed.

Agent logic moves into module-level functions (`run_classifier`, `run_retrieval`, `run_analyzer`) that are easier to test in isolation. The `BaseAgent` ABC and all agent class wrappers are removed.

---

## Alternatives considered

### LangChain Expression Language (LCEL) with `.pipe()`

LCEL chains compose well for linear pipelines, but the conditional retry loop is not a linear pipeline. Conditional branching requires `RunnableBranch` or lambda routers, which are less readable than a named edge function on a graph. LCEL also does not natively visualise graph topology.

**Rejected:** conditional branching is first-class in LangGraph; workarounds in LCEL add complexity without benefit.

### Simple LangChain chains (SequentialChain)

`SequentialChain` has no concept of conditional edges or shared typed state. Adding retry logic would require wrapping the entire chain in a `while` loop with manual state threading — exactly the Day 4 custom orchestrator problem.

**Rejected:** does not support the retry pattern.

### Custom Orchestrator (Day 4 implementation — actually shipped)

The Day 4 `Orchestrator` class was written and worked. It proved the design before the LangGraph migration. The real trade-off was that the conditional logic was encoded in imperative Python rather than a declarative graph edge.

Adding a second conditional — e.g., switching to `compact` mode on retry, or escalating to a stronger model — would have required modifying the orchestrator body, touching state mutation and routing in the same function.

**Superseded by LangGraph:** the graph externalises routing rules as named edges, making them independently testable and visible in the Mermaid topology diagram.

### Completely custom implementation

A `while` loop driving direct LLM calls with manual state passing is the simplest possible implementation. For a three-node pipeline this is reasonable. The cost is losing native LangSmith tracing and the declarative topology.

**Rejected:** LangSmith integration was important for prompt iteration during development. The graph's `get_graph().draw_mermaid()` output was used directly for documentation.

---

## Consequences

**Positive:**
- Conditional retry routing is a single named function and a single `add_conditional_edges` call — readable, testable, isolated from agent logic.
- LangSmith tracing is native: every production run produces a full trace with per-node latency and token counts at zero additional code cost.
- `IncidentState` is a Pydantic model; LangGraph validates state transitions at each edge.
- The graph topology can be rendered as a Mermaid diagram programmatically via `get_graph_diagram()` — used in the README and architecture docs.
- Agent functions (`run_classifier`, etc.) are plain Python functions, straightforward to unit-test without graph machinery.

**Negative:**
- Adds `langgraph` as a dependency (~10 MB). For a 3–4 node linear pipeline this is arguably over-engineered.
- LangGraph's API evolves quickly; the 0.2.x → 1.x migration introduced breaking changes in the conditional edge API.
- Debugging a failing graph node requires understanding LangGraph's state reducer model, which is non-obvious to contributors unfamiliar with the library.

---

## When would we revisit this?

- If the pipeline needs **parallel execution** (e.g., running classification and an initial retrieval pass simultaneously) — LangGraph supports this via the `Send` API, but adds conceptual complexity. At that point re-evaluating vs. a purely async Python implementation would be worthwhile.
- If **LangGraph's API stability** proves problematic across minor versions during active development.
- If the pipeline is reduced to a single-pass linear chain with no branching — the graph overhead stops earning its keep.
