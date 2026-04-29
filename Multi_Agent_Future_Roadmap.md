# BaseAgent: Multi-Agent Feature Roadmap

## Goal

This roadmap describes the features needed to complete BaseAgent's multi-agent orchestration capabilities. It focuses on agentic AI infrastructure, not on domain-specific logic (ontology parsing, database connectors, etc.).

---

## Success Criteria

**Prototype is complete when:**

1. An `AgentTeam` with a supervisor delegates to specialist agents, demonstrating iterative inter-agent feedback (e.g., mapping agent requests parser re-extraction).
2. At least one agent connects to a remote MCP server (e.g., OLS4, biocontext.ai) and retrieves structured biomedical data.
3. Agents operate with isolated REPL namespaces -- concurrent execution does not corrupt shared state.
4. Each agent has a distinct identity (`AgentSpec`: name, role, tool subset, skill subset, optional model override).
5. The system emits multi-agent `AgentEvent` types (`AGENT_START`, `AGENT_COMPLETE`, `SUPERVISOR_DECISION`) that a frontend can render.
6. Context window management prevents overflow during multi-step workflows.
7. Structured errors propagate from sub-agents to supervisors with retry/reroute semantics.
8. `run()` is async-first; `run_sync()` preserves backward compatibility for scripts and notebooks.
9. The existing single-agent API (`BaseAgent(llm=...).run(task)`) continues to work unchanged.

---

## Completed Features

Features 1-8 and 10 are implemented and tested. See `.claude/baseagent_modules.md` for current API details.

| Feature | Summary | Tests |
|---------|---------|-------|
| **Feature 1: MCP Overhaul** | Remote Streamable HTTP transport, auth headers with `${ENV_VAR}` interpolation, async/sync bridge fix | 13 unit tests |
| **Feature 2: AgentSpec** | `AgentSpec` dataclass for agent identity; `{role_description}` parameterization in system prompt | 22 unit tests |
| **Feature 3: REPL Namespace Isolation** | Per-instance `_repl_namespace` and `PlotCapture`; `namespace` param on `run_python_repl` and `inject_custom_functions_to_repl` | unit tests |
| **Feature 4: Extract Subgraph** | `get_subgraph()` returns uncompiled `StateGraph` for LangGraph composition; `configure()` calls it then compiles | 18 unit tests |
| **Feature 5: Context Window Management** | Sliding window truncation in `generate` and `execute_self_critic` nodes; `max_context_messages` config field; `BASE_AGENT_MAX_CONTEXT_MESSAGES` env var | 21 unit tests |
| **Feature 6: Error Handling + Termination** | Structured error hierarchy (`errors.py`); `max_iterations`, `max_cost`, `max_consecutive_errors` config fields; `LLMError` wrapping; per-run cost budget via `_run_usage_start` index | 50 unit tests |
| **Feature 7: Async-First API** | `arun()`, `aresume()`, `areject()` async counterparts alongside unchanged sync API; `_setup_run()` and `_post_stream_result()` helpers eliminate 6× duplication | 16 unit tests |
| **Feature 8: Multi-Agent Orchestration** | `AgentTeam` supervisor orchestrator; `MultiAgentState`; `MaxRoundsExceededError`; 3 stub `EventType` values (`AGENT_START`, `AGENT_COMPLETE`, `SUPERVISOR_DECISION`); `extract_agent_result` strips `<think>/<solution>` tags from results passed to supervisor; `SupervisorDecision` fields carry `Field(description=...)` for schema-level LLM guidance | 13 unit tests |
| **Feature 10: Skills System Overhaul** | Spec-driven targeted loading, progressive disclosure (catalog mode), bundled resources (`read_skill_resource`), functional `tools` field | 69 unit tests |

---

## Prototype Feature Specifications

---

### Feature 7: Async-First API ✅

**Implemented.** `arun()`, `aresume()`, `areject()` added alongside the unchanged sync API. See `examples/12_async_api.py`.

---

### Feature 8: Multi-Agent Orchestration ✅

**Implemented.** `AgentTeam` supervisor orchestrator added. See `examples/13_multi_agent.py`.


## Post-Prototype Feature Specifications

Deferred until the prototype is validated. Listed here to inform design decisions -- do not build hooks or abstractions for these unless they are zero-cost.

### P1: Subgraph Embedding (Option B Migration)

Migrate from Option A (multiple instances) to embedded LangGraph subgraphs. Each `BaseAgent`'s `get_subgraph()` (Feature 4) returns an uncompiled `StateGraph` that a parent graph embeds via `workflow.add_subgraph()`. Enables unified checkpointing, streaming, and time-travel across the full multi-agent system. Prerequisite: validated multi-agent interaction patterns from the prototype.

**Files:** `BaseAgent/base_agent.py`, `BaseAgent/multi_agent/orchestrator.py`

### P2: Semantic Memory / RAG

`MemoryStore` with SQLite FTS5 backend (no vector DB dependency). Agent-accessible tools: `write_memory(text, metadata)`, `search_memory(query, k)`. System prompt injection of top-k relevant entries. Useful for persisting and retrieving disease-specific findings across workflow steps.

**Files:** new `BaseAgent/memory.py`, `BaseAgent/base_agent.py`

### P3: Cross-Agent Shared Memory via LangGraph Store

Replace per-agent `MemoryStore` instances with LangGraph's built-in `Store` (`InMemoryStore` or `SqliteStore`) for persistent cross-agent memory. Agents write findings to `store.put(namespace=("findings", agent_name), ...)` and read across agents via `store.search(namespace=("findings",))`.

### P4: Observability & Cost Tracking

Surface existing `UsageMetrics` to frontend via `AgentEvent`. Add `usage_summary()` to `BaseAgent`. Aggregate costs across agents in the orchestrator (`orchestrator.total_usage`, `orchestrator.usage_by_agent`). Optional LangSmith integration via LangChain callbacks.

### P5: Native Tool Calling Migration

Add `tool_calling_mode` config: `"xml"` (current default), `"native"` (uses `bind_tools()`). Native mode eliminates XML tag-repair code in `nodes.py:generate()` and stop sequences. Hybrid mode allows gradual migration.

### P6: Code Execution Security

Configurable import blocklists for dangerous modules. Filesystem access restrictions (read-only paths, write-allowed paths). Optional Docker-based isolation backend. Keep `exec()` mode as default for development.

### P7: Parallel Fan-out via `Send()`

For independent sub-tasks, use LangGraph's `Send()` to dispatch to multiple agents concurrently. Requires agents to accept isolated state dicts (enabled by Feature 3 REPL isolation).

### P8: Hierarchical Orchestration

Sub-supervisors for complex task decomposition. A supervisor can delegate to another `AgentTeam` instead of directly to an agent. Add only when validated need exists.

---

## Non-Goals

These are explicitly **never to be implemented**. They represent architectural anti-patterns or redundancies with LangGraph's native capabilities.

| Item | Reason |
|------|--------|
| **Custom workflow engine** | LangGraph `StateGraph` with conditional edges IS the workflow engine. Building `BaseStep`, `Edge`, `WorkflowRunner` on top is redundant. |
| **Custom middleware pipeline** | LangGraph uses callbacks and listeners for cross-cutting concerns. A `BaseMiddleware` with `on_request`/`on_response` creates a competing execution model. |
| **Structured output for entire execution loop** | BaseAgent executes code via REPL and returns computed results, not raw LLM text. Applying `with_structured_output()` to the main loop would change it from a code-executing agent into a text-generating agent. (Exception: the supervisor uses structured output for routing decisions -- this is fine.) |
| **Custom message types** | Use LangChain's existing `BaseMessage` subclasses (`HumanMessage`, `AIMessage`, `SystemMessage`) with the `name` field for sender attribution. No `AgentMessage` class. |
| **Frontend implementation** | Build only the streaming event API (`AgentEvent`, `EventType`, `run_stream()`). Frontend rendering is a separate project that consumes this API. |
| **Custom vector DB integration** | Use SQLite FTS5 (post-prototype P2) or LangGraph `Store` (post-prototype P3) for memory. No Pinecone/Weaviate/ChromaDB integration. |
| **Custom orchestration primitives** | No custom `TaskQueue`, `AgentPool`, or `MessageBus`. LangGraph's `StateGraph`, `Send()`, and `Store` provide these capabilities natively. |

---

## Planned File Additions

For current file structure, see `.claude/baseagent_reference.md`.

New files to be created by future features:
- `BaseAgent/multi_agent/` -- New subpackage: `state.py`, `orchestrator.py` (Feature 8)
- `BaseAgent/tests/test_multi_agent.py` -- Feature 8 tests
- `examples/13_multi_agent.py` -- AgentTeam demo (Feature 8)

---

## Key Design Decisions

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Agent composition: subgraphs vs. multiple instances | **Option A (multiple instances)** for prototype | Validate interaction patterns before refactoring `configure()`. Option B is post-prototype P1. |
| Orchestration pattern | **Flat supervisor** first | Avoid premature complexity. Hierarchical is post-prototype P8. |
| REPL isolation strategy | **Per-instance namespace** with module-level fallback | Safety for concurrency without breaking existing callers. |
| Supervisor implementation | **Custom `StateGraph`**, not `langgraph-prebuilt` | BaseAgent owns its graph topology; `create_supervisor()` hides too much. |
| Cross-agent memory | **None for prototype**; LangGraph `Store` post-prototype | Orchestrator state (`results` dict) is sufficient for the prototype. |
| Async API timing | **Before orchestration** (Feature 7 before 8) | Orchestrator benefits from `await agent.run()` for streaming and non-blocking dispatch. |
| Context window timing | **Implemented** (Feature 5 ✅) | Sliding window via `max_context_messages`; supervisor-level isolation is a Feature 8 design constraint. |
| Skill loading strategy | **Spec-driven targeted loading** (load by name, not glob) | Each agent needs 1-3 skills. Load only what's specified in `AgentSpec.skill_names`. Legacy glob preserved when `spec=None`. |
| Skill prompt injection | **Progressive disclosure** (metadata-only initial prompt) | Catalog in system prompt, full body loaded on demand by retriever. No threshold -- uniform behavior regardless of skill count. |
| Skill retrieval independence | **Separate `_select_skills_for_prompt()`** from tool retrieval | Skill retrieval is always-on (`skill_retrieval=True`); tool retrieval is opt-in (`use_tool_retriever`). Different concerns, different LLM calls. |


---

## BaseAgent Workflow Architecture

```
AgentTeam -- supervisor LLM coordinates specialist agents
  |
  +-- oncology_agent          disease ontology analyst + domain validator
  |     tools: ols4_lookup
  |     skills: ontology-design, biomedical-validation
  |
  +-- database_agent          biomedical database scout
  |     tools: ols4_lookup, string_db_search, biocontext_query
  |     skills: database-evaluation
  |
  +-- software_engineer_agent parser writer + data extraction
  |     tools: run_python_repl              ^
  |     skills: parser-development          | re-route with feedback
  |                                         |
  +-- mapping_agent           ontology alignment
  |     tools: run_python_repl              |
  |     skills: ontology-mapping            |
  |     -- signals needs_revision ----------+
  |
  +-- memgraph_agent          Cypher export + graph validation
        tools: run_python_repl
        skills: memgraph-export
```

**Event flow for a frontend rendering this workflow:**

```
SUPERVISOR_DECISION  {next_agent: "oncology_agent",
                      rationale: "Begin with disease ontology definition"}
AGENT_START          {agent: "oncology_agent"}
  THINKING           "Analyzing the reference OWL ontology for Alzheimer's disease..."
  CODE_EXECUTING     "import owlready2; onto = get_ontology('ad_reference.owl').load()"
  CODE_RESULT        "Proposed schema: 47 entity types, 23 relationship types"
AGENT_COMPLETE       {agent: "oncology_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "HITL",
                      rationale: "Ontology draft ready for user confirmation"}
APPROVAL_REQUIRED    {message: "Ontology defined (47 entities, 23 relationships). Confirm to proceed?"}
  ... user reviews and approves ...
SUPERVISOR_DECISION  {next_agent: "database_agent",
                      rationale: "Ontology confirmed, identify source databases"}
AGENT_START          {agent: "database_agent"}
  ...
AGENT_COMPLETE       {agent: "database_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "HITL",
                      rationale: "Database selection ready for user confirmation"}
APPROVAL_REQUIRED    {message: "Selected 4 databases (STRING, OMIM, UniProt, ChEBI). Confirm?"}
  ... user reviews and approves ...
SUPERVISOR_DECISION  {next_agent: "software_engineer_agent",
                      rationale: "Sources confirmed, begin data extraction"}
AGENT_START          {agent: "software_engineer_agent"}
  ...
AGENT_COMPLETE       {agent: "software_engineer_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "mapping_agent",
                      rationale: "Extracted data ready for ontology alignment"}
AGENT_START          {agent: "mapping_agent"}
  ...
AGENT_COMPLETE       {agent: "mapping_agent", status: "needs_revision",
                      feedback: "STRING TSV missing 'entity_type' column required by ontology schema"}
SUPERVISOR_DECISION  {next_agent: "software_engineer_agent",
                      rationale: "Mapping failed -- re-routing with parser feedback"}
AGENT_START          {agent: "software_engineer_agent"}
  ...
AGENT_COMPLETE       {agent: "software_engineer_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "mapping_agent",
                      rationale: "Re-extracted data ready, retry ontology alignment"}
AGENT_START          {agent: "mapping_agent"}
  ...
AGENT_COMPLETE       {agent: "mapping_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "memgraph_agent",
                      rationale: "Mapping complete, export to graph database"}
AGENT_START          {agent: "memgraph_agent"}
  ...
AGENT_COMPLETE       {agent: "memgraph_agent", status: "success"}
SUPERVISOR_DECISION  {next_agent: "FINISH"}
```

---

## Backwards Compatibility Constraints

All changes must preserve the existing single-agent API without modification:

```python
# Must continue to work unchanged, with identical behavior:
agent = BaseAgent(llm="claude-sonnet-4-20250514")
log, answer = agent.run_sync("What is 2+2?")

agent2 = BaseAgent(require_approval="always")
log, payload = agent2.run_sync("Write a bash script")
if agent2.is_interrupted:
    log, answer = agent2.resume_sync()
```

**Rules:**
- `AgentSpec` is optional. When `spec=None`, all defaults are identical to today.
- `run_sync()` replaces `run()` as the synchronous entry point. `run()` becomes async.
- `run_python_repl()` without `namespace` falls back to the module-level global (no breakage for external callers).
- `AgentTeam` lives in `BaseAgent/multi_agent/`. No changes to existing top-level imports.
- `BaseAgent` is never subclassed by the multi-agent system; it is composed.
- New `BaseAgentConfig` fields all default to `None` (disabled), preserving current behavior.
