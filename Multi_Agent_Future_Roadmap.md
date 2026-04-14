# BaseAgent: Multi-Agent Feature Roadmap

## Goal

This roadmap describes the features needed to complete BaseAgent's multi-agent orchestration capabilities. It focuses on agentic AI infrastructure, not on domain-specific logic (ontology parsing, database connectors, etc.).

---

## Success Criteria

**Prototype is complete when:**

1. A `WorkflowOrchestrator` executes a configurable sequential pipeline (e.g., the 5-step BaseAgent workflow), with HITL checkpoints between steps.
2. At least one workflow step uses a hierarchical `MultiAgentOrchestrator` with a supervisor delegating to specialist agents, demonstrating iterative inter-agent feedback (e.g., mapping agent requests parser re-extraction).
3. At least one agent connects to a remote MCP server (e.g., OLS4, biocontext.ai) and retrieves structured biomedical data.
4. Agents operate with isolated REPL namespaces -- concurrent execution does not corrupt shared state.
5. Each agent has a distinct identity (`AgentSpec`: name, role, tool subset, skill subset, optional model override).
6. The system emits multi-agent `AgentEvent` types (`AGENT_START`, `AGENT_COMPLETE`, `SUPERVISOR_DECISION`, `WORKFLOW_STEP_START`, `WORKFLOW_STEP_COMPLETE`) that a frontend can render.
7. Context window management prevents overflow during multi-step workflows.
8. Structured errors propagate from sub-agents to supervisors with retry/reroute semantics.
9. `run()` is async-first; `run_sync()` preserves backward compatibility for scripts and notebooks.
10. The existing single-agent API (`BaseAgent(llm=...).run(task)`) continues to work unchanged.

---

## Completed Features

Features 1-6 and 10 are implemented and tested. See `.claude/baseagent_modules.md` for current API details.

| Feature | Summary | Tests |
|---------|---------|-------|
| **Feature 1: MCP Overhaul** | Remote Streamable HTTP transport, auth headers with `${ENV_VAR}` interpolation, async/sync bridge fix | 13 unit tests |
| **Feature 2: AgentSpec** | `AgentSpec` dataclass for agent identity; `{role_description}` parameterization in system prompt | 22 unit tests |
| **Feature 3: REPL Namespace Isolation** | Per-instance `_repl_namespace` and `PlotCapture`; `namespace` param on `run_python_repl` and `inject_custom_functions_to_repl` | unit tests |
| **Feature 4: Extract Subgraph** | `get_subgraph()` returns uncompiled `StateGraph` for LangGraph composition; `configure()` calls it then compiles | 18 unit tests |
| **Feature 5: Context Window Management** | Sliding window truncation in `generate` and `execute_self_critic` nodes; `max_context_messages` config field; `BASE_AGENT_MAX_CONTEXT_MESSAGES` env var | 21 unit tests |
| **Feature 6: Error Handling + Termination** | Structured error hierarchy (`errors.py`); `max_iterations`, `max_cost`, `max_consecutive_errors` config fields; `LLMError` wrapping; per-run cost budget via `_run_usage_start` index | 50 unit tests |
| **Feature 10: Skills System Overhaul** | Spec-driven targeted loading, progressive disclosure (catalog mode), bundled resources (`read_skill_resource`), functional `tools` field | 69 unit tests |

---

## Prototype Feature Specifications

---

### Feature 7: Async-First API

**Priority:** HIGH -- frontend cannot block on synchronous `run()`. Non-blocking execution enables efficient multi-agent orchestration and web integration.

**Current state:** `run()` at `base_agent.py:956` is synchronous, uses `app.stream()`. `run_stream()` is already async.

**Phase 1 -- Core async conversion**

- Convert `run()` to `async def run()` using `app.astream()`
- Update `resume()` and `reject()` to async equivalents

**Phase 2 -- Backward compatibility wrapper**

- Add `run_sync()` convenience wrapper via `asyncio.run()`
- `run_sync()` matches current `run()` signature and return type exactly
- Existing notebooks/scripts switch to `run_sync()` with zero behavior change
- Update examples to use `run_sync()`

**Files to modify:**
- `BaseAgent/base_agent.py` -- convert `run()`, `resume()`, `reject()` to async; add `run_sync()`, `resume_sync()`, `reject_sync()`

---

### Feature 8: Multi-Agent Orchestration

**Priority:** BLOCKER -- no agent coordination without this.

**Depends on:** Feature 2 (AgentSpec), Feature 3 (REPL isolation), Feature 7 (async API).

**Current state:** No orchestration layer, no multi-agent state schema, no supervisor logic. `BaseAgent` is single-agent only. `BaseAgent/multi_agent/` does not exist.

**Architecture decision:** Use **Option A** (multiple `BaseAgent` instances coordinated by an external orchestrator) for the prototype. Each sub-agent's `run()` executes to completion inside its agent node. Option B (embedded LangGraph subgraphs) is deferred to post-prototype.

**Phase 1 -- Multi-agent state schema**

New file `BaseAgent/multi_agent/state.py`:

```python
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Shared conversation
    sender: str            # Name of the agent that last produced output
    next_agent: str | None # Routing decision; None = done
    task: str              # Original user task
    results: dict[str, str]  # agent_name -> final output string
```

No `task_queue` (LangGraph's `Send()` handles fan-out if needed later). No custom `AgentMessage` class (use LangChain's existing message types with the `name` field for sender attribution).

**Phase 2 -- `MultiAgentOrchestrator`**

New file `BaseAgent/multi_agent/orchestrator.py`:

```python
class MultiAgentOrchestrator:
    def __init__(
        self,
        agents: list[BaseAgent],          # Each must have a unique spec.name
        supervisor_llm: str | None = None,
        supervisor_source: SourceType | None = None,
        max_rounds: int = 10,
        checkpoint_db_path: str = ":memory:",
    ): ...

    async def run(self, task: str, thread_id: str | None = None) -> tuple[list, str]: ...
    def run_sync(self, task: str, thread_id: str | None = None) -> tuple[list, str]: ...
    async def run_stream(self, task: str, ...) -> AsyncIterator[AgentEvent]: ...
```

**Supervisor node logic:**
1. Receives `MultiAgentState` (messages + results so far)
2. Calls supervisor LLM with a prompt listing available agents (names, roles) and results so far
3. Uses `with_structured_output` to extract `{"next_agent": "<name_or_FINISH>"}`
4. Sets `state["next_agent"]`; `"FINISH"` maps to `None`

**Agent node logic (one node per registered agent):**
1. Extracts the latest task/instruction from `state["messages"]`
2. Calls `await agent.run(sub_task)` -- the full single-agent loop runs to completion
3. Writes **only the result string** to `state["results"][agent.name]` — do not forward the sub-agent's full message history (Feature 5 Phase 2 constraint: supervisor context must not include sub-agent conversation histories)
4. Appends the result as `AIMessage(name=agent.name, content=result)`
5. Returns updated state

**Routing:**
```python
def _route(self, state: MultiAgentState) -> str:
    next_agent = state.get("next_agent")
    if not next_agent or next_agent == "FINISH" or next_agent not in self.agents:
        return END
    return next_agent
```

**Graph topology:** `START -> supervisor -> [agent_a | agent_b | ...] -> supervisor -> ... -> END`

**Phase 3 -- `SequentialOrchestrator`**

A simpler pattern for fixed pipelines:

```python
class SequentialOrchestrator:
    def __init__(self, agents: list[BaseAgent], checkpoint_db_path: str = ":memory:"): ...
    async def run(self, task: str) -> tuple[list, str]: ...
    def run_sync(self, task: str) -> tuple[list, str]: ...
```

Hard-coded edge chain: `agent_0 -> agent_1 -> ... -> agent_n -> END`. Each agent receives the previous agent's output as its task.

**Phase 4 -- Multi-agent event types**

Add to `BaseAgent/events.py`:

```python
AGENT_START = "agent_start"
"""A sub-agent began executing its task. Metadata includes agent name and role."""

AGENT_COMPLETE = "agent_complete"
"""A sub-agent finished executing. Content is the result summary."""

SUPERVISOR_DECISION = "supervisor_decision"
"""Supervisor LLM decided which agent to route to next. Content is the decision rationale."""
```

Emit these from orchestrator `run_stream()` at agent dispatch and supervisor decision points.

**Files to modify:**
- `BaseAgent/multi_agent/__init__.py` (new) -- exports
- `BaseAgent/multi_agent/state.py` (new) -- `MultiAgentState`
- `BaseAgent/multi_agent/orchestrator.py` (new) -- `MultiAgentOrchestrator`, `SequentialOrchestrator`
- `BaseAgent/events.py` -- 3 new `EventType` values
- `BaseAgent/tests/test_multi_agent.py` (new) -- routing tests, round-trip tests with mock LLMs

---

### Feature 9: Workflow Orchestration (BaseAgent Pipeline)

**Priority:** HIGH -- the core BaseAgent multi-agent architecture.

**Depends on:** Feature 8 (MultiAgentOrchestrator, SequentialOrchestrator).

**Current state:** No workflow-level orchestration. No concept of a multi-step pipeline where each step can be a single agent or a supervisor+specialist team.

**Phase 1 -- `WorkflowStep` and `WorkflowOrchestrator`**

New file `BaseAgent/multi_agent/workflow.py`:

```python
from dataclasses import dataclass

@dataclass
class WorkflowStep:
    name: str                                    # e.g. "define_ontology"
    description: str                             # Human-readable step description
    executor: BaseAgent | MultiAgentOrchestrator # Single agent or supervisor+team
    hitl_checkpoint: bool = False                # Pause for human review after step

class WorkflowOrchestrator:
    def __init__(
        self,
        steps: list[WorkflowStep],
        checkpoint_db_path: str = ":memory:",
    ): ...

    async def run(self, task: str, thread_id: str | None = None) -> tuple[list, dict[str, str]]: ...
    def run_sync(self, task: str, ...) -> tuple[list, dict[str, str]]: ...
    async def run_stream(self, task: str, ...) -> AsyncIterator[AgentEvent]: ...
    async def resume(self) -> tuple[list, dict[str, str]]: ...
```

**Execution model:**
1. Iterates through `steps` sequentially
2. For each step, calls `executor.run(task_with_previous_results)`
3. If `hitl_checkpoint=True`, calls `interrupt()` to pause for human review
4. On `resume()`, continues to next step
5. Returns `(log, results_dict)` where `results_dict` maps step names to outputs

**Phase 2 -- Workflow event types**

Add to `BaseAgent/events.py`:

```python
WORKFLOW_STEP_START = "workflow_step_start"
"""A workflow pipeline step began. Metadata includes step name and index."""

WORKFLOW_STEP_COMPLETE = "workflow_step_complete"
"""A workflow pipeline step completed. Content is the step output summary."""
```

**Phase 3 -- BaseAgent workflow definition (demo/example)**

A supervisor agent coordinate the knowledge graph construction task across a team of specialist agents registered with a single `MultiAgentOrchestrator`. The supervisor's routing logic encodes the pipeline order and handles HITL interrupts and iterative corrections without crossing step boundaries.

**Pipeline order** (enforced via the supervisor system prompt):
1. `oncology_agent` → propose disease-specific ontology schema → [HITL: user confirms]
2. `database_agent` → identify and evaluate source databases → [HITL: user confirms]
3. `software_engineer_agent` → write and run extraction / parser scripts
4. `mapping_agent` → align extracted data to ontology; if data is incomplete or incorrectly structured, returns `AgentResult(status="needs_revision", feedback="...")` → supervisor re-routes back to `software_engineer_agent` with the feedback injected as context
5. `memgraph_agent` → export mapped data to memgraph as a knowledge graph

Each agent specifies its skills via `AgentSpec.skill_names`. The `skills_directory` parameter tells the agent where to find skill subdirectories. Skills are loaded by name from `{skills_directory}/{skill_name}/SKILL.md` -- no glob, no load-all-then-filter.

```python
from BaseAgent import BaseAgent
from BaseAgent.agent_spec import AgentSpec
from BaseAgent.multi_agent import MultiAgentOrchestrator

SKILLS_DIR = "skills"  # Conventional: skills/<skill-name>/SKILL.md

agent = MultiAgentOrchestrator(
    agents=[
        BaseAgent(spec=AgentSpec(
            name="oncology_agent",
            role="A disease domain expert that reads OWL reference ontologies, proposes "
                 "a disease-specific schema with entity and relationship types, and validates "
                 "the schema against current biomedical literature and clinical knowledge.",
            tool_names=["ols4_lookup"],
            skill_names=["ontology-design", "biomedical-validation"],
        ), skills_directory=SKILLS_DIR),
        BaseAgent(spec=AgentSpec(
            name="database_agent",
            role="A biomedical database specialist that identifies and evaluates source databases "
                 "for extracting disease-specific entities and relationships. Considers access "
                 "methods, data formats, coverage, licensing, and update frequency.",
            tool_names=["ols4_lookup", "string_db_search", "biocontext_query"],
            skill_names=["database-evaluation"],
        ), skills_directory=SKILLS_DIR),
        BaseAgent(spec=AgentSpec(
            name="software_engineer_agent",
            role="A biomedical software engineer that develops database-specific parser scripts "
                 "to extract entities and relationships into intermediate CSV/TSV files following "
                 "the confirmed ontology schema. Revises parsers when the mapping agent signals "
                 "that extracted data is missing required columns or entity types.",
            tool_names=["run_python_repl"],
            skill_names=["parser-development"],
        ), skills_directory=SKILLS_DIR),
        BaseAgent(spec=AgentSpec(
            name="mapping_agent",
            role="An ontology mapping agent that aligns extracted entities to OWL ontology terms "
                 "and produces standardized mapping files. If extracted data is missing required "
                 "entity types or columns, signals needs_revision with specific feedback so the "
                 "supervisor can re-route to the software engineer for re-extraction.",
            tool_names=["run_python_repl"],
            skill_names=["ontology-mapping"],
        ), skills_directory=SKILLS_DIR),
        BaseAgent(spec=AgentSpec(
            name="memgraph_agent",
            role="A graph database engineer that converts mapped data into memgraph-compatible "
                 "Cypher import scripts and validates the resulting knowledge graph structure "
                 "against the confirmed ontology schema.",
            tool_names=["run_python_repl"],
            skill_names=["memgraph-export"],
        ), skills_directory=SKILLS_DIR),
    ],
    supervisor_llm="claude-sonnet-4-20250514",
)

log, result = agent.run_sync("Build an Alzheimer's disease knowledge graph")
```

**Files to modify:**
- `BaseAgent/multi_agent/workflow.py` (new) -- `WorkflowStep`, `WorkflowOrchestrator`
- `BaseAgent/multi_agent/__init__.py` -- export `WorkflowStep`, `WorkflowOrchestrator`
- `BaseAgent/events.py` -- 2 new `EventType` values
- `BaseAgent/tests/test_workflow.py` (new) -- workflow pipeline tests with mock agents
- `examples/08_multi_agent_workflow.py` (new) -- BaseAgent multi-agent demo script

---

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

### P7: Backward Workflow Navigation

Allow the system to go back to previous workflow steps. For example, if a user requests additional entities from a source database, the system reviews how the new data changes the ontology and re-executes from the appropriate step. Requires: workflow state tracking, step invalidation logic, re-execution strategy.

**Files:** `BaseAgent/multi_agent/workflow.py`

### P8: Parallel Fan-out via `Send()`

For independent sub-tasks, use LangGraph's `Send()` to dispatch to multiple agents concurrently. Requires agents to accept isolated state dicts (enabled by Feature 3 REPL isolation).

### P9: Hierarchical Orchestration

Sub-supervisors for complex task decomposition. A supervisor can delegate to another `MultiAgentOrchestrator` instead of directly to an agent. Add only when validated need exists.

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
- `BaseAgent/multi_agent/` -- New subpackage: `state.py`, `orchestrator.py`, `workflow.py`, `types.py` (Features 8-9)

---

## Implementation Order

```
                                                    Depends On       Effort
== GROUP C (after Group B) =========================================================
Feature 7   Async-first API                         5, 6 ✅          ~1 week
  Phase 1   Convert run()/resume()/reject() to async
  Phase 2   run_sync()/resume_sync()/reject_sync() wrappers

== GROUP D (after Group C) =========================================================
Feature 8   Multi-agent orchestration               2, 3, 7          ~3 days
  Phase 1   MultiAgentState schema
  Phase 2   MultiAgentOrchestrator (supervisor pattern)
  Phase 3   SequentialOrchestrator
  Phase 4   Multi-agent event types (AGENT_START, AGENT_COMPLETE, SUPERVISOR_DECISION)

== GROUP E (after Group D) =========================================================
Feature 9   Workflow orchestration                  8                ~3 days
  Phase 1   WorkflowStep + WorkflowOrchestrator
  Phase 2   Workflow event types (WORKFLOW_STEP_START, WORKFLOW_STEP_COMPLETE)
  Phase 3   BaseAgent multi-agent demo script
```

**Critical path:** `[1, 2, 3, 4, 5, 6, 10 ✅] -> 7 -> 8 -> 9`

**Minimum viable prototype:** Features 1-6 + 8 + 10 Phase 1-3 (supervisor orchestrator, spec-driven skills with progressive disclosure)

**Full prototype:** All 10 features

---

## Key Design Decisions

| Decision | Resolution | Rationale |
|----------|-----------|-----------|
| Agent composition: subgraphs vs. multiple instances | **Option A (multiple instances)** for prototype | Validate interaction patterns before refactoring `configure()`. Option B is post-prototype P1. |
| Orchestration pattern | **Flat supervisor** first | Avoid premature complexity. Hierarchical is post-prototype P9. |
| REPL isolation strategy | **Per-instance namespace** with module-level fallback | Safety for concurrency without breaking existing callers. |
| Supervisor implementation | **Custom `StateGraph`**, not `langgraph-prebuilt` | BaseAgent owns its graph topology; `create_supervisor()` hides too much. |
| Cross-agent memory | **None for prototype**; LangGraph `Store` post-prototype | Orchestrator state (`results` dict) is sufficient for the prototype. |
| Async API timing | **Before orchestration** (Feature 7 before 8) | Orchestrator benefits from `await agent.run()` for streaming and non-blocking dispatch. |
| Context window timing | **Implemented** (Feature 5 ✅) | Sliding window via `max_context_messages`; supervisor-level isolation is a Feature 8 design constraint. |
| Workflow architecture | **`WorkflowOrchestrator` wrapping `MultiAgentOrchestrator`s** | The sequential pipeline maps naturally to a workflow of orchestrated steps. |
| Skill loading strategy | **Spec-driven targeted loading** (load by name, not glob) | Each agent needs 1-3 skills. Load only what's specified in `AgentSpec.skill_names`. Legacy glob preserved when `spec=None`. |
| Skill prompt injection | **Progressive disclosure** (metadata-only initial prompt) | Catalog in system prompt, full body loaded on demand by retriever. No threshold -- uniform behavior regardless of skill count. |
| Skill retrieval independence | **Separate `_select_skills_for_prompt()`** from tool retrieval | Skill retrieval is always-on (`skill_retrieval=True`); tool retrieval is opt-in (`use_tool_retriever`). Different concerns, different LLM calls. |


---

## BaseAgent Workflow Architecture

```
MultiAgentOrchestrator -- BaseAgent supervisor
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
- `MultiAgentOrchestrator` and `WorkflowOrchestrator` live in `BaseAgent/multi_agent/`. No changes to existing top-level imports.
- `BaseAgent` is never subclassed by the multi-agent system; it is composed.
- New `BaseAgentConfig` fields all default to `None` (disabled), preserving current behavior.
