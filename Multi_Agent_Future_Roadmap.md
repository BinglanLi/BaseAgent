# CuraAgent: Multi-Agent Feature Roadmap

## Goal

CuraAgent is a multi-agent AI system that autonomously develops disease-specific knowledge graphs. Given a user prompt specifying a disease scope (e.g., "Alzheimer's disease", "tuberculosis"), a template codebase, and a reference ontology, CuraAgent produces a `<disease>-KG` -- a Python-based codebase for building a disease-specific, memgraph-compatible, OWL-guided knowledge graph.

Building a biomedical knowledge graph entails several critical steps: **define and confirm ontology -> identify source databases -> extract data via database-specific parsers -> map extracted data to ontology -> export mapped data to memgraph**. Each step may require a team of expert agents coordinated by a supervisor agent. CuraAgent supports iterative human-in-the-loop conversations at each step -- for example, confirming ontology structure before proceeding, or specifying exact biomedical entities to extract from a source database.

BaseAgent provides the foundation: LangGraph `StateGraph`, multi-provider LLM support, persistent checkpointing, interrupt/resume, typed streaming events, and a skills system for behavioral parameterization. This roadmap describes the framework-level features that must be built on top of BaseAgent to realize CuraAgent. It focuses on agentic AI infrastructure, not on implementing CuraAgent's specific domain logic (ontology parsing, database connectors, etc.).

---

## Success Criteria

**Prototype is complete when:**

1. A `WorkflowOrchestrator` executes a configurable sequential pipeline (e.g., the 5-step CuraAgent workflow), with HITL checkpoints between steps.
2. At least one workflow step uses a `MultiAgentOrchestrator` with a supervisor + 2 specialist agents.
3. At least one agent connects to a remote MCP server (e.g., OLS4, biocontext.ai) and retrieves structured biomedical data.
4. Agents operate with isolated REPL namespaces -- concurrent execution does not corrupt shared state.
5. Each agent has a distinct identity (`AgentSpec`: name, role, tool subset, skill subset, optional model override).
6. The system emits multi-agent `AgentEvent` types (`AGENT_START`, `AGENT_COMPLETE`, `SUPERVISOR_DECISION`, `WORKFLOW_STEP_START`, `WORKFLOW_STEP_COMPLETE`) that a frontend can render.
7. Context window management prevents overflow during multi-step workflows.
8. Structured errors propagate from sub-agents to supervisors with retry/reroute semantics.
9. `run()` is async-first; `run_sync()` preserves backward compatibility for scripts and notebooks.
10. The existing single-agent API (`BaseAgent(llm=...).run(task)`) continues to work unchanged.

---

## Existing Features (Do Not Reimplement)

These are complete. New work must build on -- not duplicate -- these APIs.

| Feature | Key APIs | Notes |
|---------|----------|-------|
| **Streaming events** | `run_stream()`, `AgentEvent`, `EventType` (`events.py`) | 10 event types including `APPROVAL_REQUIRED`. `AgentEvent.to_json()` produces SSE/WebSocket-ready payloads. |
| **Persistent checkpointing** | `SqliteSaver`, `checkpoint_db_path` config, `thread_id` param on `run()` | Default `":memory:"` preserves ephemeral behavior. File path enables cross-session resume. |
| **Human-in-the-loop** | `approval_gate` node, `interrupt()`, `resume()`, `reject()`, `require_approval` config | Graph pauses via LangGraph `interrupt()`; resumes via `Command(resume=...)`. Policies: `"never"` / `"always"` / `"dangerous_only"`. |
| **Agent Skills** | `Skill` model, `add_skill()`, `load_skills()`, SKILL.md format, `skills_directory` config | Markdown behavioral instructions injected into system prompt. `trigger="auto"` for retriever-managed; `trigger="manual"` for always-on. |
| **Multi-provider LLM** | `get_llm()`, `SourceType`, `UsageMetrics` (`llm.py`) | 9 providers: OpenAI, AzureOpenAI, Anthropic, AnthropicFoundry, Gemini, Ollama, Bedrock, Groq, Custom. |
| **Resource management** | `ResourceManager`, `selected` flag on all resource models, `add_tool()`, `add_mcp()` (stdio only) | Four resource types: `Tool`, `DataLakeItem`, `Library`, `Skill`. `selected` flag controls prompt inclusion. |
| **Tool retriever** | `ToolRetriever.prompt_based_retrieval()`, `_RESOURCE_SELECTION_PROMPT` | LLM-based selection of tools, data, libraries, and skills when `use_tool_retriever=True`. |

---

## Prototype Feature Specifications

### Feature 1: MCP Overhaul

**Priority:** BLOCKER -- no biomedical database access without this.

**Current state:** `add_mcp()` silently skips any server with a `url` or `type: "remote"` field (`base_agent.py:335`). `make_mcp_wrapper` returns an unawaited `Task` object in Jupyter contexts due to a bug in the async/sync bridge. 9 servers in `mcp_biocontext_auto.yaml` are silently skipped.

**Phase 1 -- Async/sync bridge fix (one-line, ships immediately)**

In `make_mcp_wrapper`, remove the `get_running_loop` branch. `nest_asyncio` is already applied at `add_mcp()` entry; `asyncio.run()` works in both Jupyter and scripts:

```python
# Remove try/except block, replace with:
return asyncio.run(async_tool_call())
```

**Phase 2 -- Remote transport (unblocks OLS4 and biocontext servers)**

Dispatch on config in `add_mcp()`:

```python
if "url" in server_config:
    from mcp.client.streamable_http import streamablehttp_client
    async with streamablehttp_client(url, headers=headers) as (read, write, _):
        async with ClientSession(read, write) as session:
            # Same discover/wrapper pattern as stdio
else:
    # Existing stdio_client path -- unchanged
```

**Phase 3 -- Auth headers for remote servers**

Add optional `headers` dict to server config schema. Support `${ENV_VAR}` interpolation (mirrors existing env var handling). Thread headers into `streamablehttp_client` call:

```yaml
servers:
  biocontext:
    url: https://mcp.biocontext.ai/mcp/
    headers:
      Authorization: "Bearer ${BIOCONTEXT_API_KEY}"
```

**Files to modify:**
- `BaseAgent/base_agent.py` -- transport dispatch in `add_mcp()`, async fix in `make_mcp_wrapper`, headers threading
- `BaseAgent/tests/test_add_mcp.py` -- remote transport tests (mock `streamablehttp_client`)

---

### Feature 2: Agent Identity (`AgentSpec` + System Prompt Parameterization)

**Priority:** BLOCKER -- no distinct agent personas without this.

**Current state:** `_SYSTEM_PROMPT_HEADER` in `prompts.py:7-9` hardcodes `"You are a helpful biomedical assistant..."`. No concept of agent name, role, or persona. No mechanism to give different agents different tool/skill subsets at instantiation time.

**Phase 1 -- `AgentSpec` dataclass**

New file `BaseAgent/agent_spec.py`:

```python
from dataclasses import dataclass, field
from BaseAgent.llm import SourceType

@dataclass
class AgentSpec:
    name: str                                   # e.g. "ontology_analyst", "parser_developer"
    role: str                                   # Injected into system prompt
    system_prompt_override: str | None = None   # Full override; None = use default template
    tool_names: list[str] | None = None         # Tool subset; None = all
    skill_names: list[str] | None = None        # Skill subset; None = all
    llm: str | None = None                      # Model override per-agent
    source: SourceType | None = None            # Provider override per-agent
    temperature: float | None = None
```

**Phase 2 -- Wire `AgentSpec` into `BaseAgent`**

`BaseAgent.__init__` gains `spec: AgentSpec | None = None`. When provided:
- `self.name = spec.name` (defaults to `"agent"` when `spec=None`)
- `spec.role` fills `{role_description}` in `_SYSTEM_PROMPT_HEADER`
- `spec.tool_names` calls `resource_manager.select_tools_by_names()` after loading
- `spec.skill_names` calls `resource_manager.select_skills_by_names()` after loading
- `spec.llm` / `spec.source` / `spec.temperature` override the corresponding `default_config` values

**Phase 3 -- Parameterize system prompt header**

Replace the hardcoded framing in `_SYSTEM_PROMPT_HEADER`:

```python
# Before:
"You are a helpful biomedical assistant assigned with the task of problem-solving."
# After:
"You are {role_description}."
```

`_generate_system_prompt()` fills `role_description` from `self.spec.role` when a spec is set, or from the original default string when not.

**Backwards compatibility:** `spec=None` produces identical behavior to today.

**Files to modify:**
- `BaseAgent/agent_spec.py` (new) -- `AgentSpec` dataclass
- `BaseAgent/base_agent.py` -- `__init__` (accept `spec`), `configure` (apply spec overrides)
- `BaseAgent/prompts.py` -- `{role_description}` slot in `_SYSTEM_PROMPT_HEADER`
- `BaseAgent/__init__.py` -- export `AgentSpec`

---

### Feature 3: REPL Namespace Isolation

**Priority:** BLOCKER -- concurrent agents corrupt each other's variables, plots, and injected functions without this.

**Current state:** `_persistent_namespace` is a module-level global dict in `support_tools.py`. `inject_custom_functions_to_repl()` in `tool_bridge.py` writes to `builtins._BaseAgent_custom_functions` and targets the same global. `_captured_plots` and `_base_agent_patched` are also module-level globals. Multiple concurrent agents would overwrite each other's state.

**Phase 1 -- Parameterize `run_python_repl`**

```python
def run_python_repl(command: str, namespace: dict | None = None) -> str:
    # When namespace is provided, exec() uses it
    # When None, falls back to module-level _persistent_namespace for backwards compat
```

**Phase 2 -- Per-instance namespace in `BaseAgent`**

`BaseAgent.__init__` initializes `self._repl_namespace: dict = {}`. `NodeExecutor.execute()` passes `agent._repl_namespace` to `run_python_repl()`.

**Phase 3 -- Parameterize `inject_custom_functions_to_repl`**

```python
def inject_custom_functions_to_repl(custom_functions, namespace: dict | None = None):
    # Targets a specific namespace; None falls back to global
```

**Phase 4 -- Per-instance `PlotCapture`**

Replace module-level `_captured_plots` / `_base_agent_patched` with a per-instance class:

```python
class PlotCapture:
    def __init__(self):
        self._plots: list[str] = []
        self._patched: bool = False
    def apply_patches(self) -> None: ...
    def get_plots(self) -> list[str]: ...
    def clear(self) -> None: ...
```

`BaseAgent` instantiates `self._plot_capture = PlotCapture()` and passes it into the execute node.

**Risk mitigation:** Keep the module-level `_persistent_namespace` as a fallback. Grep for all direct imports before changing:
- `BaseAgent/tools/support_tools.py` -- definition site
- `BaseAgent/utils/tool_bridge.py` -- reads the global directly
- Test fixtures referencing the global

**Files to modify:**
- `BaseAgent/tools/support_tools.py` -- `namespace` param on `run_python_repl`, `PlotCapture` class
- `BaseAgent/utils/tool_bridge.py` -- `namespace` param on `inject_custom_functions_to_repl`
- `BaseAgent/base_agent.py` -- `self._repl_namespace`, `self._plot_capture` in `__init__`
- `BaseAgent/nodes.py` -- pass namespace and plot capture through `execute()`
- `BaseAgent/tests/test_repl_isolation.py` (new) -- verify namespace isolation between instances

---

### Feature 4: Extract Subgraph

**Priority:** MEDIUM -- low-risk prep that enables LangGraph-native composition later.

**Current state:** `configure()` at `base_agent.py:742` builds and compiles the full `StateGraph` inline. No way to get an uncompiled graph for embedding in a parent graph.

**Single phase -- Split `configure()` into two methods:**

```python
def get_subgraph(self, self_critic=False, test_time_scale_round=0) -> StateGraph:
    """Return this agent's workflow as an uncompiled StateGraph."""
    # All current configure() logic minus .compile()
    return workflow

def configure(self, self_critic=False, test_time_scale_round=0):
    workflow = self.get_subgraph(self_critic, test_time_scale_round)
    self.checkpointer = self._create_checkpointer()
    self.app = workflow.compile(checkpointer=self.checkpointer)
```

**Files to modify:**
- `BaseAgent/base_agent.py` -- split `configure()` into `get_subgraph()` + `configure()`

---

### Feature 5: Context Window Management

**Priority:** HIGH -- multi-agent amplifies unbounded message history from a limitation into a blocker.

**Current state:** All messages kept in `state["input"]` list indefinitely. Output truncated to 10K chars in `nodes.py:execute()`, but message count is unlimited. `recursion_limit` hardcoded to 500 in `base_agent.py`.

**Phase 1 -- Sliding window for single-agent**

- Preserve system prompt message + first user message + last N messages
- Token budget check before LLM calls in `generate` node
- New `BaseAgentConfig` fields:
  - `max_context_messages: int | None = None` (default `None` = disabled, preserves current behavior)
  - `context_strategy: str | None = None` (`"sliding_window"` | `None`)

**Phase 2 -- Supervisor-level truncation (after Feature 8)**

- The orchestrator passes only each agent's final result string to the supervisor -- not the agent's full conversation history
- Full histories live in per-agent checkpointers and are accessible for debugging but don't bloat supervisor context

**Files to modify:**
- `BaseAgent/nodes.py` -- context truncation before LLM invoke in `generate` node
- `BaseAgent/config.py` -- `max_context_messages`, `context_strategy` fields + env var overrides
- `BaseAgent/multi_agent/orchestrator.py` (Phase 2) -- supervisor receives summarized results only

---

### Feature 6: Error Handling + Termination Conditions

**Priority:** HIGH -- graceful degradation; essential for supervisor retry logic.

**Current state:** `execute` node in `nodes.py` has no try/except around `run_with_timeout`. Generate node retries on parse failure up to 2 times (`critic_count` counter). `recursion_limit` hardcoded to 500. No structured error types. No mechanism for a sub-agent to signal structured failure to a parent.

**Phase 1 -- Structured error types**

New file `BaseAgent/errors.py`:

```python
class BaseAgentError(Exception): ...
class ExecutionError(BaseAgentError): ...
class ParseError(BaseAgentError): ...
class TimeoutError(BaseAgentError): ...
class LLMError(BaseAgentError): ...
class BudgetExceededError(BaseAgentError): ...
```

Wrap `execute` node in try/except with structured error context. Errors emitted as `AgentEvent(type=ERROR)` for frontend display.

**Phase 2 -- Configurable termination conditions**

New `BaseAgentConfig` fields:
- `max_iterations: int | None = None` (replaces hardcoded `recursion_limit=500`)
- `max_cost: float | None = None`
- `max_consecutive_errors: int | None = None`

Check conditions in `routing_function` before each generate cycle. Cost tracking uses existing `UsageMetrics` from `llm.py`.

**Phase 3 -- `AgentResult` for inter-agent error propagation (after Feature 8)**

New file `BaseAgent/multi_agent/types.py`:

```python
from pydantic import BaseModel
from typing import Literal

class AgentResult(BaseModel):
    agent_name: str
    status: Literal["success", "error", "timeout"]
    output: str
    error: str | None = None
    usage: list = []  # list[UsageMetrics]
```

Supervisor receives `AgentResult` instead of raw strings and can retry, reroute, or abort based on `status`.

**Files to modify:**
- `BaseAgent/errors.py` (new) -- structured error types
- `BaseAgent/nodes.py` -- error handling in `execute` and `generate` nodes
- `BaseAgent/config.py` -- termination configuration fields + env var overrides
- `BaseAgent/multi_agent/types.py` (new, Phase 3) -- `AgentResult`

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
3. Writes the result to `state["results"][agent.name]`
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

### Feature 9: Workflow Orchestration (CuraAgent Pipeline)

**Priority:** HIGH -- the core CuraAgent architecture.

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

**Phase 3 -- CuraAgent workflow definition (demo/example)**

Example instantiation of the full CuraAgent pipeline:

```python
from BaseAgent import BaseAgent
from BaseAgent.agent_spec import AgentSpec
from BaseAgent.multi_agent import MultiAgentOrchestrator, WorkflowOrchestrator, WorkflowStep

# Step 1: Define ontology (supervisor + 2 specialists)
ontology_team = MultiAgentOrchestrator(agents=[
    BaseAgent(spec=AgentSpec(
        name="ontology_analyst",
        role="An ontology analyst that reads OWL reference ontologies and proposes "
             "a disease-specific schema with entity types and relationship types.",
        skill_names=["ontology-design"],
    )),
    BaseAgent(spec=AgentSpec(
        name="domain_expert",
        role="A biomedical domain expert that validates proposed ontology entities "
             "and relationships against current literature and clinical knowledge.",
        skill_names=["biomedical-validation"],
    )),
], supervisor_llm="claude-sonnet-4-20250514")

# Step 2: Identify source databases (single agent)
database_scout = BaseAgent(spec=AgentSpec(
    name="database_scout",
    role="A database specialist that identifies and evaluates biomedical databases "
         "for extracting disease-specific entities and relationships. Considers "
         "access methods, data formats, coverage, and update frequency.",
    tool_names=["ols4_lookup", "string_db_search", "biocontext_query"],
    skill_names=["database-evaluation"],
))

# Step 3: Extract data via parsers (supervisor + 2 specialists)
extraction_team = MultiAgentOrchestrator(agents=[
    BaseAgent(spec=AgentSpec(
        name="parser_developer",
        role="A software engineer that develops database-specific parser scripts "
             "to extract biomedical entities and relationships into intermediate "
             "CSV/TSV files following the confirmed ontology schema.",
        tool_names=["run_python_repl"],
        skill_names=["parser-development"],
    )),
    BaseAgent(spec=AgentSpec(
        name="qa_agent",
        role="A quality assurance agent that validates extracted data against "
             "the ontology schema, checks for completeness, and reports issues.",
        tool_names=["run_python_repl"],
        skill_names=["data-validation"],
    )),
], supervisor_llm="claude-sonnet-4-20250514")

# Step 4: Map to ontology (single agent)
mapping_agent = BaseAgent(spec=AgentSpec(
    name="mapping_agent",
    role="An ontology mapping agent that aligns extracted entities to OWL ontology "
         "terms, resolves ambiguities, and produces standardized mapping files.",
    tool_names=["run_python_repl"],
    skill_names=["ontology-mapping"],
))

# Step 5: Export to memgraph (single agent)
export_agent = BaseAgent(spec=AgentSpec(
    name="export_agent",
    role="A graph database engineer that converts mapped data into memgraph-compatible "
         "Cypher import scripts and validates the resulting knowledge graph structure.",
    tool_names=["run_python_repl"],
    skill_names=["memgraph-export"],
))

# Assemble the CuraAgent pipeline
cura_agent = WorkflowOrchestrator(steps=[
    WorkflowStep("define_ontology", "Define and confirm disease ontology", ontology_team, hitl_checkpoint=True),
    WorkflowStep("identify_databases", "Identify source databases", database_scout, hitl_checkpoint=True),
    WorkflowStep("extract_data", "Extract data via parsers", extraction_team, hitl_checkpoint=False),
    WorkflowStep("map_to_ontology", "Map extracted data to ontology", mapping_agent, hitl_checkpoint=False),
    WorkflowStep("export_to_memgraph", "Export to memgraph", export_agent, hitl_checkpoint=False),
])

log, results = cura_agent.run_sync("Build an Alzheimer's disease knowledge graph")
```

**Files to modify:**
- `BaseAgent/multi_agent/workflow.py` (new) -- `WorkflowStep`, `WorkflowOrchestrator`
- `BaseAgent/multi_agent/__init__.py` -- export `WorkflowStep`, `WorkflowOrchestrator`
- `BaseAgent/events.py` -- 2 new `EventType` values
- `BaseAgent/tests/test_workflow.py` (new) -- workflow pipeline tests with mock agents
- `examples/08_multi_agent_workflow.py` (new) -- CuraAgent demo script

---

## Post-Prototype Feature Specifications

Deferred until the prototype is validated. Listed here to inform design decisions -- do not build hooks or abstractions for these unless they are zero-cost.

### P1: Subgraph Embedding (Option B Migration)

Migrate from Option A (multiple instances) to embedded LangGraph subgraphs. Each `BaseAgent`'s `get_subgraph()` (Feature 4) returns an uncompiled `StateGraph` that a parent graph embeds via `workflow.add_subgraph()`. Enables unified checkpointing, streaming, and time-travel across the full multi-agent system. Prerequisite: validated multi-agent interaction patterns from the prototype.

**Files:** `BaseAgent/base_agent.py`, `BaseAgent/multi_agent/orchestrator.py`

### P2: Semantic Memory / RAG

`MemoryStore` with SQLite FTS5 backend (no vector DB dependency). Agent-accessible tools: `write_memory(text, metadata)`, `search_memory(query, k)`. System prompt injection of top-k relevant entries. Useful for CuraAgent to persist and retrieve disease-specific findings across workflow steps.

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

## Proposed File Structure

```
BaseAgent/
  agent_spec.py                   # NEW (Feature 2) -- AgentSpec dataclass
  errors.py                       # NEW (Feature 6) -- Structured error types

  multi_agent/                    # NEW subpackage (Features 8-9)
    __init__.py                   # Exports: MultiAgentOrchestrator, SequentialOrchestrator,
    |                             #   WorkflowOrchestrator, WorkflowStep, AgentResult
    state.py                      # MultiAgentState TypedDict
    orchestrator.py               # Supervisor + Sequential orchestrators
    workflow.py                   # WorkflowStep + WorkflowOrchestrator (CuraAgent pipeline)
    types.py                      # AgentResult (Feature 6 Phase 3)

  base_agent.py                   # MODIFIED: spec param, self._repl_namespace,
  |                               #   self._plot_capture, get_subgraph(),
  |                               #   remote MCP transport, async run()
  prompts.py                      # MODIFIED: {role_description} slot in _SYSTEM_PROMPT_HEADER
  config.py                       # MODIFIED: max_context_messages, context_strategy,
  |                               #   max_iterations, max_cost, max_consecutive_errors
  nodes.py                        # MODIFIED: namespace/plot_capture passthrough,
  |                               #   context truncation, error handling
  events.py                       # MODIFIED: 5 new EventType values
  state.py                        # UNCHANGED (single-agent state)
  __init__.py                     # MODIFIED: export AgentSpec

  tools/
    support_tools.py              # MODIFIED: PlotCapture class; namespace param on run_python_repl
  utils/
    tool_bridge.py                # MODIFIED: namespace param on inject_custom_functions_to_repl

  tests/
    test_multi_agent.py           # NEW -- orchestrator routing + round-trip tests
    test_workflow.py              # NEW -- workflow pipeline tests
    test_repl_isolation.py        # NEW -- verify namespace isolation between instances
    test_agent_spec.py            # NEW -- AgentSpec + prompt parameterization tests
    test_context_window.py        # NEW -- context truncation tests
    test_add_mcp.py               # EXTENDED -- remote transport tests via mock
```

---

## Implementation Order

```
                                                    Depends On       Effort
== GROUP A (parallel, no dependencies) =============================================
Feature 1   MCP Overhaul                            --               ~1 wk
  Phase 1   Async/sync bridge fix                                    1 line (day 1)
  Phase 2   Remote transport                                         ~3 days
  Phase 3   Auth headers                                             ~1 day

Feature 2   AgentSpec + prompt parameterization     --               ~1.5 days
  Phase 1   AgentSpec dataclass                                      ~0.5 day
  Phase 2   Wire into BaseAgent                                      ~0.5 day
  Phase 3   Parameterize _SYSTEM_PROMPT_HEADER                       ~0.5 day

Feature 3   REPL namespace isolation                --               ~2 days
  Phase 1   Parameterize run_python_repl                             ~0.5 day
  Phase 2   Per-instance namespace in BaseAgent                      ~0.5 day
  Phase 3   Parameterize inject_custom_functions                     ~0.5 day
  Phase 4   PlotCapture class                                        ~0.5 day

Feature 4   Extract subgraph                        --               ~0.5 day
  Single    Split configure() into get_subgraph() + configure()

== GROUP B (after Group A completes) ===============================================
Feature 5   Context window management               --               ~2 days
  Phase 1   Sliding window for single-agent
  Phase 2   Supervisor-level truncation (after Feature 8)

Feature 6   Error handling + termination            --               ~2 days
  Phase 1   Structured error types (errors.py)
  Phase 2   Configurable termination (config fields)
  Phase 3   AgentResult for inter-agent propagation (after Feature 8)

== GROUP C (after Group B) =========================================================
Feature 7   Async-first API                         5, 6             ~1 week
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
  Phase 3   CuraAgent demo script
```

**Critical path:** `[1, 2, 3, 4 parallel] -> [5, 6 parallel] -> 7 -> 8 -> 9`

**Minimum viable prototype:** Features 1-6 + 8 (supervisor orchestrator only, no workflow layer)

**Full prototype:** All 9 features

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
| Context window timing | **Before orchestration** (Feature 5 before 8) | Multi-agent amplifies unbounded history from a limitation into a blocker. |
| Workflow architecture | **`WorkflowOrchestrator` wrapping `MultiAgentOrchestrator`s** | CuraAgent's sequential pipeline maps naturally to a workflow of orchestrated steps. |

---

## Resolved Conflicts Between Source Documents

| Conflict | Resolution |
|----------|------------|
| **Agent composition:** LangGraph subgraphs (`Future_Roadmap`) vs. multiple instances (`multi-agent-plan`) | Option A (multiple instances) first. `get_subgraph()` (Feature 4) is low-risk prep for Option B later. |
| **Async API:** `Future_Roadmap` lists it as item 3 (before composition); `multi-agent-plan` defers it | **Prototype Feature 7**, sequenced before orchestration. Frontend compat and efficient agent dispatch require it. |
| **Memory:** `Future_Roadmap` item 6 (SQLite FTS5, medium priority) vs. `multi-agent-plan` Phase 4 (LangGraph Store, deferred) | **Both deferred to post-prototype** (P2 and P3). Prototype uses orchestrator state for inter-agent data passing. |
| **Error handling:** Separate `errors.py` (`Future_Roadmap`) vs. `AgentResult` (`multi-agent-plan`) | **Both, phased.** `errors.py` for single-agent (Feature 6 Phase 1-2); `AgentResult` for inter-agent propagation (Feature 6 Phase 3). |
| **Context window:** `Future_Roadmap` says "anytime"; `multi-agent-plan` says "before orchestrator" | **Before orchestrator** (Feature 5 in Group B). Both docs agree this is important; sequencing is now explicit. |
| **Extract subgraph:** `Future_Roadmap` item 5 (as_subgraph, medium priority); `multi-agent-plan` Phase 1.4 (1 day) | **Prototype Feature 4** (0.5 day). Low risk, no reason to defer. Enables Option B later. |
| **`AgentSpec` location:** `multi-agent-plan` puts it in `multi_agent/agent_spec.py` | **`BaseAgent/agent_spec.py`** at package root. Used by single-agent too (persona without multi-agent). Re-exported from `multi_agent/__init__.py`. |

---

## CuraAgent Workflow Architecture

```
WorkflowOrchestrator (sequential pipeline, HITL between steps)
  |
  +-- Step 1: Define Ontology [hitl_checkpoint=True]
  |     +-- MultiAgentOrchestrator (supervisor + specialists)
  |           +-- ontology_analyst   -- reads OWL reference, proposes schema
  |           +-- domain_expert      -- validates against literature
  |     +-- [PAUSE] User confirms ontology before proceeding
  |
  +-- Step 2: Identify Source Databases [hitl_checkpoint=True]
  |     +-- Single agent: database_scout
  |           -- evaluates databases for access, format, coverage
  |     +-- [PAUSE] User confirms data sources before proceeding
  |
  +-- Step 3: Extract Data via Parsers
  |     +-- MultiAgentOrchestrator (supervisor + specialists)
  |           +-- parser_developer   -- writes extraction scripts
  |           +-- qa_agent           -- validates extracted data
  |
  +-- Step 4: Map to Ontology
  |     +-- Single agent: mapping_agent
  |           -- aligns entities to OWL terms, resolves ambiguities
  |
  +-- Step 5: Export to Memgraph
        +-- Single agent: export_agent
              -- generates Cypher import scripts, validates graph
```

**Event flow for a frontend rendering this workflow:**

```
WORKFLOW_STEP_START  {step: "define_ontology", index: 0}
  AGENT_START        {agent: "ontology_analyst"}
    THINKING         "Analyzing the reference OWL ontology..."
    CODE_EXECUTING   "import owlready2; onto = ..."
    CODE_RESULT      "Found 47 entity types, 23 relationship types"
  AGENT_COMPLETE     {agent: "ontology_analyst", status: "success"}
  SUPERVISOR_DECISION {next_agent: "domain_expert", rationale: "Need validation..."}
  AGENT_START        {agent: "domain_expert"}
    ...
  AGENT_COMPLETE     {agent: "domain_expert", status: "success"}
  SUPERVISOR_DECISION {next_agent: "FINISH"}
WORKFLOW_STEP_COMPLETE {step: "define_ontology", index: 0}
APPROVAL_REQUIRED    {message: "Ontology defined. Review before proceeding?"}
  ... user reviews and approves ...
WORKFLOW_STEP_START  {step: "identify_databases", index: 1}
  ...
```

---

## Backwards Compatibility Constraints

All changes must preserve the existing single-agent API without modification:

```python
# Must continue to work unchanged, with identical behavior:
agent = BaseAgent(llm="claude-sonnet-4-20250514")
log, answer = agent.run_sync("What is 2+2?")

agent2 = BaseAgent(require_approval="dangerous_only")
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
