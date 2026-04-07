# Multi-Agent Readiness Assessment & Evolution Plan

## Current State Assessment

### Strengths (solid foundations)

1. **LangGraph `StateGraph` already in use** — The project uses `StateGraph` with conditional edges, the same framework powering LangGraph's multi-agent patterns (supervisor, hierarchical, swarm). Graph compilation, checkpointing, and streaming infrastructure is already wired up.

2. **`NodeExecutor` decouples nodes from the agent** — Graph nodes (`generate`, `execute`, `retrieve`, `approval_gate`) are methods on `NodeExecutor(agent)`, not closures inside `configure()`. This was partially addressed in the prior plan. Nodes are already independently testable with mock agents. The remaining work is making the *graph topology itself* extractable, not the node logic.

3. **Decoupled LLM factory** (`llm.py`) — `get_llm()` supports 9 providers and returns a generic `BaseChatModel`. Different agents can use different models (cheap model for routing, expensive one for reasoning) without changes to this module. `UsageMetrics` extraction works per-call, which matters for cost tracking in multi-agent scenarios.

4. **Resource management with `selected` flag** — `ResourceManager` with `select_tools_by_names()` / `select_data_by_names()` / `select_skills_by_names()` makes it feasible to give each agent a distinct subset of resources. The `selected` flag pattern is the right abstraction for per-agent scoping.

5. **Skills system provides behavioral parameterization** — `Skill` objects are markdown instruction sets injected into the system prompt. A specialized agent is already partway defined: it's a `BaseAgent` with a specific skill set, tool subset, and a role description. The skills system completes the behavioral half.

6. **Interrupt/resume infrastructure** — `approval_gate` + `interrupt()` + `Command(resume=...)` proves the codebase can handle inter-step coordination. This is architecturally similar to inter-agent handoffs.

### Gaps (work needed)

| # | Gap | Severity | Detail |
|---|-----|----------|--------|
| 1 | **`AgentState` tightly couples to the single-agent XML protocol** | **Critical** | `AgentState` has 4 fields (`input`, `next_step`, `pending_code`, `pending_language`) all bound to the `<think>/<execute>/<solution>` tag protocol of a single agent. Multi-agent needs per-agent message threads and a routing field, while preserving the tag protocol *within* each sub-agent's subgraph. |
| 2 | **Graph topology baked into `configure()`** | **High** | `NodeExecutor` extracted node *logic*, but `configure()` still builds and compiles a specific `retrieve → generate → [approval_gate →] execute → ... → END` topology inline. Composing agents requires returning an uncompiled `StateGraph` from a `get_subgraph()` method that a parent graph can embed. |
| 3 | **No agent identity** | **High** | No concept of agent name, role, or persona. `_generate_system_prompt()` produces a single prompt with hardcoded "biomedical assistant" framing (`_SYSTEM_PROMPT_HEADER` in `prompts.py`). Different agents need different system prompts, tool subsets, and model selections. |
| 4 | **Global REPL namespace** (upgraded from Medium) | **High** | `_persistent_namespace` is a **module-level global dict** in `support_tools.py`. `inject_custom_functions_to_repl()` also writes to `builtins._BaseAgent_custom_functions`. The matplotlib patches use module-level `_captured_plots` and a `_base_agent_patched` flag. Multiple concurrent agents corrupt each other's variables, plots, and injected functions. |
| 5 | **No orchestration layer** | **Medium** | No supervisor, router, or planner that can decompose a task and delegate to sub-agents. |
| 6 | **Shared mutable instance state in `run()`** | **Medium** | `self.log`, `self._execution_results`, `self.critic_count`, `self.user_task`, `self.thread_id`, `self._run_config`, `self._interrupted` are all set during `run()`. Concurrent runs on the same instance race. This matters less if each sub-agent is a separate instance, but is a latent bug. |
| 7 | **Unbounded message history (worsens in multi-agent)** | **Medium** | `state["input"]` grows indefinitely. Already acknowledged in CLAUDE.md for single-agent. Multi-agent amplifies this: a supervisor accumulating sub-agent outputs hits context limits faster. No truncation or summarization exists anywhere. |
| 8 | **System prompt generation is interleaved with resource logic** | **Medium** | `_generate_system_prompt()` mixes string formatting with ResourceManager queries and template assembly. Parameterizing it for different agent personas requires untangling prompt structure from resource selection. Mislabeled Low in the prior plan. |
| 9 | **No error propagation between agents** | **Low** | The graph handles parsing errors (retry 2x) and execution timeouts. No mechanism exists for a sub-agent to signal structured failure to a parent graph. |
| 10 | **Synchronous-only `run()`** | **Low** | `run()` blocks; `run_stream()` is async. Multi-agent benefits from async parallel execution, but LangGraph handles this internally and it can be deferred. |

---

## Critical Design Decisions

### Decision 1: Multiple instances vs. embedded subgraphs?

**Option A — Multiple `BaseAgent` instances composed by an external orchestrator.**
The orchestrator calls `agent.run()` on each sub-agent and stitches results together.

- **Pro:** Zero refactoring of `BaseAgent` internals. Complete isolation. Easy to reason about.
- **Con:** No shared checkpointing across the whole run. No LangGraph streaming/time-travel for the full system. Inter-agent communication is ad-hoc.

**Option B — `BaseAgent` subgraph embedded in a parent `StateGraph`.**
`configure()` returns an uncompiled `StateGraph`; a parent graph embeds each agent subgraph via `workflow.add_subgraph()`.

- **Pro:** Full LangGraph feature set (streaming, checkpointing, interrupt/resume across the multi-agent system). Clean state passing. One checkpointer for the whole system.
- **Con:** Requires refactoring `configure()` and `AgentState`. State schema design is more complex.

**Recommendation:** Implement **Option A first** for a working prototype. Migrate to **Option B** once multi-agent interaction patterns are validated. Option A ships value immediately; Option B is the long-term architecture.

### Decision 2: Flat supervisor vs. hierarchical?

Start with a **flat supervisor** that directly manages specialist agents. Add hierarchy only when task decomposition requires sub-supervisors — not by default.

### Decision 3: Shared REPL vs. isolated REPLs?

**Isolated REPLs.** Each agent gets its own namespace dict. Shared state between agents must be explicit (passed through orchestrator state), never implicit via a shared global.

### Decision 4: LangGraph-native vs. custom supervisor?

Use a **custom `StateGraph`-based supervisor**, consistent with BaseAgent's pattern of owning its graph topology. `langgraph-prebuilt`'s `create_supervisor()` hides too much for a framework that exposes fine-grained control.

### Decision 5: Shared memory implementation?

Use **LangGraph `Store`** for cross-agent persistent memory instead of building a custom `SharedMemory` class. It's already available via the checkpointer infrastructure and avoids reinventing the wheel.

---

## Implementation Plan

### Phase 1: Make `BaseAgent` composable (prerequisite for everything)

**Goal:** A single `BaseAgent` becomes a self-contained, configurable unit that can be instantiated with different personas. No multi-agent code yet; this phase makes the existing agent ready to participate in multi-agent.

#### 1.1 Add agent identity via `AgentSpec`

Create a lightweight spec that captures what makes one agent differ from another. Composes with (does not replace) `BaseAgentConfig`.

```python
# New file: BaseAgent/agent_spec.py
from dataclasses import dataclass, field
from BaseAgent.llm import SourceType

@dataclass
class AgentSpec:
    name: str                                   # e.g. "researcher", "coder", "reviewer"
    role: str                                   # Injected into system prompt (replaces "biomedical assistant")
    system_prompt_override: str | None = None   # Full override; None = use default template
    tool_names: list[str] | None = None         # Subset of tools; None = all
    skill_names: list[str] | None = None        # Subset of skills; None = all
    llm: str | None = None                      # Model override per-agent
    source: SourceType | None = None            # Provider override per-agent
    temperature: float | None = None            # Temperature override
```

`BaseAgent.__init__` gains an optional `spec: AgentSpec | None = None` parameter. When provided:
- `self.name = spec.name` (defaults to `"agent"` for backwards compatibility)
- `spec.role` fills the `{role_description}` prompt slot (see 1.2)
- `spec.tool_names` calls `resource_manager.select_tools_by_names()` after loading
- `spec.skill_names` calls `resource_manager.select_skills_by_names()` after loading
- `spec.llm` / `spec.source` / `spec.temperature` override the corresponding `default_config` values

**Files affected:** new `BaseAgent/agent_spec.py`, `base_agent.py` (`__init__`, `configure`), `prompts.py`

**Backwards compatibility:** When `spec=None`, behavior is identical to today.

#### 1.2 Parameterize the system prompt header

Replace the hardcoded "biomedical assistant" framing in `_SYSTEM_PROMPT_HEADER` with a `{role_description}` slot:

```python
# prompts.py — before:
"You are a helpful biomedical AI assistant..."

# After:
"You are {role_description}."
```

`_generate_system_prompt()` fills `role_description` from `self.spec.role` when a spec is set, or from a default string when not.

**Files affected:** `prompts.py`, `base_agent.py` (`_generate_system_prompt`)

#### 1.3 Isolate REPL execution contexts *(highest risk change in Phase 1)*

Each `BaseAgent` instance must own its own Python execution namespace. The current module-level globals must be eliminated for concurrent multi-agent use.

**Changes:**
1. `run_python_repl(command, namespace=None)` — when `namespace` is provided, `exec()` uses it; when `None`, falls back to module-level `_persistent_namespace` for backwards compatibility.
2. `BaseAgent.__init__` initializes `self._repl_namespace: dict = {}`.
3. `NodeExecutor.execute()` passes `agent._repl_namespace` to `run_python_repl()`.
4. `inject_custom_functions_to_repl(custom_functions, namespace=None)` — targets a specific namespace instead of the global.
5. Move plot capture into a `PlotCapture` class owned per-agent instance:

```python
# BaseAgent/tools/support_tools.py
class PlotCapture:
    """Per-agent matplotlib plot capture. Replaces module-level _captured_plots."""
    def __init__(self):
        self._plots: list[str] = []
        self._patched: bool = False

    def apply_patches(self) -> None: ...
    def capture(self) -> None: ...
    def get_plots(self) -> list[str]: ...
    def clear(self) -> None: ...
```

`BaseAgent` instantiates `self._plot_capture = PlotCapture()` and passes it into the execute node.

**Risk:** All code importing `_persistent_namespace` directly breaks. Grep before changing:
- `BaseAgent/tools/support_tools.py` — definition
- `BaseAgent/utils/tool_bridge.py` — reads the global directly
- Any test fixtures referencing the global

Keep the module-level `_persistent_namespace` as a fallback to avoid breaking callers that haven't migrated.

**Files affected:** `support_tools.py`, `tool_bridge.py`, `base_agent.py`, `nodes.py`

#### 1.4 Extract the subgraph (enables Option B later)

Split `configure()` into `get_subgraph()` and `configure()`. Existing behavior is preserved:

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

This is a low-risk refactor and unlocks Option B (embedded subgraphs) without committing to it yet.

**Files affected:** `base_agent.py`

---

### Phase 2: Orchestration layer (multi-agent begins)

**Goal:** Build a supervisor that coordinates specialized `BaseAgent` instances.

#### 2.1 Multi-agent state schema

Keep it minimal. LangGraph's state is the source of truth; don't duplicate what the framework already manages.

```python
# New file: BaseAgent/multi_agent/state.py
from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]  # Shared conversation
    sender: str           # Name of the agent that last produced output
    next_agent: str | None  # Routing decision; None = done
    task: str             # Original user task
    results: dict[str, str]  # agent_name -> final output string
```

No `task_queue` — LangGraph's `Send()` handles fan-out. No `agent_states` dict — each sub-agent manages its own internal state. No custom `AgentMessage` class — use LangChain's existing message types with the `name` field for sender attribution.

**Files affected:** new `BaseAgent/multi_agent/__init__.py`, new `BaseAgent/multi_agent/state.py`

#### 2.2 `MultiAgentOrchestrator`

A supervisor pattern: an LLM decides which agent to route to next, or declares the task complete.

```python
# New file: BaseAgent/multi_agent/orchestrator.py
class MultiAgentOrchestrator:
    def __init__(
        self,
        agents: list[BaseAgent],
        supervisor_llm: str | None = None,
        supervisor_source: SourceType | None = None,
        max_rounds: int = 10,
        checkpoint_db_path: str = ":memory:",
    ):
        if not agents:
            raise ValueError("At least one agent is required")
        if len({a.name for a in agents}) != len(agents):
            raise ValueError("Agent names must be unique")
        self.agents = {a.name: a for a in agents}
        self.max_rounds = max_rounds
        self._supervisor_llm = self._init_supervisor_llm(supervisor_llm, supervisor_source)
        self._build_graph(checkpoint_db_path)

    def run(self, task: str, thread_id: str | None = None) -> tuple[list, str]:
        """Execute the multi-agent workflow synchronously."""
        ...
```

**Supervisor node logic:**
1. Receives `MultiAgentState` (messages + results so far)
2. Calls supervisor LLM with a prompt listing available agents, their roles, and results so far
3. Uses `with_structured_output` to extract `{"next_agent": "<name_or_FINISH>"}` reliably
4. Sets `state["next_agent"]`; `FINISH` maps to `None`

**Agent node logic (one node per agent):**
1. Extracts the latest task from `state["messages"]`
2. Calls `agent.run(sub_task)` — the full single-agent loop runs to completion
3. Writes the result to `state["results"][agent.name]`
4. Appends the result as an `AIMessage(name=agent.name, content=result)`
5. Returns updated state

**Routing:**
```python
def _route(self, state: MultiAgentState) -> str:
    next_agent = state.get("next_agent")
    if next_agent is None or next_agent == "FINISH":
        return END
    if next_agent not in self.agents:
        return END  # Supervisor hallucinated an agent name; terminate safely
    return next_agent
```

**Graph topology:**
```
START → supervisor → [agent_a | agent_b | agent_c] → supervisor → ... → END
```

**Files affected:** new `BaseAgent/multi_agent/orchestrator.py`

#### 2.3 Routing strategies

Start with supervisor only. Add others as validated needs arise:

| Pattern | Phase | Implementation |
|---------|-------|----------------|
| **Supervisor** | 2 | LLM picks next agent via structured output |
| **Sequential** | 2 | Fixed pipeline: hard-coded edge chain in a `SequentialOrchestrator` |
| **Parallel** | 3 | `Send()` to multiple agents; merge via aggregation node |
| **Hierarchical** | Deferred | Sub-supervisors; add only when validated need exists |

---

### Phase 3: Production hardening

**Goal:** Make multi-agent reliable, observable, and cost-aware.

#### 3.1 Error propagation

Define a structured result type so failures surface to the supervisor:

```python
# BaseAgent/multi_agent/types.py
from pydantic import BaseModel

class AgentResult(BaseModel):
    agent_name: str
    status: Literal["success", "error", "timeout"]
    output: str
    error: str | None = None
    usage: list = []  # list[UsageMetrics]
```

The supervisor receives `AgentResult` instead of raw strings and can retry, reroute, or abort based on `status`.

#### 3.2 Context window management *(also needed for single-agent)*

Implement before multi-agent amplifies the problem:

- **Per-agent:** Truncate `state["input"]` when it exceeds a configurable token budget (configurable via `BaseAgentConfig.max_context_tokens`). Strategy: keep system message + first user message + last N messages.
- **Supervisor:** Pass only each agent's final result string to the supervisor — not the agent's full conversation history. Full histories live in per-agent checkpointers.

This is roadmap item 8 and should be implemented here, not after.

#### 3.3 Cost tracking

Extend `UsageMetrics` aggregation to the orchestrator level:

```python
orchestrator.total_usage    # Aggregated UsageMetrics across all agents + supervisor
orchestrator.usage_by_agent  # dict[agent_name, list[UsageMetrics]]
```

The supervisor can use cost metadata to prefer cheaper agents for simpler subtasks (optional, phase 4+).

#### 3.4 Testing strategy

Multi-agent tests require different fixtures than single-agent:

- **Unit tests:** Mock each sub-agent's `run()` return value. Test supervisor routing logic in isolation.
- **Integration tests:** Use mock LLMs (existing `conftest.py` pattern) for each agent and the supervisor. Verify full round-trips.
- **Deterministic routing tests:** Fix supervisor LLM output via `MagicMock` to exercise each routing branch.

```python
# BaseAgent/tests/test_multi_agent.py
def test_supervisor_routes_to_researcher(mock_supervisor_llm):
    mock_supervisor_llm.invoke.return_value = MagicMock(content='{"next_agent": "researcher"}')
    orchestrator = MultiAgentOrchestrator([researcher, coder], ...)
    log, result = orchestrator.run("Summarize recent LLM papers")
    assert "researcher" in orchestrator.usage_by_agent

def test_agent_failure_terminates_gracefully(mock_failing_agent):
    mock_failing_agent.run.side_effect = RuntimeError("LLM timeout")
    orchestrator = MultiAgentOrchestrator([mock_failing_agent], ...)
    log, result = orchestrator.run("Any task")
    # Should not raise; should return an error result
    assert result  # some output, not a crash

def test_repl_namespace_isolated():
    """Two agents running concurrently should not share REPL state."""
    agent_a = BaseAgent(spec=AgentSpec(name="a", role="Agent A"))
    agent_b = BaseAgent(spec=AgentSpec(name="b", role="Agent B"))
    assert agent_a._repl_namespace is not agent_b._repl_namespace
```

**Files affected:** new `BaseAgent/tests/test_multi_agent.py`, new `BaseAgent/tests/test_repl_isolation.py`

---

### Phase 4: Shared memory and advanced patterns (deferred)

**Goal:** Enable agents to build on each other's work across turns.

#### 4.1 Use LangGraph `Store` for persistent cross-agent memory

Don't build a custom `SharedMemory` class — `SharedMemory` in the prior plan reinvents a primitive LangGraph already provides. Use `InMemoryStore` (or `SqliteStore` for persistence):

```python
from langgraph.store.memory import InMemoryStore

store = InMemoryStore()
# In agent node: write findings
store.put(namespace=("findings", agent_name), key="result", value=output)
# In supervisor: read across agents
findings = store.search(namespace=("findings",))
```

#### 4.2 Result aggregation

A dedicated aggregation node that combines multi-agent outputs:

```python
def aggregate(state: MultiAgentState) -> MultiAgentState:
    """Synthesize results from all agents into a coherent final answer."""
    results = state["results"]
    # LLM-based synthesis, or template-based merging — configurable per-orchestrator
    ...
```

#### 4.3 Parallel execution with `Send()`

For independent sub-tasks, use LangGraph's `Send()` for fan-out:

```python
def decompose_and_fanout(state: MultiAgentState):
    sub_tasks = decompose_task(state["task"])  # LLM or heuristic decomposition
    return [Send(agent_name, {"task": sub_task}) for agent_name, sub_task in sub_tasks]
```

This requires agents to accept isolated state dicts — natural if Phase 1.3 (REPL isolation) is complete.

---

## Implementation Priority & Sequencing

```
Phase 1.1  AgentSpec (agent identity)            ████████████████  HIGH    ~1 day
Phase 1.2  Parameterize system prompt            ████████████      HIGH    ~0.5 day
Phase 1.3  Isolate REPL namespaces               ████████████████  HIGH    ~2 days  ← highest risk
Phase 1.4  Extract subgraph (get_subgraph)       ████████          MEDIUM  ~1 day
Phase 3.2  Context window management             ████████████      MEDIUM  ~2 days  ← unblock before Phase 2
Phase 2.1  MultiAgentState schema                █████████████     HIGH    ~0.5 day
Phase 2.2  MultiAgentOrchestrator                ████████████████  HIGH    ~2-3 days
Phase 2.3  Sequential routing                    ████████          MEDIUM  ~1 day
Phase 3.1  Error propagation (AgentResult)       ████████          MEDIUM  ~1 day
Phase 3.3  Cost tracking                         ██████            LOW     ~0.5 day
Phase 3.4  Tests                                 ████████████████  HIGH    ongoing
Phase 4.*  Store, aggregation, parallel fanout   ████████          LOW     deferred
```

**Critical path (minimum viable multi-agent):**
`1.1 → 1.2 → 1.3 → [3.2] → 2.1 → 2.2`

Context window management (3.2) should come before or alongside Phase 2, not after — multi-agent makes unbounded history a blocker, not just a limitation.

---

## Backwards Compatibility Constraints

All changes must preserve the existing single-agent API without modification:

```python
# Must continue to work unchanged, with identical behavior:
agent = BaseAgent(llm="claude-sonnet-4-20250514")
log, answer = agent.run("What is 2+2?")

agent2 = BaseAgent(require_approval="dangerous_only")
log, payload = agent2.run("Write a bash script")
if agent2.is_interrupted:
    log, answer = agent2.resume()
```

Rules:
- `AgentSpec` is optional. When omitted, all defaults are identical to today.
- `run_python_repl()` without `namespace` falls back to the module-level global (no breakage).
- `MultiAgentOrchestrator` lives in a new subpackage (`BaseAgent/multi_agent/`). No changes to existing top-level imports.
- `BaseAgent` is never subclassed by the multi-agent system; it is composed.

---

## Proposed File Structure

```
BaseAgent/
  multi_agent/                    # New subpackage (Phase 2+)
    __init__.py                   # Exports: MultiAgentOrchestrator, AgentSpec
    agent_spec.py                 # AgentSpec dataclass (Phase 1.1)
    state.py                      # MultiAgentState TypedDict (Phase 2.1)
    orchestrator.py               # MultiAgentOrchestrator + SequentialOrchestrator (Phase 2.2)
    types.py                      # AgentResult (Phase 3.1)

  base_agent.py                   # Modified: spec param, get_subgraph(), self._repl_namespace
  prompts.py                      # Modified: {role_description} slot in _SYSTEM_PROMPT_HEADER
  config.py                       # Modified: add max_context_tokens field (Phase 3.2)

  tools/
    support_tools.py              # Modified: PlotCapture class; namespace param on run_python_repl
  utils/
    tool_bridge.py                # Modified: namespace param on inject_custom_functions_to_repl

  tests/
    test_multi_agent.py           # New: orchestrator routing and round-trip tests
    test_repl_isolation.py        # New: verify namespace isolation between instances
```

---

## Overall Readiness

The project is approximately **25-30% ready** (revised down from 40%).

**What's solid:**

- LangGraph `StateGraph` ✅
- Multi-provider LLM factory ✅
- `NodeExecutor` (testable nodes) ✅ — partially addressed from prior plan
- Resource management with `selected` flag ✅
- Skills system (behavioral parameterization) ✅
- Checkpointing + interrupt/resume ✅

**What's not started:**

- Agent identity / role / persona system — 0%
- REPL namespace isolation — 0%
- System prompt parameterization — 0%
- Orchestration layer — 0%
- Context window management — 0%
- Multi-agent state / error propagation — 0%

The prior 40% estimate over-weighted the foundations without accounting for the depth of the REPL isolation work, the lack of any prompt parameterization, and the complete absence of orchestration code. The foundations are genuinely solid, but the multi-agent-specific work hasn't begun.
