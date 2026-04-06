# BaseAgent Feature Implementation Roadmap (Revised)

Goal: Evolve BaseAgent toward a multi-agent system with a frontend, where agents surface their reasoning and support human interrupt/resume.

Based on "Designing Multi-Agent Systems" by Victor Dibia, and revised after a thorough codebase audit against the original 12-item roadmap.

---

## Critique of the Original Roadmap

The original roadmap proposed 12 features across 36 implementation phases. After auditing the codebase, it has fundamental problems:

### 1. Rebuilds LangGraph Features

Items 1 (Async+Streaming), 5 (Human-in-the-Loop), 10 (Agents as Tools), 11 (Multi-Agent Orchestration), and 12 (Workflow Patterns) propose building capabilities that LangGraph already ships natively:

- **Async**: LangGraph compiled graphs support `ainvoke()`, `astream()`, `astream_events()` out of the box. BaseAgent already uses `app.stream()` at `base_agent.py:961`.
- **HITL**: LangGraph provides `interrupt_before`/`interrupt_after` on nodes. The original roadmap proposed building `ApprovalRequest`, `ApprovalHandler`, and a custom `human_loop/` package.
- **Workflows**: `StateGraph` with conditional edges (already used at `base_agent.py:850-870`) is a DAG execution engine. Building `BaseStep`, `Edge`, `WorkflowRunner` on top is redundant.
- **Agent composition**: LangGraph supports subgraph composition natively.

### 2. Middleware Conflicts with LangGraph Architecture

LangGraph uses callbacks and listeners for cross-cutting concerns, not middleware pipelines. The proposed `BaseMiddleware` with `on_request`/`on_response`/`on_tool_call` creates a competing execution model. Cost tracking already exists via `UsageMetrics` (`llm.py:30`), recorded at `base_agent.py:648` and `base_agent.py:823`.

### 3. Structured Output Misunderstands the Execution Model

BaseAgent executes code via REPL and returns computed results, not raw LLM text. The `<execute>`/`<solution>` XML tag parsing (`base_agent.py:664-666`) routes execution. Applying `with_structured_output()` would change the agent from a code-executing agent into a text-generating agent, contradicting its core design.

### 4. Items 6 and 7 Overlap ~80%

Long-term Memory (RAG) and Agent-Managed Memory share storage infrastructure, retrieval interface, and integration points. They should be one feature with two phases.

### 5. ~~Current Streaming is Untyped~~ (RESOLVED)

`events.py` defines `AgentEvent`/`EventType`. `run_stream()` uses `astream_events()` and maps nodes to typed events. `go()` is preserved for backward compatibility.

### ~~6. MemorySaver is In-Memory Only~~ (RESOLVED)

`SqliteSaver` now replaces `MemorySaver`. State persists to a SQLite file across process restarts. Thread IDs are auto-generated UUIDs per `run()` call; callers can supply one for cross-session resume. The default `checkpoint_db_path=":memory:"` preserves the old ephemeral behavior.

### 7. Missing Critical Needs

- No code execution sandboxing (`exec()` in `support_tools.py`)
- Minimal error handling in execute node (`nodes.py:execute()`)
- No API documentation

### 8. Wrong Sequencing for the Actual Goal

Async/streaming was listed as the first priority in isolation. The actual goal (frontend + HITL + multi-agent) requires streaming events, persistent checkpointing, and interrupt/resume as a tightly-coupled critical path, not independent features.

---

## HIGH PRIORITY -- Frontend & HITL Critical Path

### ✅ 1. Persistent Checkpointing + Interrupt/Resume

**Why**: Human interrupt/resume requires persistent state. `MemorySaver` is in-memory only.

**Status**: **DONE**. Implemented on branch `feat-persistent-checkpointing-interrupt-resume`.

**What was built**:

```
Phase 1: Persistent Checkpointing ✅
  - SqliteSaver replaces MemorySaver; falls back gracefully if package absent
  - checkpoint_db_path added to BaseAgentConfig (default: ":memory:" preserves old behavior)
  - BASE_AGENT_CHECKPOINT_DB_PATH env var override
  - UUID thread_id auto-generated per run(); caller can supply one for resume
  - close() / __del__() release the SQLite connection

Phase 2: Interrupt/Resume ✅
  - approval_gate node calls interrupt() (modern LangGraph API, not interrupt_before)
  - interrupt() surfaces {code, language, message} payload to caller
  - agent.resume() sends Command(resume=True) to approve and continue
  - agent.reject(feedback) sends Command(resume={approved: False, feedback}) to regenerate
  - is_interrupted property; run()/resume()/reject() return (log, payload) uniformly
  - run_stream() yields AgentEvent(APPROVAL_REQUIRED) on interrupt

Phase 3: Selective Interruption ✅
  - require_approval config: "always" | "never" (default) | "dangerous_only"
  - "dangerous_only": routing_function_with_approval sends bash/R through gate; Python passes direct
  - BASE_AGENT_REQUIRE_APPROVAL env var override
  - Graph topology selected at configure() time; no runtime overhead when "never"
```

**Files Modified**:
- `BaseAgent/base_agent.py`: `_create_checkpointer()`, thread_id, `run()`, `resume()`, `reject()`, `close()`, `is_interrupted`, graph topology in `configure()`
- `BaseAgent/config.py`: `checkpoint_db_path`, `require_approval` fields + env overrides
- `BaseAgent/nodes.py`: `approval_gate()`, `routing_function_with_approval()`, pending fields in `generate()`/`execute()`
- `BaseAgent/state.py`: `pending_code`, `pending_language` fields
- `BaseAgent/events.py`: `APPROVAL_REQUIRED` event type
- `pyproject.toml`: `langgraph-checkpoint-sqlite>=2.0.0` dependency
- `BaseAgent/tests/test_checkpointing.py` (new): 22 tests
- `BaseAgent/tests/test_interrupt_resume.py` (new): 24 tests

---

### 2. Async-First API

**Why**: Frontend cannot block on synchronous `go()`. Non-blocking execution is required for web integration.

**Current State**: `go()` is synchronous, uses `app.stream()` at `base_agent.py:738`.

**Implementation Plan**:

```
Phase 1: Core Async
  - Convert go() to async def go() using app.astream()
  - Add go_sync() convenience wrapper via asyncio.run()
  - Update configure() to support async node functions in nodes.py

Phase 2: Backward Compatibility
  - go_sync() matches current go() signature and return type
  - Existing notebooks/scripts use go_sync() with no changes
  - New frontend code uses async go() or run_stream()
```

**Files to Modify**:
- `BaseAgent/base_agent.py`: Convert `go()`, add `go_sync()`

**Estimated Effort**: 1-2 weeks

---

### 3. Error Handling + Termination Conditions

**Why**: Users watching a frontend need structured error reporting. Runaway agents need budget and iteration limits.

**Current State**: Execute node (`nodes.py:execute()`) has no try/except around `run_with_timeout`. Parse errors use a counter that terminates after 2 failures (in `nodes.py:generate()`). `recursion_limit` hardcoded to 500 (`base_agent.py:732`). `BaseAgentConfig` has no termination fields.

**Implementation Plan**:

```
Phase 1: Structured Errors
  - Define error types in new errors.py: ExecutionError, ParseError,
    TimeoutError, LLMError, BudgetExceededError
  - Wrap execute node in try/except with structured error context
  - Errors emitted as AgentEvent(type=ERROR) for frontend display

Phase 2: Termination Conditions
  - Add to BaseAgentConfig: max_iterations (replace recursion_limit=500),
    max_cost, max_consecutive_errors
  - Check conditions in routing_function before each generate cycle
  - Cost tracking uses existing UsageMetrics (llm.py, nodes.py:generate())
  - Composable: terminate when ANY condition is met

Phase 3: Recovery
  - On repeated execution failures, inject error context for LLM self-correction
  - On LLM rate limit, exponential backoff
  - On context overflow, trigger context window management (item 7)
```

**Files to Modify/Create**:
- `BaseAgent/errors.py` (new): Structured error types
- `BaseAgent/nodes.py`: Error handling in execute and generate nodes
- `BaseAgent/config.py`: Termination configuration fields

**Estimated Effort**: 2 weeks

---

## MEDIUM PRIORITY -- Multi-Agent Enablement

### 4. Agent Composition via LangGraph Subgraphs

**Why**: Multi-agent systems need agents that can delegate to other agents.

**Current State**: Single agent only. No composition mechanism.

**Implementation Plan**:

```
Phase 1: Subgraph Wrapper
  - Add as_subgraph() method to BaseAgent
  - Returns a compiled StateGraph that can be added as a node in a parent graph
  - Shared state protocol for inter-agent communication

Phase 2: Event Propagation
  - Child agent events bubble up to parent's event stream
  - Parent frontend can render nested agent reasoning

Phase 3: Orchestration Patterns
  - Sequential: agents run in order
  - Conditional: parent routes to specialist agents based on task
  - Use LangGraph native subgraph composition throughout
```

**Files to Modify**:
- `BaseAgent/base_agent.py`: Add `as_subgraph()` method

**Estimated Effort**: 2-3 weeks

---

### 5. Memory & Persistence (Merged Original Items 6+7)

**Why**: Cross-session learning and knowledge persistence enable agents to build on past work.

**Current State**: `MemorySaver` used for within-session LangGraph checkpointing only. No semantic memory.

**Implementation Plan**:

```
Phase 1: Cross-Session Persistence (mostly done by item 1)
  - SqliteSaver provides session persistence and resume
  - Previous sessions resumable by passing same thread_id

Phase 2: Semantic Memory Store
  - Simple MemoryStore interface: store(text, metadata), search(query, k)
  - SQLite FTS5 backend (no vector DB dependency for PoC)
  - Register memory tools via add_tool(): write_memory, search_memory
  - Inject relevant memories into system prompt during _generate_system_prompt()

Phase 3: Automatic Memory
  - Auto-extract key results from successful agent runs
  - Memory importance scoring and decay
  - Cross-session pattern recognition
```

**Files to Modify/Create**:
- `BaseAgent/memory.py` (new): Memory store interface and SQLite FTS5 implementation
- `BaseAgent/base_agent.py`: Memory injection into prompt

**Estimated Effort**: 3-4 weeks

---

### 6. Context Window Management

**Why**: Long multi-agent conversations will exceed context limits. Message history currently grows unbounded.

**Current State**: All messages kept in `state["input"]` list. Output truncated to 10K chars in `nodes.py:execute()`, but message count is unlimited.

**Implementation Plan**:

```
Phase 1: Basic Strategies
  - Sliding window: keep last N messages
  - Preserve system prompt and first user message, trim middle
  - Token budget tracking before LLM calls in generate node
  - Configurable strategy via BaseAgentConfig

Phase 2: Smart Truncation
  - Summarize old execution results instead of keeping full text
  - Keep error messages (they inform about what not to do)
  - Trigger when approaching model's context limit
```

**Files to Modify**:
- `BaseAgent/nodes.py`: Context management before LLM invoke in generate node
- `BaseAgent/config.py`: `max_context_messages`, `context_strategy` fields

**Estimated Effort**: 1-2 weeks

---

## ✅ Agent Skills System (DONE)

**Status**: **DONE**. Implemented as a new `Skill` resource type following the same patterns as `CustomTool`, `CustomData`, and `CustomSoftware`.

**What was built**:

```
Phase 1: Skill Model + ResourceManager ✅
  - Skill Pydantic model (name, description, trigger, tools, instructions, source_path, selected)
  - skills: list[Skill] added to ResourceCollection
  - ResourceManager skill CRUD: add_skill, get_all_skills, get_selected_skills,
    get_skill_by_name, remove_skill_by_name, select_skills_by_names
  - select_all_resources / deselect_all_resources updated for skills
  - get_summary includes skill counts

Phase 2: BaseAgent API + Prompt Integration ✅
  - BaseAgent._parse_skill_file(path): parses YAML frontmatter + markdown body from SKILL.md
  - BaseAgent.add_skill(skill_or_path): registers a Skill or loads from file, regenerates prompt
  - BaseAgent.load_skills(directory): loads all SKILL.md and *.skill.md files from a directory
  - _generate_system_prompt() injects selected skills into prompt (name, description, instructions)
  - skills_directory config field + BASE_AGENT_SKILLS_DIRECTORY env var for auto-load at startup
  - Skill exported from BaseAgent package: from BaseAgent import Skill

Phase 3: Retriever Integration ✅
  - ToolRetriever.prompt_based_retrieval accepts all_skills, returns selected_skills
  - _RESOURCE_SELECTION_PROMPT extended with AVAILABLE SKILLS + SKILLS: [indices] response format
  - _select_resources_for_prompt passes auto-trigger skills to retriever and calls
    select_skills_by_names with results
  - Skills with trigger="manual" are never deselected by the retriever
```

**Files Modified**:
- `BaseAgent/resources.py`: `Skill` model + `skills` field on `ResourceCollection`
- `BaseAgent/resource_manager.py`: Skill CRUD + selection methods + `select/deselect_all_resources` + `get_summary`
- `BaseAgent/base_agent.py`: `_parse_skill_file()`, `add_skill()`, `load_skills()`, prompt assembly, retriever integration, `configure()` auto-load, `skills_directory` param
- `BaseAgent/prompts.py`: `_SKILLS_SECTION`, `_SKILL_ENTRY_TEMPLATE`, extended `_RESOURCE_SELECTION_PROMPT`
- `BaseAgent/config.py`: `skills_directory` field + `BASE_AGENT_SKILLS_DIRECTORY` env var
- `BaseAgent/__init__.py`: Export `Skill`
- `BaseAgent/retriever.py`: `SKILLS:` index parsing + `selected_skills` return key
- `BaseAgent/tests/test_skills.py` (new): 42 tests (model, CRUD, file parsing, prompt injection, add_skill, load_skills, retriever)
- `BaseAgent/tests/test_retriever.py`: Updated output key assertion

**Future Work for Skills**:
- **Dynamic skill activation**: LLM can request a skill mid-run via a new graph node or routing logic. Would require `active_skills: list[str]` in `AgentState` and a detection step in `generate()`.
- **Progressive disclosure**: Show only skill name+description when context is limited; inject full instructions only when activated. Aligns with Roadmap Item 6 (Context Window Management).
- **Subagent execution**: A skill with `trigger="subagent"` runs as a child BaseAgent subgraph (Roadmap Item 4).
- **Supporting files**: Skills can bundle scripts, templates, example data referenced in their instructions.
- **Remote skill sources**: Load skills from URLs or registries (similar to Claude's `/v1/skills` API).
- **Skill versioning**: A `skills-lock.json` mechanism for pinning skill versions (mirrors the `.claude/skills/` pattern).

---

## SECONDARY -- Important but Not on Critical Path

### 7. Observability & Cost Tracking

Surface existing `UsageMetrics` to frontend via AgentEvent. Add `usage_summary()` method to BaseAgent. Optional LangSmith integration via LangChain callbacks.

### 8. Native Tool Calling Migration

Add `tool_calling_mode` config: `"xml"` (current default), `"native"` (uses `bind_tools()`). Hybrid mode allows gradual migration. Native mode eliminates XML tag-repair code in `nodes.py:generate()` and stop sequences.

### 9. Code Execution Security

Add configurable import blocklists for dangerous modules. Filesystem access restrictions. Optional Docker-based isolation backend. Keep `exec()` mode as default for development.

### 10. Test Coverage Expansion

Priority targets: `resource_manager.py` (714 lines, needs deeper coverage), `llm.py` provider paths, execution loop integration tests with mocked LLM responses.

### 11. Documentation

API reference for `BaseAgent`, `add_tool()`, `add_mcp()`, `go()`, `run_stream()`. Architecture guide. Tool development guide. LLM provider configuration guide.

---

## Implementation Order

```
Step   Item                                Depends On         Effort    Parallel?
✅ --  Streaming Event Architecture        --                 DONE      --
✅ 1   Persistent Checkpoint + HITL        --                 DONE      --
✅ --  Agent Skills System                 --                 DONE      --
 2     Async-First API                     streaming (done)   1-2 wk    Yes (with 3)
 3     Error Handling + Termination        streaming (done)   2 wk      Yes (with 2)
 4     Agent Composition                   2                  2-3 wk    --
 5     Memory & Persistence                1 (done)           3-4 wk    --
 6     Context Window Management           --                 1-2 wk    Yes (anytime)
```

Items 2 and 3 can start in parallel now. Streaming Event Architecture, Persistent Checkpointing + HITL, and Agent Skills are complete. Item 5 (Memory) is unblocked since item 1 is done. Item 6 can proceed independently at any time. Skills future work (dynamic activation, progressive disclosure) is unblocked by item 6.

---

## Original Roadmap Cross-Reference

| # | Original Item | Disposition |
|---|---|---|
| 1 | Async + Streaming | **DONE**. `events.py` (AgentEvent/EventType) + `run_stream()` using `astream_events()`. Remaining async work is in roadmap item 2. |
| 2 | Middleware System | **Dropped**. Use LangGraph callbacks/listeners instead |
| 3 | Structured Output | **Deferred**. Only for final answer extraction, not entire execution loop |
| 4 | Termination Conditions | Merged into item 4 (Error Handling + Termination) |
| 5 | Human-in-the-Loop | **DONE**. Implemented as roadmap item 1. Uses LangGraph `interrupt()` + `Command(resume=...)` (modern API), not `interrupt_before`. `approval_gate` node, `require_approval` config, `resume()`/`reject()` methods. |
| 6 | Long-term Memory (RAG) | Merged with item 7 into item 6 (Memory & Persistence) |
| 7 | Agent-Managed Memory | Merged with item 6 |
| 8 | Context Engineering | Revised as item 7 (Context Window Management) |
| 9 | OpenTelemetry | Folded into item 8 (Observability). Optional, via LangChain callbacks |
| 10 | Agents as Tools | Revised as item 5 (Agent Composition). Uses LangGraph subgraphs |
| 11 | Multi-Agent Orchestration | Folded into item 5. No custom orchestrators needed |
| 12 | Workflow Patterns | **Dropped**. LangGraph `StateGraph` already is a workflow engine |

---

## References

- Book: "Designing Multi-Agent Systems" by Victor Dibia (2025)
- LangGraph Documentation: Checkpointing, Human-in-the-Loop, Streaming
- Original roadmap: Based on book chapters, revised against actual codebase state
