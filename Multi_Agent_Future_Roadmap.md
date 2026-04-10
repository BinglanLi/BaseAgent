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

Features 1-4 are implemented and tested. See `.claude/baseagent_modules.md` for current API details.

| Feature | Summary | Tests |
|---------|---------|-------|
| **Feature 1: MCP Overhaul** | Remote Streamable HTTP transport, auth headers with `${ENV_VAR}` interpolation, async/sync bridge fix | 13 unit tests |
| **Feature 2: AgentSpec** | `AgentSpec` dataclass for agent identity; `{role_description}` parameterization in system prompt | 22 unit tests |
| **Feature 3: REPL Namespace Isolation** | Per-instance `_repl_namespace` and `PlotCapture`; `namespace` param on `run_python_repl` and `inject_custom_functions_to_repl` | unit tests |
| **Feature 4: Extract Subgraph** | `get_subgraph()` returns uncompiled `StateGraph` for LangGraph composition; `configure()` calls it then compiles | 18 unit tests |

---

## Prototype Feature Specifications

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

### Feature 10: Skills System Overhaul (Spec-Driven Loading + Progressive Disclosure)

**Priority:** HIGH -- the current implementation defeats its own purpose and will not scale for multi-agent workflows.

**Depends on:** None (can be built in parallel with other features).

**Current state and problems:**

The skills system exists (Feature "Agent Skills" in Existing Features) but has several design flaws that contradict the intended purpose of skills and will become blockers as multi-agent workflows add more domain-specific skills:

1. **Full skill body always injected into context** (`base_agent.py:774-784`). Every selected skill has its entire `instructions` field dumped into the system prompt via `_SKILL_ENTRY_TEMPLATE`. There is no progressive disclosure -- if 5 skills are selected, all 5 bodies go in at once. The YAML frontmatter (name, description) was designed for lightweight selection, but selection immediately triggers full injection. This directly contradicts the reference pattern from [anthropics/skills](https://github.com/anthropics/skills/blob/main/skills/skill-creator/SKILL.md), which specifies three loading levels:
   - Level 1: **Metadata** (name + description) -- always in context (~100 words)
   - Level 2: **SKILL.md body** -- loaded when skill triggers (<500 lines ideal)
   - Level 3: **Bundled resources** -- loaded as needed (unlimited, scripts execute without loading)

2. **No bundled resources support**. `_parse_skill_file` (`base_agent.py:264-308`) only reads the single SKILL.md file. The reference pattern describes `scripts/`, `references/`, `assets/` directories alongside SKILL.md. A skill like `ontology-mapping` may need reference docs (OWL specs, mapping examples) and template scripts (Cypher export patterns) that should load on demand -- not bloat the system prompt or be absent entirely.

3. **No context budget awareness**. Skills are injected unconditionally into the system prompt. In multi-agent workflows where each specialist agent may need 2-3 domain skills, unbounded injection will collide with Feature 5 (Context Window Management).

4. **Retriever makes all-or-nothing decisions**. At `base_agent.py:1034-1038`, the retriever selects skills based on `{name, description}` -- appropriate for lightweight triage. But selection immediately injects the full body. There is no intermediate step where the agent could inspect a skill summary before committing to loading the full instructions.

5. **`tools` field is purely decorative** (`resources.py:347`). `Skill.tools` is shown in the prompt but never used functionally -- it doesn't filter available tools, validate that referenced tools exist, or influence retriever decisions.

6. **No skill directory tracking**. `source_path` stores the SKILL.md file path, but nothing records the parent directory. Even if bundled resources existed, the system wouldn't know where to find `references/` or `scripts/` relative to the skill.

7. **Loading is wasteful for multi-agent**. `configure()` loads ALL skills from `skills_directory` via glob, then `spec.skill_names` filters via `select_skills_by_names()`. In BaseAgent multi-agent workflows, each agent needs 1-3 skills from a shared pool. Loading all skills just to discard most of them is unnecessary overhead and risks cross-agent state confusion.

**Skill directory convention:**

Skills reside in a conventional directory structure under the project root:

```
skills/                              # skills_directory points here
  ontology-design/
    SKILL.md                         # required: YAML frontmatter + markdown body
    references/                      # optional: bundled reference docs
      owl_spec.md
    scripts/                         # optional: bundled template scripts
      template.py
  database-evaluation/
    SKILL.md
  parser-development/
    SKILL.md
    scripts/
      csv_template.py
  ontology-mapping/
    SKILL.md
    references/
      owl_mapping_guide.md
  memgraph-export/
    SKILL.md
    scripts/
      cypher_template.py
```

`AgentSpec.skill_names=["ontology-design"]` resolves to `{skills_directory}/ontology-design/SKILL.md`. Each agent loads **only** the skills specified in its spec -- no glob, no load-all-then-filter.

**Phase 1 -- Model cleanup: remove `trigger`, add `source_dir`**

Remove `trigger` from the `Skill` model and all code that branches on it:

- `resources.py:344` -- remove `trigger` field from `Skill`
- `resource_manager.py:526-532` -- remove `trigger="manual"` guard; all skills follow the same selection logic
- `base_agent.py:1033-1037` -- retriever passes all skills as candidates (no trigger filter)
- `agent_spec.py:42` -- update docstring
- `tests/test_skills.py` -- remove `test_manual_trigger`, update `test_select_skills_by_names_manual_not_deselected`

Add `source_dir: str | None` to the `Skill` model. When loading from a file, set `source_dir` to the SKILL.md's parent directory. Add a `has_bundled_resources` computed property that checks for `references/`, `scripts/`, or `assets/` subdirectories.

```python
# In resources.py, add to Skill:
source_dir: Optional[str] = Field(None, description="Directory containing the SKILL.md and optional bundled resources")

@property
def has_bundled_resources(self) -> bool:
    if not self.source_dir:
        return False
    d = Path(self.source_dir)
    return any((d / sub).is_dir() for sub in ("references", "scripts", "assets"))

@property
def bundled_resource_manifest(self) -> dict[str, list[str]]:
    """List files in each bundled resource subdirectory."""
    if not self.source_dir:
        return {}
    manifest = {}
    for sub in ("references", "scripts", "assets"):
        sub_dir = Path(self.source_dir) / sub
        if sub_dir.is_dir():
            manifest[sub] = [f.name for f in sorted(sub_dir.iterdir()) if f.is_file()]
    return manifest
```

In `_parse_skill_file`, set `source_dir=str(path.parent)`.

**Files to modify (Phase 1):**
- `BaseAgent/resources.py` -- remove `trigger`, add `source_dir` + computed properties
- `BaseAgent/resource_manager.py` -- remove trigger guard in `select_skills_by_names`
- `BaseAgent/base_agent.py` -- `_parse_skill_file` (set `source_dir`), `_select_resources_for_prompt` (remove trigger filter)
- `BaseAgent/agent_spec.py` -- update docstring
- `BaseAgent/tests/test_skills.py` -- remove trigger tests, add `source_dir` / bundled resource tests

**Phase 2 -- Spec-driven targeted skill loading**

Replace the current "load all → filter" loading model with targeted loading. Each agent loads only the skills specified in its `AgentSpec.skill_names` directly from their conventional directory paths.

Add `_resolve_skill_path(skill_name) -> Path | None` to `BaseAgent`:

```python
def _resolve_skill_path(self, skill_name: str) -> Path | None:
    """Resolve a skill name to its SKILL.md path using directory conventions.

    Looks up {skills_directory}/{skill_name}/SKILL.md.
    Returns None if the file does not exist or skills_directory is not set.
    """
    if not self.skills_directory:
        return None
    candidate = Path(self.skills_directory) / skill_name / "SKILL.md"
    return candidate if candidate.is_file() else None
```

Rewrite the skill loading block in `configure()` (`base_agent.py:867-876`):

```python
# Skill loading: targeted when spec provides skill_names, legacy glob otherwise
if self.spec is not None and self.spec.skill_names is not None:
    # Targeted: load only the skills named in the agent spec
    for skill_name in self.spec.skill_names:
        path = self._resolve_skill_path(skill_name)
        if path:
            skill = self._parse_skill_file(path)
            self.resource_manager.add_skill(skill)
        else:
            print(f"Warning: skill '{skill_name}' not found in '{self.skills_directory}'")
elif self.skills_directory:
    # Legacy fallback: load all skills from directory via glob
    self.load_skills(self.skills_directory)

# Apply AgentSpec tool filters (skill filtering no longer needed -- we only loaded what we need)
if self.spec is not None:
    if self.spec.tool_names is not None:
        self.resource_manager.select_tools_by_names(self.spec.tool_names)
```

The `select_skills_by_names()` call is removed from `configure()` -- since we only loaded the skills we need, there is nothing to filter. `load_skills()` and `add_skill()` remain as public API for ad-hoc single-agent usage.

**Files to modify (Phase 2):**
- `BaseAgent/base_agent.py` -- `_resolve_skill_path()` (new), `configure()` (rewrite skill loading block)

**Phase 3 -- Progressive disclosure: metadata-only prompt + retriever-driven loading**

This is the core behavioral change. The system prompt shows only skill metadata at configuration time; the `retrieve` node loads full bodies on demand when the task arrives.

**New config field:**
- `BaseAgentConfig.skill_retrieval: bool = True` (default `True` -- always retrieve skills on demand)
- Env var override: `BASE_AGENT_SKILL_RETRIEVAL`

When `skill_retrieval=True` (default):
- `configure()` / `_generate_system_prompt(is_retrieval=False)`: injects only the **skill catalog** -- name, description, and bundled resource manifest per skill (~2-3 lines each)
- `retrieve` node: always runs skill selection (even if `use_tool_retriever=False`), then regenerates system prompt with full instructions for selected skills
- The agent sees full instructions only for task-relevant skills

When `skill_retrieval=False`:
- Falls back to current behavior: all loaded skills' full instructions injected into the system prompt at configure time
- Provided as an escape hatch for simple single-skill setups

**New prompt templates** in `prompts.py`:

```python
_SKILL_CATALOG_SECTION = \
"""

AVAILABLE SKILLS
===============================
The following skills provide specialized domain knowledge and workflows.
Relevant skill instructions will be loaded based on the current task.
{skill_catalog}
"""

_SKILL_CATALOG_ENTRY = \
"""- {skill_name}: {skill_description}{bundled_note}"""

_SKILL_SELECTION_PROMPT = \
"""Select which skills are relevant for this task.

TASK: {query}

AVAILABLE SKILLS:
{skills}

Respond with: SKILLS: [list of indices]
If none are relevant: SKILLS: []
"""
```

**System prompt changes** in `_generate_system_prompt()` (`base_agent.py:774-784`):

```python
selected_skills = self.resource_manager.get_selected_skills()
if selected_skills:
    if self.skill_retrieval and not is_retrieval:
        # Catalog mode: metadata only (initial prompt)
        catalog_entries = []
        for skill in selected_skills:
            bundled = ""
            if skill.has_bundled_resources:
                bundled = " [has bundled resources]"
            catalog_entries.append(_SKILL_CATALOG_ENTRY.format(
                skill_name=skill.name,
                skill_description=skill.description,
                bundled_note=bundled,
            ))
        prompt_modifier += _SKILL_CATALOG_SECTION.format(
            skill_catalog="\n".join(catalog_entries)
        )
    else:
        # Full injection: either retrieval pass or opt-out
        skill_parts = []
        for skill in selected_skills:
            skill_parts.append(_SKILL_ENTRY_TEMPLATE.format(
                skill_name=skill.name,
                skill_description=skill.description,
                skill_instructions=skill.instructions,
            ))
        prompt_modifier += _SKILLS_SECTION.format(skills_content="\n".join(skill_parts))
```

**New method** `_select_skills_for_prompt(prompt)` in `BaseAgent`:

```python
def _select_skills_for_prompt(self, prompt: str) -> None:
    """Select relevant skills based on task prompt via lightweight LLM call."""
    all_skills = [
        {"name": s.name, "description": s.description}
        for s in self.resource_manager.get_all_skills()
    ]
    if not all_skills:
        return

    skills_text = "\n".join(
        f"{i}. {s['name']}: {s['description']}" for i, s in enumerate(all_skills)
    )
    selection_prompt = _SKILL_SELECTION_PROMPT.format(query=prompt, skills=skills_text)
    response = self.llm.invoke([HumanMessage(content=selection_prompt)])

    # Parse SKILLS: [indices] response
    match = re.search(r"SKILLS:\s*\[(.*?)\]", response.content, re.IGNORECASE)
    if match and match.group(1).strip():
        indices = [int(idx.strip()) for idx in match.group(1).split(",") if idx.strip()]
        selected_names = [all_skills[i]["name"] for i in indices if i < len(all_skills)]
    else:
        selected_names = []

    self.resource_manager.select_skills_by_names(selected_names)
```

**Retrieve node changes** (`nodes.py:310-332`):

```python
def retrieve(self, state: "AgentState") -> "AgentState":
    agent = self.agent
    prompt = state["input"][-1].content

    # Skill retrieval: always runs when skill_retrieval=True and skills exist
    if agent.skill_retrieval and agent.resource_manager.get_all_skills():
        agent._select_skills_for_prompt(prompt)
        agent.system_prompt = agent._generate_system_prompt(
            self_critic=agent.self_critic,
            is_retrieval=True,
        )

    # Tool/data/library retrieval: only when use_tool_retriever=True
    if agent.use_tool_retriever:
        agent._select_resources_for_prompt(prompt)
        agent.system_prompt = agent._generate_system_prompt(
            self_critic=agent.self_critic,
            is_retrieval=True,
        )

    return state
```

The retrieve node is no longer a no-op when `use_tool_retriever=False` -- it always processes skill retrieval when `skill_retrieval=True` and there are registered skills.

**Files to modify (Phase 3):**
- `BaseAgent/config.py` -- `skill_retrieval: bool = True` field + env var override
- `BaseAgent/prompts.py` -- `_SKILL_CATALOG_SECTION`, `_SKILL_CATALOG_ENTRY`, `_SKILL_SELECTION_PROMPT` (new templates)
- `BaseAgent/base_agent.py` -- `_generate_system_prompt()` (catalog vs. full injection), `_select_skills_for_prompt()` (new method), `configure()` (store `self.skill_retrieval`)
- `BaseAgent/nodes.py` -- `retrieve` node: skill retrieval always runs independently of `use_tool_retriever`
- `BaseAgent/tests/test_skills.py` -- progressive disclosure tests (catalog in initial prompt, full body after retrieval)

**Phase 4 -- Bundled resource access via REPL**

Register a built-in helper function `read_skill_resource(skill_name, path)` that reads files from a skill's bundled resource directory. This lets the agent load reference docs and templates on demand from within `<execute>` blocks:

```python
# Injected into REPL namespace when skills with bundled resources are present
def read_skill_resource(skill_name: str, path: str, _agent=None) -> str:
    """Read a bundled resource file from a skill directory.

    Args:
        skill_name: Name of the skill (e.g. "ontology-mapping")
        path: Relative path within the skill directory (e.g. "references/owl_spec.md")
    """
    if _agent is None:
        raise RuntimeError("read_skill_resource requires an agent context")
    skill = _agent.resource_manager.get_skill_by_name(skill_name)
    if not skill or not skill.source_dir:
        raise FileNotFoundError(f"Skill '{skill_name}' not found or has no source directory")
    full_path = Path(skill.source_dir) / path
    if not full_path.is_file():
        raise FileNotFoundError(f"Resource '{path}' not found in skill '{skill_name}'")
    return full_path.read_text(encoding="utf-8")
```

Injected into the per-instance REPL namespace via `functools.partial(_agent=self)` in `_inject_custom_functions_to_repl()` when any loaded skill has bundled resources.

The skill instructions would then say things like:
```markdown
When you need the OWL mapping reference, load it:
read_skill_resource("ontology-mapping", "references/owl_mapping_guide.md")
```

**Files to modify (Phase 4):**
- `BaseAgent/tools/support_tools.py` -- `read_skill_resource` helper function
- `BaseAgent/tools/tool_description/support_tools.py` -- tool description for `read_skill_resource`
- `BaseAgent/base_agent.py` -- inject into REPL namespace in `_inject_custom_functions_to_repl()`
- `BaseAgent/tests/test_skills.py` -- bundled resource access tests

**Phase 5 -- Functional `tools` field**

Make the `Skill.tools` field actionable:
- At `configure()` time, validate that all tools listed in `skill.tools` actually exist in `ResourceManager`. Emit a warning for missing tools.
- When a skill is selected by the retriever in `_select_skills_for_prompt()`, auto-select its referenced tools (even if the retriever didn't explicitly pick them). This ensures skills always have their required tools available.

**Files to modify (Phase 5):**
- `BaseAgent/base_agent.py` -- `configure()` (tool validation), `_select_skills_for_prompt()` (auto-select skill tools)
- `BaseAgent/tests/test_skills.py` -- tool validation and auto-selection tests

**Backwards compatibility:**
- `spec=None` + `skills_directory` set: legacy `load_skills()` glob behavior preserved
- `skill_retrieval=False`: restores current all-in body injection behavior
- `add_skill()` and `load_skills()` remain as public API for ad-hoc usage
- Default (`skill_retrieval=True`) changes the prompt behavior but is strictly better -- agents see the same skill instructions, just loaded on demand rather than always present

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
- `BaseAgent/errors.py` -- Structured error types (Feature 6)
- `BaseAgent/multi_agent/` -- New subpackage: `state.py`, `orchestrator.py`, `workflow.py`, `types.py` (Features 8-9)

---

## Implementation Order

```
                                                    Depends On       Effort
== GROUP A (parallel, no dependencies) =============================================
Feature 10  Skills system overhaul                  --               ~3 days
  Phase 1   Model cleanup (remove trigger, add source_dir)            ~0.5 day
  Phase 2   Spec-driven targeted skill loading                       ~0.5 day
  Phase 3   Progressive disclosure in prompt injection               ~1 day
  Phase 4   Bundled resource access via REPL                         ~0.5 day
  Phase 5   Functional tools field                                   ~0.5 day

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
  Phase 3   BaseAgent multi-agent demo script
```

**Critical path:** `[1, 2, 3, 4, 10 parallel] -> [5, 6 parallel] -> 7 -> 8 -> 9`

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
| Context window timing | **Before orchestration** (Feature 5 before 8) | Multi-agent amplifies unbounded history from a limitation into a blocker. |
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
