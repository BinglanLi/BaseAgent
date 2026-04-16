# BaseAgent Examples

This directory contains 12 example scripts for onboarding and reference.
Read them in order for a guided tour, or jump to the topic you need.

## Getting Started

Install BaseAgent and set your API key:

```bash
pip install -e .
```

```bash
# .env
ANTHROPIC_API_KEY=your_key_here
```

## Examples

### 1. Basic Usage (`01_basic_usage.py`)
Initialize an agent, run a task, and read usage metrics.

### 2. Custom Tools (`02_custom_tools.py`)
Add Python functions as agent tools. Name, description, and parameter schema
are auto-derived from the function's docstring and type hints.

### 3. MCP Servers (`03_mcp_servers.py`)
Connect external tool servers via the Model Context Protocol (local stdio and
remote Streamable HTTP transports).

### 4. Resources and Tool Retrieval (`04_resources_and_retrieval.py`)
Register datasets (`DataLakeItem`) and libraries (`Library`) for the agent to
reference. Enable `use_tool_retriever=True` to have the agent select tools by
semantic similarity instead of passing the full list.

### 5. Custom Configuration (`05_custom_configuration.py`)
Tune agent behavior — sliding context window, model selection, and more.

### 6. Agent Identity (`06_agent_identity.py`)
Give an agent a name, role, tool subset, and model via `AgentSpec`. Foundation
for multi-agent systems where each agent has a distinct persona.

### 7. Skills (`07_skills.py`)
Bundle reusable Markdown instructions as skills and attach them via
`add_skill()`, `load_skills()`, or `AgentSpec(skill_names=[...])`.

### 8. Human-in-the-Loop (`08_human_in_the_loop.py`)
Pause execution before each code block for review. Demonstrates
`require_approval`, `agent.is_interrupted`, `agent.resume()`, and
`agent.reject(feedback)`.

### 9. Async API and Streaming (`09_async_and_streaming.py`)
Two async patterns in one file:
- **`arun()`** — non-blocking equivalent of `run()`; use in async orchestrators
  or with `asyncio.gather()` to run agents concurrently.
- **`run_stream()`** — async generator that yields `AgentEvent` objects during
  execution; use to feed a UI or collect a real-time trace.

### 10. Error Handling (`10_error_handling.py`)
Configure iteration limits, cost budgets, and consecutive-error thresholds.
Catch `LLMError`, `AgentTimeoutError`, and `BaseAgentError`.

### 11. Multi-Agent Orchestration (`11_multi_agent.py`)
Coordinate specialist agents with `AgentTeam`. A supervisor LLM routes work
between agents until the task is complete or `MaxRoundsExceededError` is
raised.

### 12. Persistent Conversations (`12_persistence.py`)
Persist conversation history across runs with `checkpoint_db_path` and
`thread_id`. Pass a file path instead of the default `":memory:"` to resume
a session after a process restart.

## Configuration Files

- `mcp_config.yaml` — Example MCP server configuration
- `skills/` — Example skill bundles used by `07_skills.py`

## Need Help?

- Check the main [README](../README.md) for detailed documentation
- Visit the [GitHub repository](https://github.com/BinglanLi/BaseAgent) for issues and discussions
