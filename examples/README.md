# BaseAgent Examples

This directory contains example scripts demonstrating various features of BaseAgent.

## Getting Started

Make sure you have BaseAgent installed:

```bash
pip install baseagent
# or for development
pip install -e .
```

Set up your environment variables in a `.env` file:

```bash
ANTHROPIC_API_KEY=your_key_here
# or other provider keys
```

## Examples

### 1. Basic Usage (`01_basic_usage.py`)
Initialize BaseAgent and run a simple task. Shows how to read usage metrics.

```bash
python examples/01_basic_usage.py
```

### 2. Custom Tools (`02_custom_tools.py`)
Add your own Python functions as agent tools. Tool name, description, and
parameter schema are derived automatically from the function's docstring and
type hints.

```bash
python examples/02_custom_tools.py
```

### 3. MCP Servers (`03_mcp_servers.py`)
Integrate Model Context Protocol servers for extended functionality (local
stdio and remote Streamable HTTP transports).

```bash
python examples/03_mcp_servers.py
```

### 4. Resource Management (`04_resource_management.py`)
Register data sources and software libraries so the agent can reference them
when planning tasks.

```bash
python examples/04_resource_management.py
```

### 5. Custom Configuration (`05_custom_configuration.py`)
Customize agent behavior — sliding context window, model selection, and more.

```bash
python examples/05_custom_configuration.py
```

### 6. Tool Retrieval (`06_tool_retrieval.py`)
Enable automatic tool selection (`use_tool_retriever=True`) so the agent
picks relevant tools by semantic similarity rather than passing the full list.

```bash
python examples/06_tool_retrieval.py
```

### 7. Human-in-the-Loop (`07_human_in_the_loop.py`)
Pause execution before each code block for human review. Demonstrates
`require_approval`, `agent.is_interrupted`, `agent.resume()`, and
`agent.reject(feedback)`.

```bash
python examples/07_human_in_the_loop.py
```

### 8. Agent Identity (`08_agent_identity.py`)
Customise the agent's role, tool subset, and model via `AgentSpec`.

```bash
python examples/08_agent_identity.py
```

### 9. REPL Isolation (`09_repl_isolation.py`) *(developer / no API key needed)*
Demonstrates per-instance REPL namespace isolation and plot capture internals.
Uses `unittest.mock` — no API key required.

```bash
python examples/09_repl_isolation.py
```

### 10a. Extract Subgraph (`10_extract_subgraph.py`) *(developer / no API key needed)*
Shows how to extract the underlying LangGraph `StateGraph` for embedding in
a larger pipeline. Uses `unittest.mock` — no API key required.

> **Note:** Two files share the `10_` prefix (`10_extract_subgraph.py` and
> `10_skills.py`). This is a known numbering quirk; both files are valid.

```bash
python examples/10_extract_subgraph.py
```

### 10b. Skills (`10_skills.py`)
Bundle reusable Markdown instructions as skills and attach them to an agent
via `add_skill()`, `load_skills()`, or `AgentSpec(skill_names=[...])`.

```bash
python examples/10_skills.py
```

### 11. Error Handling and Termination (`11_error_handling.py`)
Configure iteration limits, cost budgets, and consecutive-error thresholds.
Catch structured errors (`LLMError`, `AgentTimeoutError`, `BaseAgentError`).

```bash
python examples/11_error_handling.py
```

### 12. Async API (`12_async_api.py`)
Use `arun()`, `aresume()`, and `areject()` in async contexts. Demonstrates
`asyncio.gather()` for concurrent agents.

```bash
python examples/12_async_api.py
```

### 13. Multi-Agent Orchestration (`13_multi_agent.py`)
Coordinate multiple specialist agents with `AgentTeam`. A supervisor LLM
routes work between agents until the task is complete or `max_rounds` is
reached (`MaxRoundsExceededError`).

```bash
python examples/13_multi_agent.py
```

### 15. Streaming Events (`15_streaming.py`)
Stream typed `AgentEvent` objects in real time using `run_stream()`. Filter
to specific `EventType` values (e.g. `FINAL_ANSWER`, `THINKING`) or
serialise events to JSON for WebSocket / SSE payloads.

```bash
python examples/15_streaming.py
```

### 16. Persistent Conversations (`16_persistence.py`)
Persist conversation history across runs with `checkpoint_db_path` and
`thread_id`. Pass a file path instead of the default `":memory:"` to
resume a session after a process restart.

```bash
python examples/16_persistence.py
```

## Configuration Files

- `mcp_config.yaml` — Example MCP server configuration file

## Need Help?

- Check the main [README](../README.md) for detailed documentation
- Visit the [GitHub repository](https://github.com/BinglanLi/BaseAgent) for issues and discussions
- Review the [Contributing Guide](../CONTRIBUTING.md) to contribute examples
