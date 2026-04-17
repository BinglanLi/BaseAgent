# Unit Tests for BaseAgent

This directory contains comprehensive unit and integration tests for BaseAgent using pytest.

## Quick Start

### Run All Tests (No API Keys Required)

```bash
# From the project root
cd /Users/lib/GitHub/BaseAgent
pytest BaseAgent/tests/ -v

# Or from the tests directory
cd BaseAgent/tests
pytest -v
```

### Run Fast Unit Tests Only

```bash
# Recommended for development — no API keys needed, runs in seconds
pytest BaseAgent/tests/ -m unit

# Or skip integration/LLM/MCP markers directly
pytest BaseAgent/tests/ -m "not integration and not llm and not mcp"
```

### Run Specific Test Files

```bash
pytest BaseAgent/tests/test_add_tool.py -v
pytest BaseAgent/tests/test_nodes.py -v
pytest BaseAgent/tests/test_async_api.py -v
pytest BaseAgent/tests/test_utils_schema.py -v
pytest BaseAgent/tests/test_integration.py -v        # requires API key
pytest BaseAgent/tests/test_mcp_integration.py -v    # requires MCP config
pytest BaseAgent/tests/test_llm_providers.py -v      # requires API key
```

## Test Organization

### Shared Test Helpers (`helpers/`)

The `helpers/` directory contains factory functions used across multiple test files:

- **`helpers/node_helpers.py`** — three factory functions:
  - `make_mock_agent_attrs(**overrides)` — returns a `MagicMock` agent used by `NodeExecutor` unit tests
  - `make_state(messages, pending_code, pending_language, next_step)` — builds a minimal graph state dict
  - `make_base_agent(llm_content, **kwargs)` — returns a real `BaseAgent` with a patched LLM (no API key needed)

Import them as:
```python
from helpers.node_helpers import make_mock_agent_attrs, make_state, make_base_agent
```

### Unit Tests

All unit test files carry `pytestmark = pytest.mark.unit` at module level, so `pytest -m unit` collects them reliably.

| File | What it tests |
|------|--------------|
| `test_add_tool.py` | `add_tool()` registration, schema generation, prompt injection |
| `test_agent_spec.py` | `AgentSpec` fields, source overrides, spec propagation |
| `test_async_api.py` | `arun()` / `astream()` async API, stream logging deduplication |
| `test_checkpointing.py` | Graph checkpointing, resume from saved state |
| `test_config_termination.py` | Termination conditions (max iterations, cost, errors) |
| `test_context_window.py` | Context window trimming logic |
| `test_errors.py` | Error handling and propagation |
| `test_events.py` | Event emission and ordering |
| `test_examples_streaming_persistence.py` | Streaming + persistence example workflows |
| `test_extract_subgraph.py` | `extract_subgraph()` / delegate configuration |
| `test_interrupt_resume.py` | Human-in-the-loop interrupts, resume, stream logging |
| `test_llm_usage_metrics.py` | Token usage extraction (Anthropic, OpenAI, Bedrock, Ollama) |
| `test_mcp_unit.py` | MCP tool loading without live MCP server |
| `test_multi_agent.py` | `AgentTeam` orchestration |
| `test_nodes.py` | Individual graph node functions |
| `test_repl_isolation.py` | REPL namespace isolation between runs |
| `test_resource_manager.py` | `ResourceManager` CRUD and retrieval |
| `test_retriever.py` | Tool retriever / embedding-based selection |
| `test_skills.py` | Skill loading, injection, and prompt generation |
| `test_utils_download.py` | Download utilities |
| `test_utils_formatting.py` | Message formatting and pretty-print helpers |
| `test_utils_schema.py` | `function_to_api_schema()`, docstring parsing, type conversion |

### Integration / External Tests

These tests require API keys or external services and are excluded from `pytest -m unit`:

| File | Marker | Requires |
|------|--------|---------|
| `test_integration.py` | `integration` | LLM API key |
| `test_mcp_integration.py` | `mcp` | `test_mcp_config.yaml` + MCP server |
| `test_llm_providers.py` | `llm` | LLM API key |

## Markers

Defined in `pytest.ini`:

| Marker | Meaning |
|--------|---------|
| `unit` | Fast, no I/O, no API keys — the default development test suite |
| `integration` | End-to-end tests that hit a real LLM |
| `mcp` | Tests that require a live MCP server config |
| `llm` | Tests that exercise LLM provider connectivity |
| `slow` | Long-running tests |

### Running Tests by Marker

```bash
# Fast development loop (no API keys)
pytest BaseAgent/tests/ -m unit

# Full suite minus live-service tests
pytest BaseAgent/tests/ -m "not integration and not llm and not mcp"

# Only integration tests
pytest BaseAgent/tests/ -m integration -v

# Only MCP tests
pytest BaseAgent/tests/ -m mcp -v
```

## Test Configuration

`pytest.ini` (project root) defines markers and test paths.

### Running Tests with Coverage

```bash
pytest BaseAgent/tests/ --cov=BaseAgent --cov-report=html --cov-report=term-missing
open htmlcov/index.html  # macOS
```

## Shared Fixtures (`conftest.py`)

| Fixture | Description |
|---------|-------------|
| `base_agent` | Fresh `BaseAgent` instance (patched LLM) |
| `sample_function` | Simple greet function |
| `typed_function` | `calculate_sum(x, y, verbose)` |
| `complex_function` | Function with multiple parameter types |
| `math_functions` | Tuple of `(add, multiply, power)` |
| `mcp_config_path` | Path to `test_mcp_config.yaml` |
| `clear_repl_namespace` | Clears REPL namespace before/after each test |
| `temp_file` | Creates a temporary file, yields path, then removes it |

## Writing New Tests

### Basic structure

```python
import pytest
from helpers.node_helpers import make_base_agent

pytestmark = pytest.mark.unit


class TestMyFeature:
    def test_basic(self):
        agent = make_base_agent()
        agent.add_tool(lambda x: x, )
        assert len(agent.resource_manager.collection.custom_tools) == 1
```

### Marking integration tests

```python
@pytest.mark.integration
class TestFullWorkflow:
    def test_end_to_end(self):
        ...
```

## Troubleshooting

**Import errors** — run from the project root (`/Users/lib/GitHub/BaseAgent`) or from `BaseAgent/tests/`:
```bash
cd BaseAgent/tests && pytest -v
```

**LLM API errors** — unit tests patch the LLM and don't need API keys. Only `integration` / `llm` tests do.

**MCP tests skipped** — expected when `test_mcp_config.yaml` is missing. Create it and configure an MCP server to enable them.

**Test discovery issues** — ensure files are `test_*.py`, classes `Test*`, functions `test_*`.

## Continuous Integration

```bash
# Install dependencies
pip install pytest pytest-cov

# Unit tests only (no API keys needed)
pytest BaseAgent/tests/ -m unit --cov=BaseAgent --cov-report=xml

# Full suite
pytest BaseAgent/tests/ -m "not integration and not llm and not mcp" \
    --cov=BaseAgent --cov-report=xml --cov-report=term-missing -v
```

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [pytest parametrize](https://docs.pytest.org/en/stable/parametrize.html)
