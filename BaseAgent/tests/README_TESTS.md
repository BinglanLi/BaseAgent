# Unit Tests for BaseAgent

This directory contains comprehensive unit and integration tests for BaseAgent functionality using pytest.

## Quick Start

### Install pytest

```bash
# Using pip
pip install pytest pytest-cov

# Using uv (if using uv for package management)
uv pip install pytest pytest-cov
```

### Run All Tests

```bash
# From the project root
cd /Users/lib/GitHub/agent-playground
pytest BaseAgent/tests/ -v

# Or from the tests directory
cd BaseAgent/tests
pytest -v
```

### Run Specific Test Files

```bash
# Test add_tool functionality
pytest BaseAgent/tests/test_add_tool.py -v

# Test add_mcp functionality
pytest BaseAgent/tests/test_add_mcp.py -v

# Test schema generation
pytest BaseAgent/tests/test_hybrid_schema.py -v

# Integration tests
pytest BaseAgent/tests/test_integration.py -v

# LLM usage metrics
pytest BaseAgent/tests/test_llm_usage_metrics.py -v
```

### Run Tests by Marker

```bash
# Run only integration tests
pytest BaseAgent/tests/ -m integration -v

# Skip integration tests (run only unit tests)
pytest BaseAgent/tests/ -m "not integration" -v

# Run MCP tests (will skip if config not available)
pytest BaseAgent/tests/ -m mcp -v
```

## Test Organization

### Unit Tests

#### `test_add_tool.py`
Tests for custom tool registration via `add_tool()`:
- ✅ Simple function with basic parameters
- ✅ Complex function with multiple parameter types
- ✅ Functions without type hints
- ✅ REPL injection (functions available in code execution)
- ✅ Prompt generation (tools appear in system prompt)
- ✅ Selection state management
- ✅ Schema generation and validation

#### `test_add_mcp.py`
Tests for MCP (Model Context Protocol) tool integration:
- ✅ MCP tool loading from config file
- ✅ Tools stored in ResourceManager
- ✅ MCP tools available in REPL
- ✅ MCP tools appear in prompts
- ✅ Module naming conventions

**Note:** MCP tests require a `test_mcp_config.yaml` file. Tests will be automatically skipped if the config file is not present.

#### `test_hybrid_schema.py`
Tests for API schema generation from functions:
- ✅ Schema extraction from typed functions
- ✅ Schema extraction from untyped functions
- ✅ Docstring parsing
- ✅ Complex type handling (List, Dict, Optional)
- ✅ String input handling
- ✅ Type conversion utilities

#### `test_llm_usage_metrics.py`
Tests for LLM usage tracking:
- ✅ Usage metrics extraction from different providers
- ✅ OpenAI token usage
- ✅ Anthropic token usage
- ✅ Bedrock usage metrics
- ✅ Ollama usage metrics

### Integration Tests

#### `test_integration.py`
Full end-to-end workflow tests:
- ✅ Complete tool workflow (add → register → inject → verify)
- ✅ Multiple tool management
- ✅ Tool isolation and independence
- ✅ Tool selection behavior
- ✅ Error handling in custom tools

## Test Configuration

Tests are configured via `pytest.ini` with the following settings:

### Markers
- `integration`: Integration tests (can be skipped for faster unit tests)
- `slow`: Slow-running tests
- `mcp`: Tests requiring MCP configuration
- `llm`: Tests requiring LLM API access

### Running Tests with Coverage

```bash
# Run with coverage report
pytest BaseAgent/tests/ --cov=BaseAgent --cov-report=html --cov-report=term-missing

# View HTML coverage report
open htmlcov/index.html  # macOS
# or
xdg-open htmlcov/index.html  # Linux
```

## Shared Fixtures

The tests use shared fixtures defined in `conftest.py`:

### Available Fixtures
- `base_agent`: Fresh BaseAgent instance
- `sample_function`: Simple greet function for testing
- `typed_function`: Well-typed calculate_sum function
- `complex_function`: Function with multiple parameter types
- `math_functions`: Tuple of (add, multiply, power) functions
- `mcp_config_path`: Path to MCP test config file
- `clear_repl_namespace`: Clears REPL namespace before/after test
- `temp_file`: Helper for creating temporary test files

## Writing New Tests

### Basic Test Structure

```python
class TestMyFeature:
    """Test my new feature."""
    
    def test_basic_functionality(self, base_agent: BaseAgent):
        """Test basic functionality."""
        # Arrange
        def my_func(x: int) -> int:
            """My test function."""
            return x * 2
        
        # Act
        base_agent.add_tool(my_func)
        
        # Assert
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 1
        assert custom_tools[0].name == 'my_func'
```

### Using Fixtures

```python
def test_with_fixtures(self, base_agent: BaseAgent, sample_function: Callable):
    """Test using provided fixtures."""
    base_agent.add_tool(sample_function)
    assert len(base_agent.resource_manager.collection.custom_tools) == 1
```

### Parametrized Tests

```python
@pytest.mark.parametrize("input,expected", [
    (5, 10),
    (10, 20),
    (0, 0),
])
def test_with_parameters(self, base_agent: BaseAgent, input: int, expected: int):
    """Test with multiple parameter sets."""
    def double(x: int) -> int:
        return x * 2
    
    base_agent.add_tool(double)
    tool = base_agent.resource_manager.collection.custom_tools[0]
    assert tool.function(input) == expected
```

### Marking Tests

```python
@pytest.mark.integration
class TestIntegration:
    """Integration tests that can be skipped for faster runs."""
    pass

@pytest.mark.slow
def test_slow_operation(self, base_agent: BaseAgent):
    """Test that takes a long time."""
    pass
```

## Troubleshooting

### Import Errors

```bash
# Make sure you're in the project root
cd /Users/lib/GitHub/agent-playground

# Run tests from there
pytest BaseAgent/tests/ -v
```

### LLM API Errors

Some tests create BaseAgent instances which may require LLM configuration:
- Set API keys in environment variables
- Configure `.env` file
- Or set up `default_config`

### MCP Tests Skipped

This is normal if you haven't set up MCP configuration. The tests will automatically skip.

To enable MCP tests:
1. Create `BaseAgent/tests/test_mcp_config.yaml`
2. Add MCP server configuration (see example in `test_add_mcp.py`)
3. Install required MCP servers

### Test Discovery Issues

If pytest doesn't find your tests:
- Ensure test files are named `test_*.py`
- Ensure test classes are named `Test*`
- Ensure test functions are named `test_*`
- Check `pytest.ini` configuration

## Continuous Integration

To run tests in CI/CD pipelines:

```bash
# Install dependencies
pip install pytest pytest-cov

# Run tests with coverage
pytest BaseAgent/tests/ \
    --cov=BaseAgent \
    --cov-report=xml \
    --cov-report=term-missing \
    -v

# Run only unit tests (skip integration)
pytest BaseAgent/tests/ -m "not integration" -v
```

## Test Performance

### Running Faster Tests

```bash
# Skip integration tests
pytest BaseAgent/tests/ -m "not integration"

# Skip slow tests
pytest BaseAgent/tests/ -m "not slow"

# Run in parallel (requires pytest-xdist)
pip install pytest-xdist
pytest BaseAgent/tests/ -n auto
```

### Debugging Failed Tests

```bash
# Stop on first failure
pytest BaseAgent/tests/ -x

# Show local variables in failures
pytest BaseAgent/tests/ --showlocals

# Enter debugger on failure
pytest BaseAgent/tests/ --pdb

# Verbose output
pytest BaseAgent/tests/ -vv
```

## Example: Quick Manual Test

Want to test quickly in a notebook or REPL? Here's a minimal example:

```python
from BaseAgent.base_agent import BaseAgent

# Create agent
agent = BaseAgent()

# Define and add a custom function
def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b

agent.add_tool(multiply)

# Verify it's registered
print(f"Custom tools: {len(agent.resource_manager.collection.custom_tools)}")
tool = agent.resource_manager.collection.custom_tools[0]
print(f"Tool name: {tool.name}")
print(f"Has function: {tool.function is not None}")

# Test the function
result = tool.function(5, 7)
print(f"Result: {result}")  # Should print: 35
```

## Further Reading

- [pytest documentation](https://docs.pytest.org/)
- [pytest fixtures](https://docs.pytest.org/en/stable/fixture.html)
- [pytest markers](https://docs.pytest.org/en/stable/mark.html)
- [pytest parametrize](https://docs.pytest.org/en/stable/parametrize.html)

