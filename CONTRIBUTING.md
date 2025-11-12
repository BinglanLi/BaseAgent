# Contributing to BaseAgent

Thank you for your interest in contributing to BaseAgent! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/BinglanLi/BaseAgent.git
   cd BaseAgent
   ```
3. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. **Install development dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bugfix:

```bash
git checkout -b feature/your-feature-name
# or
git checkout -b fix/your-bugfix-name
```

### 2. Make Your Changes

- Write clean, readable code
- Follow the existing code style
- Add docstrings to functions and classes
- Update tests as needed
- Update documentation if you're changing functionality

### 3. Run Tests

Before submitting, make sure all tests pass:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=BaseAgent --cov-report=html

# Run specific test file
pytest BaseAgent/tests/test_integration.py
```

### 4. Code Quality

Run code quality tools:

```bash
# Format code
black BaseAgent/

# Check linting
ruff check BaseAgent/

# Type checking
mypy BaseAgent/
```

### 5. Commit Your Changes

Write clear, descriptive commit messages:

```bash
git add .
git commit -m "Add feature: description of your changes"
```

Good commit message examples:
- `Add support for custom LLM providers`
- `Fix bug in tool retrieval when no tools match`
- `Update documentation for resource manager`
- `Refactor llm.py for better error handling`

### 6. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Then open a Pull Request on GitHub with:
- Clear title describing the change
- Description of what changed and why
- Reference to any related issues
- Screenshots if relevant (for UI changes)

## Code Style Guidelines

### Python Style

- Follow PEP 8
- Use type hints where possible
- Maximum line length: 100 characters
- Use meaningful variable names
- Add docstrings to all public functions/classes

Example:

```python
def add_tool(
    self,
    name: str,
    function: Callable,
    description: str,
    required_parameters: list[dict] | None = None,
    optional_parameters: list[dict] | None = None,
) -> None:
    """
    Add a custom tool to the agent.
    
    Args:
        name: Unique name for the tool
        function: The callable function to execute
        description: Description of what the tool does
        required_parameters: List of required parameter definitions
        optional_parameters: List of optional parameter definitions
        
    Raises:
        ValueError: If a tool with this name already exists
    """
    # Implementation
```

### Testing

- Write tests for new features
- Maintain or improve code coverage
- Use descriptive test names
- Test edge cases and error conditions

Example:

```python
def test_add_tool_with_custom_function():
    """Test that custom functions can be added as tools."""
    agent = BaseAgent()
    
    def custom_func(x: int) -> int:
        return x * 2
    
    agent.add_tool(
        name="double",
        function=custom_func,
        description="Doubles a number",
        required_parameters=[{"name": "x", "type": "int"}]
    )
    
    tool = agent.resource_manager.find_tool_by_name("double")
    assert tool is not None
    assert tool.name == "double"
```

## Types of Contributions

### Bug Fixes

- Check if the bug is already reported in Issues
- If not, create an issue describing the bug
- Reference the issue in your PR

### New Features

- Discuss major features in an issue first
- Ensure the feature fits with the project goals
- Update documentation
- Add tests

### Documentation

- Fix typos or clarify existing docs
- Add examples
- Improve README or guides

### Tests

- Add missing test coverage
- Improve existing tests
- Add integration tests

## Pull Request Review Process

1. **Automated checks** will run (tests, linting)
2. **Maintainer review** - a project maintainer will review your code
3. **Feedback** - address any requested changes
4. **Approval** - once approved, your PR will be merged

## Questions or Need Help?

- Open an issue with your question
- Tag it with the "question" label
- Be specific about what you need help with

## Code of Conduct

- Be respectful and inclusive
- Welcome newcomers
- Focus on constructive feedback
- Assume good intentions

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to BaseAgent! 🚀

