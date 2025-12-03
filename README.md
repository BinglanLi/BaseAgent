# BaseAgent

A flexible and extensible agent framework built on LangChain and LangGraph for creating intelligent AI agents with resource management, tool integration, and multi-provider LLM support.

## Features

- 🤖 **Flexible LLM Support** - Works with OpenAI, Anthropic, Google Gemini, AWS Bedrock, Groq, and custom providers
- 🔧 **Dynamic Tool Integration** - Easy-to-use tool registration and management system
- 📊 **Resource Management** - Built-in management for tools, data lakes, and software libraries
- 🔄 **MCP Server Integration** - Support for Model Context Protocol servers
- 🧠 **State Management** - Powered by LangGraph for complex agent workflows
- 📈 **Usage Tracking** - Built-in metrics for token usage and cost monitoring
- 🔍 **Tool Retrieval** - Intelligent tool selection based on task requirements

## Installation

### Basic Installation

```bash
pip install baseagent
```

### With Optional Dependencies

```bash
# For specific LLM providers
pip install baseagent[openai]      # OpenAI GPT models
pip install baseagent[anthropic]   # Anthropic Claude models
pip install baseagent[google]      # Google Gemini models
pip install baseagent[aws]         # AWS Bedrock models
pip install baseagent[groq]        # Groq models

# Install all providers at once
pip install baseagent[all]

# For development
pip install baseagent[dev]
```

### From Source

```bash
git clone https://github.com/BinglanLi/BaseAgent.git
cd BaseAgent
pip install -e .
```

## Quick Start

```python
from BaseAgent import BaseAgent

# Initialize the agent
agent = BaseAgent(llm="gpt-4", path="./data")

# Run a task
result = agent.go("Analyze the dataset and create a visualization")
print(result)
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Choose your LLM provider
OPENAI_API_KEY=your_openai_key
ANTHROPIC_API_KEY=your_anthropic_key
GOOGLE_API_KEY=your_google_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
GROQ_API_KEY=your_groq_key
# Use the following variables for Azure Foundry models
# Azure Foundry - Claude models
ANTHROPIC_FOUNDRY_BASE_URL='https://<your_resource>.services.ai.azure.com/anthropic'
ANTHROPIC_FOUNDRY_API_KEY=your_azure_foundry_key
# Azure Foundry - other models: GPT, GPT-OSS, Mistral etc.
AZURE_FOUNDRY_BASE_URL='https://<your_resource>.openai.azure.com/openai/v1/'
AZURE_FOUNDRY_API_KEY=your_azure_foundry_key
AZURE_FOUNDRY_API_VERSION='2024-05-01-preview'
# Certain MCPs require API keys
GITHUB_TOKEN=your_personal_github_token # not ssh keys
```

### Agent Configuration

```python
from BaseAgent import BaseAgent
from BaseAgent.config import default_config

config = default_config.copy()
config["max_iterations"] = 20
config["temperature"] = 0.7

# BaseAgent will try to find the correct LLM provider source based on the model name
agent = BaseAgent(llm="gpt-4", config=config) # OpenAI
agent = BaseAgent(llm="claude-sonnet-4-5-20250929", config=config) # Anthropic
# You can specify the LLM provider source
agent = BaseAgent(llm='gpt-5.1', source='AzureOpenAI')
agent = BaseAgent(llm='claude-sonnet-4-5', source='AnthropicFoundry')
```

## Examples

Check out the [`examples/`](examples/) directory for detailed usage examples:

- **[Basic Usage](examples/01_basic_usage.py)** - Get started with simple tasks
- **[Custom Tools](examples/02_custom_tools.py)** - Add your own functions as tools
- **[MCP Servers](examples/03_mcp_servers.py)** - Integrate Model Context Protocol servers
- **[Resource Management](examples/04_resource_management.py)** - Manage tools, data, and libraries
- **[Custom Configuration](examples/05_custom_configuration.py)** - Customize agent behavior
- **[Tool Retrieval](examples/06_tool_retrieval.py)** - Automatic tool selection

### Run Examples

```bash
# Make sure you have BaseAgent installed
pip install -e .

# Run any example
python examples/01_basic_usage.py
```

## Development

### Local Installation & Testing

```bash
# Install in development mode
pip install -e .

# Verify installation
python -c "from BaseAgent import BaseAgent; print('✅ BaseAgent imported successfully!')"
python -c "from BaseAgent import BaseAgent; print(f'Version: {BaseAgent.__version__}')"
```

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all tests
pytest

# Run with coverage
pytest --cov=BaseAgent --cov-report=html

# View coverage report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Code Quality

```bash
# Format code
black BaseAgent/

# Lint
ruff check BaseAgent/

# Type checking
mypy BaseAgent/
```

### Building the Package

```bash
# Install build tools
pip install build twine

# Build distribution
python -m build

# This creates:
# - dist/baseagent-0.1.0-py3-none-any.whl
# - dist/baseagent-0.1.0.tar.gz

# Test in fresh environment
python -m venv test_env
source test_env/bin/activate
pip install dist/baseagent-0.1.0-py3-none-any.whl
python -c "from BaseAgent import BaseAgent"
```

## Publishing

### Test PyPI (Recommended First)

```bash
# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ baseagent
```

### Production PyPI

```bash
# Upload to PyPI
twine upload dist/*

# Now available to everyone
pip install baseagent
```

## Architecture

BaseAgent is built on several key components:

- **BaseAgent** - Main agent orchestrator with LangGraph state management
- **ResourceManager** - Centralized management for tools, data, and libraries
- **ToolRetriever** - Intelligent tool selection based on task semantics
- **LLM Integration** - Multi-provider support (OpenAI, Anthropic, Google, AWS, Groq)
- **Tool System** - Dynamic tool registration and execution

## Project Structure

```
BaseAgent/
├── BaseAgent/                      # Main package
│   ├── base_agent.py              # Core agent implementation
│   ├── config.py                  # Configuration management
│   ├── llm.py                     # LLM provider integrations
│   ├── prompts.py                 # Prompt templates
│   ├── resource_manager.py        # Resource management
│   ├── resources.py               # Resource models (Pydantic)
│   ├── retriever.py               # Tool retrieval logic
│   ├── utils.py                   # Utility functions
│   ├── env_desc.py                # Environment descriptions
│   ├── data_lake/                 # Built-in data files
│   ├── tools/                     # Tool implementations
│   └── tests/                     # Test suite
├── examples/                       # Usage examples
│   ├── 01_basic_usage.py
│   ├── 02_custom_tools.py
│   ├── 03_mcp_servers.py
│   ├── 04_resource_management.py
│   ├── 05_custom_configuration.py
│   ├── 06_tool_retrieval.py
│   └── README.md
├── .github/workflows/ci.yml       # CI/CD configuration
├── README.md                      # Main documentation
├── LICENSE                        # MIT License
├── CONTRIBUTING.md                # Contribution guidelines
├── CHANGELOG.md                   # Version history
├── pyproject.toml                 # Package configuration
└── MANIFEST.in                    # Distribution manifest
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Troubleshooting

### Import Errors

If you get import errors after installation:

```bash
# Make sure you're not in the project directory
cd /tmp
python -c "from BaseAgent import BaseAgent"
```

### Dependency Conflicts

```bash
# Create a fresh environment
python -m venv fresh_env
source fresh_env/bin/activate
pip install baseagent
```

### Build Issues

```bash
# Clean build artifacts
rm -rf build/ dist/ *.egg-info BaseAgent/__pycache__
python -m build
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [LangChain](https://github.com/langchain-ai/langchain) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Inspired by the [Biomni](https://github.com/openbmb/Biomni) project

## Support

For questions, issues, or feature requests, please open an issue on the [GitHub repository](https://github.com/BinglanLi/BaseAgent/issues).

## Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI](https://pypi.org/)
- [Test PyPI](https://test.pypi.org/)
- [Semantic Versioning](https://semver.org/)
