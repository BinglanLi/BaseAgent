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
OPENAI_API_KEY=your_key_here
# or other provider keys
```

## Examples

### 1. Basic Usage (`01_basic_usage.py`)
Learn how to initialize BaseAgent and run simple tasks.

```bash
python examples/01_basic_usage.py
```

### 2. Custom Tools (`02_custom_tools.py`)
Add your own custom functions as tools that the agent can use.

```bash
python examples/02_custom_tools.py
```

### 3. MCP Servers (`03_mcp_servers.py`)
Integrate Model Context Protocol servers for extended functionality.

```bash
python examples/03_mcp_servers.py
```

### 4. Resource Management (`04_resource_management.py`)
Manage tools, data sources, and software libraries.

```bash
python examples/04_resource_management.py
```

### 5. Custom Configuration (`05_custom_configuration.py`)
Customize agent behavior with configuration options.

```bash
python examples/05_custom_configuration.py
```

### 6. Tool Retrieval (`06_tool_retrieval.py`)
Enable automatic tool selection based on task requirements.

```bash
python examples/06_tool_retrieval.py
```

## Configuration Files

- `mcp_config.yaml` - Example MCP server configuration file

## Need Help?

- Check the main [README](../README.md) for detailed documentation
- Visit the [GitHub repository](https://github.com/BinglanLi/BaseAgent) for issues and discussions
- Review the [Contributing Guide](../CONTRIBUTING.md) to contribute examples

