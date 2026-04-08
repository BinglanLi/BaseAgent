# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial package structure
- Core BaseAgent functionality
- Multi-provider LLM support (OpenAI, Anthropic, Google, AWS, Groq)
- Resource management system
- Tool registration and retrieval
- MCP server integration
- Usage metrics tracking
- Comprehensive test suite
- Remote MCP server support via Streamable HTTP transport (`streamablehttp_client`)
- Auth headers for remote MCP servers with `${ENV_VAR}` interpolation
- **REPL Namespace Isolation**: each `BaseAgent` instance owns an isolated `_repl_namespace` dict; variables from one agent's `<execute>` blocks are invisible to other instances
- `PlotCapture` class in `support_tools.py` — per-instance matplotlib plot buffer replacing the module-level `_captured_plots` global
- `namespace` parameter on `run_python_repl()` — selects execution namespace; `None` falls back to the module-level global for backward compatibility
- `namespace` parameter on `inject_custom_functions_to_repl()` — injects custom tools into a specific namespace; `None` falls back to global
- 24 new unit tests in `tests/test_repl_isolation.py` covering all four isolation phases

### Fixed
- MCP async/sync bridge: `make_mcp_wrapper` now returns values (not unawaited Tasks) in Jupyter/nested event loop contexts

## [0.1.0] - 2025-11-12

### Added
- Initial release of BaseAgent
- Core agent functionality with LangGraph integration
- Dynamic tool management system
- Resource manager for tools, data, and libraries
- Support for multiple LLM providers
- Built-in Python REPL and Bash execution tools
- Configurable agent behavior
- Usage tracking and metrics
- Test coverage for core functionality

