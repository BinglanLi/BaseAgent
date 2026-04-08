import os
import re
import uuid
import warnings
from collections import defaultdict
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    from BaseAgent.events import AgentEvent
from dotenv import load_dotenv

from langchain_core.messages import HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _HAS_SQLITE_SAVER = True
except ImportError:
    _HAS_SQLITE_SAVER = False

from BaseAgent.llm import SourceType, get_llm
from BaseAgent.state import AgentState
from BaseAgent.prompts import (
    get_base_prompt_template,
    get_environment_resources_section,
    _PROMPT_CUSTOM_RESOURCES_SECTION,
    _CUSTOM_TOOLS_SECTION,
    _CUSTOM_DATA_SECTION,
    _CUSTOM_SOFTWARE_SECTION,
    _SKILLS_SECTION,
    _SKILL_ENTRY_TEMPLATE,
    _DEFAULT_ROLE_DESCRIPTION,
)
from BaseAgent.resources import Skill
from BaseAgent.agent_spec import AgentSpec
from BaseAgent.config import default_config
from BaseAgent.resource_manager import ResourceManager
from BaseAgent.retriever import ToolRetriever
from BaseAgent.tools.support_tools import PlotCapture, run_python_repl
from BaseAgent.env_desc import data_lake_items, libraries
from BaseAgent.utils.tool_bridge import inject_custom_functions_to_repl
from BaseAgent.utils.formatting import pretty_print
from BaseAgent.utils.schema import function_to_api_schema

if os.path.exists(".env"):
    load_dotenv(".env", override=True)
    print("Loaded environment variables from .env")

class BaseAgent:
    def __init__(
        self,
        path: str | None = None,
        llm: str | None = None,
        source: SourceType | None = None,
        timeout_seconds: int | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        use_tool_retriever: bool | None = None,
        checkpoint_db_path: str | None = None,
        require_approval: str | None = None,
        skills_directory: str | None = None,
        spec: AgentSpec | None = None,
    ):
        """
        Args:
            path: The path to the data.
            llm: The LLM to use.
            source: The source of the LLM, e.g., "OpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom".
            timeout_seconds: The timeout in seconds.
            base_url: The base URL of the LLM.
            api_key: The API key of the LLM.
            checkpoint_db_path: Path to SQLite checkpoint DB. Defaults to ":memory:" (ephemeral).
                Set to a file path (e.g. "checkpoints.db") for persistence across sessions.
            require_approval: When to pause for human code review. One of:
                "never" (default) — never interrupt;
                "always" — interrupt before every code block;
                "dangerous_only" — interrupt only for bash/R code.
            skills_directory: Path to a directory of SKILL.md files to load on startup.
            spec: Optional agent identity and persona.  When provided, ``spec.name``,
                ``spec.role``, ``spec.tool_names``, ``spec.skill_names``,
                ``spec.llm``, ``spec.source``, and ``spec.temperature`` override
                the corresponding defaults.  Explicit keyword arguments take
                priority over spec fields.
        """
        self.spec = spec

        # Agent name: from spec, or default
        self.name: str = spec.name if spec is not None else "agent"

        # Resolve settings: explicit kwargs > spec > default_config
        self.path = path if path is not None else default_config.path
        self.llm_model_name = (
            llm if llm is not None
            else (spec.llm if spec is not None and spec.llm is not None else default_config.llm)
        )
        self.source = (
            source if source is not None
            else (spec.source if spec is not None and spec.source is not None else default_config.source)
        )
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else default_config.timeout_seconds
        self.base_url = base_url if base_url is not None else default_config.base_url
        self.api_key = api_key if api_key is not None else default_config.api_key
        self.use_tool_retriever = use_tool_retriever if use_tool_retriever is not None else default_config.use_tool_retriever
        self.checkpoint_db_path = checkpoint_db_path if checkpoint_db_path is not None else default_config.checkpoint_db_path
        self.require_approval = require_approval if require_approval is not None else default_config.require_approval
        self.skills_directory = skills_directory if skills_directory is not None else default_config.skills_directory
        self.thread_id: str | None = None
        self._interrupted: bool = False
        self._run_config: dict | None = None
        self.state = AgentState(input=[], next_step=None, pending_code=None, pending_language=None)
        self._usage_metrics = []
        # Per-instance REPL namespace — isolates variables from other BaseAgent instances
        self._repl_namespace: dict = {}
        # Per-instance plot capture — isolates matplotlib output from other instances
        self._plot_capture = PlotCapture()
        #TODO: add self-critic mode
        self.self_critic = False

        # Temperature: explicit kwarg not exposed yet, so spec takes precedence over default
        _temperature = (
            spec.temperature
            if spec is not None and spec.temperature is not None
            else default_config.temperature
        )

        # Initialize the LLM agent
        self.source, self.llm = get_llm(
            self.llm_model_name,
            stop_sequences=["</execute>", "</solution>"],
            source=self.source,
            base_url=self.base_url,
            api_key=self.api_key,
            config=default_config,
            temperature=_temperature,
        )

        # Initialize the resource manager (replaces ToolRegistry)
        self.resource_manager = ResourceManager()
        
        if self.use_tool_retriever:
            # Initialize the tool retriever
            self.retriever = ToolRetriever()

        # Initialize the configuration (loads all resources and generates system prompt)
        self.configure()

        # Print the LLM configuration
        print("\n" + "=" * 50)
        print("🔧 BASE AGENT CONFIGURATION")
        print("=" * 50)
        print(f"LLM: {self.llm_model_name}")
        print(f"Source: {self.source}")
        print(f"Base URL: {self.base_url}")
        tool_names = [tool.name for tool in self.resource_manager.get_all_tools()]
        print(f"Loaded Tools: {tool_names}")
        print(f"Use Tool Retriever: {self.use_tool_retriever}")
        print("=" * 50 + "\n")

    def add_tool(self, func: Callable):
        """Add a new tool to the agent's tool registry and make it available for retrieval.

        This method accepts a Python callable (function) and automatically registers it as a tool
        that the agent can use during task execution. The function's signature, type hints, and
        docstring are analyzed to generate a structured tool schema.

        Args:
            func: A callable Python function to be added as a tool. The function should follow
                 these guidelines for optimal integration:

                 **Function Signature Requirements:**
                 - Must be a callable Python function (def, lambda, or method)
                 - Should have type hints for parameters (e.g., param: str, count: int)
                 - Can have both required and optional parameters (with defaults)
                 - Parameter types can be simple (str, int, bool, float) or complex (List[Dict], etc.)

                 **Docstring Conventions:**
                 - Should include a docstring for the function description
                 - If no docstring is provided, a default description will be generated
                 - The first line of the docstring becomes the tool description

                 **Parameter Specifications:**
                 - Required parameters: Any parameter without a default value
                 - Optional parameters: Parameters with default values (e.g., param: int = 10)
                 - Parameter descriptions: Extracted from docstring if available (Google/NumPy style)
                 - Parameter types: Automatically extracted from type hints

                 **Examples:**

                 ```python
                 # Simple tool with required and optional parameters
                 def calculate_sum(numbers: List[int], multiply_by: int = 1) -> int:
                     \"\"\"Calculate the sum of numbers and optionally multiply the result.\"\"\"
                     return sum(numbers) * multiply_by

                 agent.add_tool(calculate_sum)

                 # Tool with complex types
                 def process_data(data: dict[str, Any], threshold: float = 0.5) -> list[str]:
                     \"\"\"Process data dictionary and filter by threshold.\"\"\"
                     return [k for k, v in data.items() if v > threshold]

                 agent.add_tool(process_data)

                 # Tool without type hints (still works, but less type-safe)
                 def simple_logger(message):
                     \"\"\"Log a message to console.\"\"\"
                     print(f"[LOG] {message}")

                 agent.add_tool(simple_logger)
                 ```

                 **What Happens After Adding:**
                 1. Function signature is inspected using Python introspection
                 2. API schema is generated (optionally enhanced with LLM)
                 3. A `CustomTool` Pydantic model is created with the function reference
                 4. The tool is stored in `ResourceManager`
                 5. System prompt is regenerated to include the new tool

        Raises:
            ValueError: If the function is not callable or schema generation fails
            AttributeError: If the function cannot be inspected (e.g., built-in functions)
            Exception: For any other errors during tool registration

        Returns:
            CustomTool: Pydantic model containing the function and its complete schema

        Note:
            - Built-in functions (e.g., `print`, `len`) cannot be added as they lack source code
            - Lambda functions can be added but may have limited introspection
            - The tool becomes immediately available after successful registration
            - Tool names must be unique; adding a tool with an existing name will overwrite it

        """
        try:
            # Generate CustomTool using hybrid approach (introspection + optional LLM)
            custom_tool = function_to_api_schema(func, self.llm, enhance_description=True)

            # Add the tool to the resource manager
            if hasattr(self, "resource_manager") and self.resource_manager is not None:
                self.resource_manager.add_custom_tool(custom_tool)
                print(f"Successfully registered custom tool '{custom_tool.name}' in resource manager")

            print(f"Tool '{custom_tool.name}' successfully added and ready for use")

            # Regenerate system prompt with new tool
            self.system_prompt = self._generate_system_prompt(
                self_critic=self.self_critic,
                is_retrieval=False,
            )

            return custom_tool

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback
            traceback.print_exc()
            raise

    @staticmethod
    def _parse_skill_file(path: str | Path) -> Skill:
        """Parse a SKILL.md file into a Skill object.

        The file must start with a YAML frontmatter block delimited by ``---``
        lines, followed by a markdown body with the skill instructions.

        Args:
            path: Path to the SKILL.md file.

        Returns:
            Skill object populated from the file.

        Raises:
            ValueError: If the frontmatter is missing or malformed, or if
                required fields (name, description) are absent.
        """
        import yaml

        path = Path(path)
        content = path.read_text(encoding="utf-8")

        # Split YAML frontmatter from markdown body
        if not content.startswith("---"):
            raise ValueError(f"SKILL.md at '{path}' must start with a '---' frontmatter block")

        parts = content.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"SKILL.md at '{path}' has an unclosed frontmatter block (missing closing '---')")

        try:
            frontmatter = yaml.safe_load(parts[1]) or {}
        except yaml.YAMLError as exc:
            raise ValueError(f"SKILL.md at '{path}' has malformed YAML frontmatter: {exc}") from exc

        if "name" not in frontmatter or "description" not in frontmatter:
            raise ValueError(
                f"SKILL.md at '{path}' frontmatter must include 'name' and 'description' fields"
            )

        instructions = parts[2].strip()
        return Skill(
            **{k: v for k, v in frontmatter.items() if k in Skill.model_fields},
            instructions=instructions,
            source_path=str(path),
        )

    def add_skill(self, skill_or_path: "Skill | str | Path") -> Skill:
        """Add a skill to the agent from a Skill object or a SKILL.md file path.

        Args:
            skill_or_path: A Skill object, or a path to a SKILL.md file.

        Returns:
            The registered Skill object.
        """
        if isinstance(skill_or_path, (str, Path)):
            skill = self._parse_skill_file(skill_or_path)
        else:
            skill = skill_or_path

        self.resource_manager.add_skill(skill)
        print(f"Skill '{skill.name}' successfully added")

        self.system_prompt = self._generate_system_prompt(
            self_critic=self.self_critic,
            is_retrieval=False,
        )
        return skill

    def load_skills(self, directory: str | Path) -> list[Skill]:
        """Load all SKILL.md and *.skill.md files from a directory.

        Args:
            directory: Path to a directory containing skill files.

        Returns:
            List of successfully loaded Skill objects.
        """
        directory = Path(directory)
        if not directory.is_dir():
            print(f"Warning: skills directory '{directory}' does not exist or is not a directory")
            return []

        skill_files = list(directory.glob("**/SKILL.md")) + list(directory.glob("**/*.skill.md"))
        skills = []
        for skill_file in sorted(skill_files):
            try:
                skill = self._parse_skill_file(skill_file)
                self.resource_manager.add_skill(skill)
                skills.append(skill)
                print(f"Loaded skill '{skill.name}' from '{skill_file}'")
            except (ValueError, OSError) as exc:
                print(f"Warning: skipping '{skill_file}': {exc}")

        if skills:
            self.system_prompt = self._generate_system_prompt(
                self_critic=self.self_critic,
                is_retrieval=False,
            )
            print(f"Loaded {len(skills)} skill(s) from '{directory}'")

        return skills

    def add_mcp(self, config_path: str | Path = "./BaseAgent/test/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the BaseAgent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.
        Supports both local (stdio) and remote (Streamable HTTP) MCP server transports.
        Remote servers are identified by the presence of a ``url`` field or ``type: "remote"``
        in the server configuration. Optional ``headers`` with ``${ENV_VAR}`` interpolation
        are threaded into the HTTP transport for authenticated endpoints.

        Args:
            config_path: Path to the MCP configuration YAML file containing server
                        definitions and tool specifications.

        Raises:
            FileNotFoundError: If the config file doesn't exist
            yaml.YAMLError: If the config file is malformed
            RuntimeError: If MCP server initialization fails
        """
        import asyncio
        import os
        import sys
        import types
        from pathlib import Path

        import nest_asyncio
        import yaml
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        nest_asyncio.apply()

        def _mcp_diagnostic_hints(server_params: StdioServerParameters) -> None:
            """Print diagnostic hints for common MCP server failures."""
            cmd = server_params.command if hasattr(server_params, "command") else "unknown"
            if "docker" in str(cmd).lower():
                print("  Hint: Check if Docker is running and the image is available.")
                if hasattr(server_params, "env") and server_params.env:
                    missing_env = [k for k, v in server_params.env.items() if not v]
                    if missing_env:
                        print(f"  Hint: Missing environment variables: {', '.join(missing_env)}")

        def _interpolate_env_vars(value: str) -> str:
            """Replace ${ENV_VAR} patterns in a string with environment variable values."""
            if not isinstance(value, str):
                return value
            import re
            def _replace(match):
                return os.getenv(match.group(1), "")
            return re.sub(r"\$\{([^}]+)\}", _replace, value)

        def _process_headers(raw_headers: dict) -> dict:
            """Process header values, interpolating ${ENV_VAR} patterns."""
            if not raw_headers:
                return {}
            return {k: _interpolate_env_vars(v) for k, v in raw_headers.items()}

        def _extract_tools_from_result(tools_result) -> list[dict]:
            """Extract tool dicts from an MCP list_tools result."""
            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result
            discovered = []
            for tool in tools:
                if hasattr(tool, "name"):
                    discovered.append({
                        "name": tool.name,
                        "description": tool.description,
                        "inputSchema": tool.inputSchema,
                    })
                else:
                    print(f"Warning: Skipping tool with no name attribute: {tool}")
            return discovered

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from a local (stdio) MCP server synchronously."""
            try:
                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()
                            tools_result = await session.list_tools()
                            return _extract_tools_from_result(tools_result)

                return asyncio.run(_discover_async())
            except ExceptionGroup as eg:
                error_messages = []
                for exc in eg.exceptions:
                    error_messages.append(str(exc))
                    if hasattr(exc, '__cause__') and exc.__cause__:
                        error_messages.append(f"  Caused by: {exc.__cause__}")
                error_msg = " | ".join(error_messages) if error_messages else str(eg)
                print(f"Failed to discover tools: {error_msg}")
                _mcp_diagnostic_hints(server_params)
                return []
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                _mcp_diagnostic_hints(server_params)
                return []

        def discover_remote_tools_sync(url: str, headers: dict | None = None) -> list[dict]:
            """Discover available tools from a remote (Streamable HTTP) MCP server."""
            from mcp.client.streamable_http import streamablehttp_client

            try:
                async def _discover_async():
                    async with streamablehttp_client(url, headers=headers) as (read, write, _):
                        async with ClientSession(read, write) as session:
                            await session.initialize()
                            tools_result = await session.list_tools()
                            return _extract_tools_from_result(tools_result)

                return asyncio.run(_discover_async())
            except ExceptionGroup as eg:
                error_messages = []
                for exc in eg.exceptions:
                    error_messages.append(str(exc))
                    if hasattr(exc, '__cause__') and exc.__cause__:
                        error_messages.append(f"  Caused by: {exc.__cause__}")
                error_msg = " | ".join(error_messages) if error_messages else str(eg)
                print(f"Failed to discover remote tools: {error_msg}")
                return []
            except Exception as e:
                print(f"Failed to discover remote tools: {e}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async stdio MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for stdio MCP tool execution."""
                try:
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)

                    async def async_tool_call():
                        async with stdio_client(server_params) as (reader, writer):
                            async with ClientSession(reader, writer) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        def make_remote_mcp_wrapper(url: str, tool_name: str, doc: str, headers: dict | None = None):
            """Create a synchronous wrapper for an async remote (Streamable HTTP) MCP tool call."""
            from mcp.client.streamable_http import streamablehttp_client

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for remote MCP tool execution."""
                try:
                    async def async_tool_call():
                        async with streamablehttp_client(url, headers=headers) as (read, write, _):
                            async with ClientSession(read, write) as session:
                                await session.initialize()
                                result = await session.call_tool(tool_name, kwargs)
                                content = result.content[0]
                                if hasattr(content, "json"):
                                    return content.json()
                                return content.text

                    return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"Remote MCP tool execution failed for '{tool_name}': {e}") from e

            sync_tool_wrapper.__name__ = tool_name
            sync_tool_wrapper.__doc__ = doc
            return sync_tool_wrapper

        # MCP tools will be stored in ResourceManager.collection.custom_tools

        # Load and validate configuration
        try:
            # Resolve config_path relative to this file's location if it's a relative path
            config_path_obj = Path(config_path)
            if not config_path_obj.is_absolute():
                # Get the directory where this file (base_agent.py) is located
                base_dir = Path(__file__).parent.parent
                config_path_obj = (base_dir / config_path).resolve()
            
            config_content = config_path_obj.read_text(encoding="utf-8")
            cfg: dict[str, Any] = yaml.safe_load(config_content) or {}
        except FileNotFoundError:
            raise FileNotFoundError(f"MCP config file not found: {config_path}") from None
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in MCP config: {e}") from e

        mcp_servers: dict[str, Any] = cfg.get("mcp_servers", {})
        if not mcp_servers:
            print("Warning: No MCP servers found in configuration")
            return

        # Process each MCP server configuration
        for server_name, server_meta in mcp_servers.items():
            if not server_meta.get("enabled", True):
                continue

            # Determine transport type: remote (Streamable HTTP) or local (stdio)
            is_remote = server_meta.get("type") == "remote" or "url" in server_meta

            if is_remote:
                # --- Remote server (Streamable HTTP transport) ---
                url = server_meta.get("url")
                if not url:
                    print(f"Warning: Remote server '{server_name}' has no 'url' field")
                    continue

                # Process optional auth headers with ${ENV_VAR} interpolation
                headers = _process_headers(server_meta.get("headers", {})) or None
            else:
                # --- Local server (stdio transport) ---
                cmd_list = server_meta.get("command", [])
                if not cmd_list or not isinstance(cmd_list, list):
                    print(f"Warning: Invalid command configuration for server '{server_name}'")
                    continue

                cmd, *args = cmd_list

                # Process environment variables
                env_vars = server_meta.get("env", {})
                if env_vars:
                    processed_env = {}
                    for key, value in env_vars.items():
                        if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                            var_name = value[2:-1]
                            processed_env[key] = os.getenv(var_name, "")
                        else:
                            processed_env[key] = value
                    env_vars = processed_env

            # Create module namespace for this MCP server
            mcp_module_name = f"mcp_servers.{server_name}"
            if mcp_module_name not in sys.modules:
                sys.modules[mcp_module_name] = types.ModuleType(mcp_module_name)
            server_module = sys.modules[mcp_module_name]

            tools_config = server_meta.get("tools", [])

            if not tools_config:
                try:
                    if is_remote:
                        tools_config = discover_remote_tools_sync(url, headers)
                    else:
                        server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                        tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        transport_label = "remote" if is_remote else "local"
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server ({transport_label})")
                    else:
                        print(f"Warning: No tools discovered from {server_name} MCP server")
                        continue

                except Exception as e:
                    print(f"Failed to discover tools for {server_name}: {e}")
                    continue

            # Register each tool
            for tool_meta in tools_config:
                if isinstance(tool_meta, dict) and "name" in tool_meta:
                    # Manual tool definition
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    parameters = tool_meta.get("parameters", {})
                    # For manual tools, check if each parameter has a "required" field
                    required_param_names = []
                    for param_name, param_spec in parameters.items():
                        if param_spec.get("required", False):
                            required_param_names.append(param_name)
                else:
                    # Auto-discovered tool
                    tool_name = tool_meta.get("name")
                    description = tool_meta.get("description", f"MCP tool: {tool_name}")
                    input_schema = tool_meta.get("inputSchema", {})
                    parameters = input_schema.get("properties", {})
                    # For auto-discovered tools, get required list from inputSchema top level
                    required_param_names = input_schema.get("required", [])

                if not tool_name:
                    print(f"Warning: Skipping tool with no name in {server_name}")
                    continue

                # Create wrapper function (remote vs local)
                if is_remote:
                    wrapper_function = make_remote_mcp_wrapper(url, tool_name, description, headers)
                else:
                    wrapper_function = make_mcp_wrapper(cmd, args, tool_name, description, env_vars)

                # Add to module namespace
                setattr(server_module, tool_name, wrapper_function)

                # Build parameter lists
                required_params, optional_params = [], []
                for param_name, param_spec in parameters.items():
                    param_info = {
                        "name": param_name,
                        "type": str(param_spec.get("type", "string")),
                        "description": param_spec.get("description", ""),
                        "default": param_spec.get("default", None),
                    }

                    # Check if parameter is required based on the required_param_names list
                    if param_name in required_param_names:
                        required_params.append(param_info)
                    else:
                        optional_params.append(param_info)

                # Register in resource manager as CustomTool
                from BaseAgent.resources import CustomTool, ToolParameter

                required_params = [ToolParameter(**p) for p in required_params]
                optional_params = [ToolParameter(**p) for p in optional_params]

                custom_tool = CustomTool(
                    name=tool_name,
                    description=description,
                    module=mcp_module_name,
                    function=wrapper_function,
                    required_parameters=required_params,
                    optional_parameters=optional_params,
                )
                self.resource_manager.add_custom_tool(custom_tool)

        # Regenerate system prompt with new MCP tools
        self.system_prompt = self._generate_system_prompt(
            self_critic=self.self_critic,
            is_retrieval=False,
        )


    def _generate_system_prompt(
        self,
        self_critic: bool = False,
        is_retrieval: bool = False,
    ):
        """
        Generate the system prompt based on currently selected resources.

        Args:
            self_critic: Whether to include self-critic instructions
            is_retrieval: Whether this is for retrieval (True) or initial configuration (False)

        Returns:
            The generated system prompt
            
        Note:
            Uses resources marked with selected=True in ResourceManager.
            When tool retriever is not used, all resources have selected=True by default.
            When tool retriever is used, only retrieved resources are marked selected=True.
        """
        # If the spec provides a full override, return it directly
        if self.spec is not None and self.spec.system_prompt_override is not None:
            return self.spec.system_prompt_override

        # Base prompt
        prompt_modifier = get_base_prompt_template(self_critic=self_critic)

        # Role description: from spec, or fall back to the default
        role_description = (
            self.spec.role
            if self.spec is not None
            else _DEFAULT_ROLE_DESCRIPTION
        )
        prompt_format_dict = {"role_description": role_description}

        # Add custom resources section from resource manager
        custom_tools = self.resource_manager.collection.custom_tools
        custom_data = self.resource_manager.collection.custom_data
        custom_software = self.resource_manager.collection.custom_software
        
        has_any_custom = any([custom_tools, custom_data, custom_software])
        if has_any_custom:
            prompt_modifier += _PROMPT_CUSTOM_RESOURCES_SECTION
            if custom_tools:
                prompt_modifier += _CUSTOM_TOOLS_SECTION
                custom_tools_descriptions = [
                    f"🔧 {tool.name} (from {tool.module}): {tool.description}"
                    for tool in custom_tools
                ]
                prompt_format_dict["custom_tools"] = "\n".join(custom_tools_descriptions)
            if custom_data:
                prompt_modifier += _CUSTOM_DATA_SECTION
                custom_data_descriptions = [
                    f"📊 {data.name}: {data.description}"
                    for data in custom_data
                ]
                prompt_format_dict["custom_data"] = "\n".join(custom_data_descriptions)
            if custom_software:
                prompt_modifier += _CUSTOM_SOFTWARE_SECTION
                custom_software_descriptions = [
                    f"⚙️ {software.name}: {software.description}"
                    for software in custom_software
                ]
                prompt_format_dict["custom_software"] = "\n".join(custom_software_descriptions)

        # Add skills section
        selected_skills = self.resource_manager.get_selected_skills()
        if selected_skills:
            skill_parts = []
            for skill in selected_skills:
                skill_parts.append(_SKILL_ENTRY_TEMPLATE.format(
                    skill_name=skill.name,
                    skill_description=skill.description,
                    skill_instructions=skill.instructions,
                ))
            prompt_modifier += _SKILLS_SECTION.format(skills_content="\n".join(skill_parts))

        # Add environment resources section
        prompt_modifier += get_environment_resources_section(is_retrieval=is_retrieval)

        # Format tool descriptions for prompt
        # Use currently selected tools from ResourceManager
        tool_desc = defaultdict(list)
        
        for tool in self.resource_manager.get_selected_tools():
            # Skip run_python_repl from the tool descriptions
            if tool.name == "run_python_repl":
                continue
            
            # Convert Tool model to dict using Pydantic's model_dump
            tool_dict = tool.model_dump(exclude={'id', 'selected'})
            tool_desc[tool.module].append(tool_dict)
        
        prompt_format_dict["tool_desc"] = ResourceManager.format_tools_by_module(tool_desc)
        
        # Format data lake descriptions
        # Use currently selected data from ResourceManager
        data_lake_descriptions = []
        for data_item in self.resource_manager.get_selected_data():
            name = data_item.filename if hasattr(data_item, 'filename') else data_item.name
            data_lake_descriptions.append(f"{name}:\n {data_item.description}")
        
        data_lake_content = "\n".join(data_lake_descriptions) if data_lake_descriptions else None
        prompt_format_dict["data_lake_path"] = self.path + "/data_lake"
        prompt_format_dict["data_lake_content"] = data_lake_content
        
        # Format library descriptions
        # Use currently selected libraries from ResourceManager
        library_descriptions = []
        for library in self.resource_manager.get_selected_libraries():
            library_descriptions.append(f"{library.name}:\n {library.description}")
        
        library_content = "\n".join(library_descriptions) if library_descriptions else None
        prompt_format_dict["library_content"] = library_content

        return prompt_modifier.format(**prompt_format_dict)


    def _setup_data_lake(self):
        """
        Set up the data lake content by loading items into ResourceManager.
        """
        # Load data lake items from env_desc into resource manager
        for data_item in data_lake_items:
            # Set the path if not already set
            if not data_item.path and self.path:
                data_item.path = f"{self.path}/data_lake/{data_item.filename}"
            self.resource_manager.add_data_item(data_item)


    def _setup_library(self):
        """
        Set up the library content by loading libraries into ResourceManager.
        """
        # Load libraries from env_desc into resource manager
        for library in libraries:
            self.resource_manager.add_library(library)

    
    def configure(self, self_critic=False, test_time_scale_round=0):
        """
        Configure the agent with the initial system prompt and workflow.
        
        This method loads all built-in resources (tools, data, libraries) and generates
        the system prompt. It should be called once during initialization.

        Args:
            self_critic: Whether to enable self-critic mode
            test_time_scale_round: Number of rounds for test time scaling
        """
        # Store self_critic for later use
        self.self_critic = self_critic

        # Load all built-in resources into resource manager
        self.resource_manager.load_builtin_tools()  # Load tools from tool_description
        self._setup_data_lake()  # Load data lake items from env_desc
        self._setup_library()  # Load libraries from env_desc

        # Auto-load skills from configured directory
        if self.skills_directory:
            self.load_skills(self.skills_directory)

        # Apply AgentSpec resource filters (must happen after resources are loaded)
        if self.spec is not None:
            if self.spec.tool_names is not None:
                self.resource_manager.select_tools_by_names(self.spec.tool_names)
            if self.spec.skill_names is not None:
                self.resource_manager.select_skills_by_names(self.spec.skill_names)

        # Generate the system prompt (will be built automatically from ResourceManager)
        self.system_prompt = self._generate_system_prompt(
            self_critic=self_critic,
            is_retrieval=False,
        )

        # Build NodeExecutor and wire up the graph
        from BaseAgent.nodes import NodeExecutor
        self.node_executor = NodeExecutor(self)

        # Bind execute_self_critic with the captured test_time_scale_round
        def _execute_self_critic(state: AgentState) -> AgentState:
            return self.node_executor.execute_self_critic(state, test_time_scale_round)

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes: retrieve runs first (no-op when use_tool_retriever=False)
        workflow.add_node("retrieve", self.node_executor.retrieve)
        workflow.add_node("generate", self.node_executor.generate)
        workflow.add_node("execute", self.node_executor.execute)

        if self_critic:
            workflow.add_node("self_critic", _execute_self_critic)
            workflow.add_conditional_edges(
                "self_critic",
                self.node_executor.routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )

        # Where "end" from generate routes depends only on self_critic
        end_destination = "self_critic" if self_critic else END

        # Approval gate node (only added when approval policy requires it)
        require_approval = self.require_approval
        if require_approval != "never":
            workflow.add_node("approval_gate", self.node_executor.approval_gate)

        if require_approval == "always":
            # All execute blocks pass through approval_gate first
            workflow.add_conditional_edges(
                "generate",
                self.node_executor.routing_function,
                path_map={"execute": "approval_gate", "generate": "generate", "end": end_destination},
            )
            workflow.add_conditional_edges(
                "approval_gate",
                self.node_executor.routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        elif require_approval == "dangerous_only":
            # Bash/R goes through approval_gate; Python goes directly to execute
            workflow.add_conditional_edges(
                "generate",
                self.node_executor.routing_function_with_approval,
                path_map={
                    "execute": "execute",
                    "approval_gate": "approval_gate",
                    "generate": "generate",
                    "end": end_destination,
                },
            )
            workflow.add_conditional_edges(
                "approval_gate",
                self.node_executor.routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        else:  # "never" — direct routing, no approval gate
            workflow.add_conditional_edges(
                "generate",
                self.node_executor.routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": end_destination},
            )

        workflow.add_edge("execute", "generate")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge(START, "retrieve")

        # Compile with persistent checkpointer
        self.checkpointer = self._create_checkpointer()
        self.app = workflow.compile(checkpointer=self.checkpointer)

    def _create_checkpointer(self):
        """Create the appropriate checkpointer based on config.

        Uses SqliteSaver when available. Falls back to InMemorySaver with a
        warning if langgraph-checkpoint-sqlite is not installed and a file path
        was requested.
        """
        db_path = self.checkpoint_db_path
        if _HAS_SQLITE_SAVER:
            return SqliteSaver.from_conn_string(db_path)
        else:
            from langgraph.checkpoint.memory import MemorySaver
            if db_path != ":memory:":
                warnings.warn(
                    "langgraph-checkpoint-sqlite is not installed; falling back to "
                    "in-memory checkpointer. State will NOT persist across sessions. "
                    "Install with: pip install langgraph-checkpoint-sqlite",
                    stacklevel=3,
                )
            return MemorySaver()

    def close(self):
        """Release checkpointer resources (closes the SQLite connection if open)."""
        if hasattr(self, "checkpointer") and hasattr(self.checkpointer, "conn"):
            try:
                self.checkpointer.conn.close()
            except Exception:
                pass

    def __del__(self):
        self.close()

    @property
    def is_interrupted(self) -> bool:
        """True when the agent is paused and awaiting human approval.

        Check this after calling ``run()`` to determine whether a code block
        is pending review. If True, call ``resume()`` to approve or
        ``reject(feedback)`` to decline.
        """
        return self._interrupted

    def _select_resources_for_prompt(self, prompt: str) -> None:
        """
        Use tool retriever to select relevant resources for the given prompt.
        
        This method performs retrieval on all available resources and marks only
        the relevant ones as selected (selected=True) in the ResourceManager.

        Args:
            prompt: The user's query
        """
        # Gather all available resources from resource manager for retrieval
        # Format all resources as dictionaries for prompt-based retrieval
        
        # 1. Tools
        all_tools = [
            {"name": tool.name, "description": tool.description}
            for tool in self.resource_manager.get_all_tools()
        ]

        # 2. Data
        all_data = [
            {"name": (item.filename if hasattr(item, 'filename') else item.name), "description": item.description}
            for item in self.resource_manager.get_all_data()
        ]

        # 3. Libraries
        all_libraries = [
            {"name": lib.name, "description": lib.description}
            for lib in self.resource_manager.get_all_libraries()
        ]

        # 4. Skills (only auto-trigger skills are candidates for retrieval)
        all_skills = [
            {"name": skill.name, "description": skill.description}
            for skill in self.resource_manager.get_all_skills()
            if skill.trigger == "auto"
        ]

        # Use retrieval to get relevant resources
        all_resources = {
            "all_tools": all_tools,
            "all_data": all_data,
            "all_libraries": all_libraries,
            "all_skills": all_skills,
        }

        # Use prompt-based retrieval with the agent's LLM
        print("Conducting prompt-based retrieval to select relevant resources...")
        selected_resources = self.retriever.prompt_based_retrieval(prompt, all_resources, llm=self.llm)

        # Update selection state in ResourceManager
        selected_tool_names = [tool["name"] for tool in selected_resources["selected_tools"]]
        selected_data_names = [data["name"] for data in selected_resources["selected_data"]]
        selected_lib_names = [lib["name"] for lib in selected_resources["selected_libraries"]]
        selected_skill_names = [skill["name"] for skill in selected_resources.get("selected_skills", [])]

        self.resource_manager.select_tools_by_names(selected_tool_names)
        self.resource_manager.select_data_by_names(selected_data_names)
        self.resource_manager.select_libraries_by_names(selected_lib_names)
        self.resource_manager.select_skills_by_names(selected_skill_names)

        print(
            f"Selected {len(selected_tool_names)} tools, {len(selected_data_names)} data items, "
            f"{len(selected_lib_names)} libraries, {len(selected_skill_names)} skills"
        )


    def run(self, prompt: str, thread_id: str | None = None):
        """Execute the agent with the given prompt.

        Args:
            prompt: The user's query.
            thread_id: Optional session identifier. When provided, the same
                thread_id can be reused across process restarts to resume a
                persisted conversation. When omitted a fresh UUID is generated.

        Returns:
            ``(log, content)`` where *content* is the final answer string when
            the agent completes normally, or the interrupt payload dict when
            ``require_approval != "never"`` and a code block needs review.
            Check :attr:`is_interrupted` to distinguish the two cases.
        """
        self.critic_count = 0
        self.user_task = prompt

        tid = thread_id or str(uuid.uuid4())
        self.thread_id = tid
        inputs = {
            "input": [HumanMessage(content=prompt)],
            "next_step": None,
            "pending_code": None,
            "pending_language": None,
        }
        config = {"recursion_limit": 500, "configurable": {"thread_id": tid}}
        self.log = []
        self._run_config = config

        last_msg = None
        final_state = None
        for state in self.app.stream(inputs, stream_mode="values", config=config):
            last_msg = state["input"][-1]
            out = pretty_print(last_msg)
            self.log.append(out)
            final_state = state

        self._conversation_state = final_state

        # Detect interrupt: check for pending interrupts in the graph state
        graph_state = self.app.get_state(config)
        if graph_state.tasks and any(
            hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
        ):
            self._interrupted = True
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return self.log, task.interrupts[0].value
            # Fallback (should not be reached)
            return self.log, final_state.get("pending_code", "")

        self._interrupted = False
        return self.log, last_msg.content if last_msg else ""

    def resume(self):
        """Approve the pending code block and continue execution.

        Call this after :meth:`run` returned with :attr:`is_interrupted` ``True``.

        Returns:
            ``(log, content)`` — same shape as :meth:`run`. May itself be
            interrupted again if the agent generates another code block that
            requires approval.

        Raises:
            RuntimeError: If the agent is not currently interrupted.
        """
        if not self.is_interrupted:
            raise RuntimeError("Cannot resume: agent is not in an interrupted state")

        config = self._run_config
        last_msg = None
        final_state = None

        for state in self.app.stream(Command(resume=True), stream_mode="values", config=config):
            last_msg = state["input"][-1]
            out = pretty_print(last_msg)
            self.log.append(out)
            final_state = state

        self._conversation_state = final_state

        graph_state = self.app.get_state(config)
        if graph_state.tasks and any(
            hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
        ):
            self._interrupted = True
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return self.log, task.interrupts[0].value
            return self.log, final_state.get("pending_code", "")

        self._interrupted = False
        return self.log, last_msg.content if last_msg else ""

    def reject(self, feedback: str = "User rejected this code. Try a different approach."):
        """Reject the pending code block and provide feedback to the agent.

        The *feedback* string is passed as the ``Command(resume=...)`` value to
        the ``approval_gate`` node, which injects it as a :class:`HumanMessage`
        and routes back to the generate node so the agent can try again.

        Args:
            feedback: Explanation of why the code was rejected. The agent uses
                this to generate an alternative approach.

        Returns:
            ``(log, content)`` — same shape as :meth:`run`.

        Raises:
            RuntimeError: If the agent is not currently interrupted.
        """
        if not self.is_interrupted:
            raise RuntimeError("Cannot reject: agent is not in an interrupted state")

        config = self._run_config
        last_msg = None
        final_state = None

        for state in self.app.stream(
            Command(resume={"approved": False, "feedback": feedback}),
            stream_mode="values",
            config=config,
        ):
            last_msg = state["input"][-1]
            out = pretty_print(last_msg)
            self.log.append(out)
            final_state = state

        self._conversation_state = final_state

        graph_state = self.app.get_state(config)
        if graph_state.tasks and any(
            hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
        ):
            self._interrupted = True
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    return self.log, task.interrupts[0].value
            return self.log, final_state.get("pending_code", "")

        self._interrupted = False
        return self.log, last_msg.content if last_msg else ""

    async def run_stream(
        self,
        prompt: str,
        event_types: set | None = None,
        thread_id: str | None = None,
    ) -> AsyncIterator["AgentEvent"]:
        """Stream typed AgentEvent objects as the agent executes.

        Uses LangGraph's ``astream_events`` (v2) under the hood, mapping each
        node lifecycle event to a typed AgentEvent.  The caller can optionally
        filter to a subset of event types.

        Args:
            prompt: The user task to execute.
            event_types: Optional set of EventType values to include.  When
                ``None`` (default), all events are yielded.

        Yields:
            AgentEvent objects in emission order.

        Example::

            async for event in agent.run_stream("What is 2+2?"):
                print(event.event_type, event.content[:80])

            # Filter to only final answers and errors:
            from BaseAgent.events import EventType
            async for event in agent.run_stream(
                "Analyse data.csv",
                event_types={EventType.FINAL_ANSWER, EventType.ERROR},
            ):
                print(event.to_json())
        """
        from BaseAgent.events import AgentEvent, EventType  # local import avoids top-level cycle risk

        self.critic_count = 0
        self.user_task = prompt

        tid = thread_id or str(uuid.uuid4())
        self.thread_id = tid
        inputs = {
            "input": [HumanMessage(content=prompt)],
            "next_step": None,
            "pending_code": None,
            "pending_language": None,
        }
        config = {"recursion_limit": 500, "configurable": {"thread_id": tid}}
        self._run_config = config

        async for raw_event in self.app.astream_events(inputs, config=config, version="v2"):
            agent_event = self._map_langgraph_event(raw_event)
            if agent_event is None:
                continue
            if event_types is not None and agent_event.event_type not in event_types:
                continue
            yield agent_event

        # Detect interrupt after stream ends
        graph_state = await self.app.aget_state(config)
        if graph_state.tasks and any(
            hasattr(t, "interrupts") and t.interrupts for t in graph_state.tasks
        ):
            self._interrupted = True
            for task in graph_state.tasks:
                if hasattr(task, "interrupts") and task.interrupts:
                    payload = task.interrupts[0].value
                    event = AgentEvent(
                        event_type=EventType.APPROVAL_REQUIRED,
                        content=payload.get("code", "") if isinstance(payload, dict) else str(payload),
                        node_name="approval_gate",
                        metadata={
                            "language": payload.get("language", "python") if isinstance(payload, dict) else "python",
                            "message": payload.get("message", "") if isinstance(payload, dict) else "",
                        },
                    )
                    if event_types is None or EventType.APPROVAL_REQUIRED in event_types:
                        yield event
                    return
        else:
            self._interrupted = False

    def _map_langgraph_event(self, event: dict) -> "AgentEvent | None":
        """Map a raw LangGraph v2 event dict to an AgentEvent.

        Processes ``on_chain_start`` and ``on_chain_end`` events for the three
        core graph nodes (``retrieve``, ``generate``, ``execute``).  All other
        events return ``None`` and are silently dropped.

        Node → event mapping:
        - ``retrieve`` start  → RETRIEVAL_START
        - ``retrieve`` end    → RETRIEVAL_COMPLETE
        - ``generate`` end    → THINKING | FINAL_ANSWER | ERROR (based on tag)
        - ``execute`` start   → CODE_EXECUTING (with code content)
        - ``execute`` end     → CODE_RESULT (with observation content)
        """
        from BaseAgent.events import AgentEvent, EventType
        from langchain_core.messages import AIMessage

        event_name = event.get("event", "")
        node_name = event.get("metadata", {}).get("langgraph_node", "")

        if node_name not in {"retrieve", "generate", "execute", "self_critic"}:
            return None

        if event_name == "on_chain_start":
            if node_name == "retrieve":
                return AgentEvent(
                    event_type=EventType.RETRIEVAL_START,
                    content="Starting resource retrieval",
                    node_name=node_name,
                )

            if node_name == "execute":
                # Parse the code from the state that was passed into this node
                state = event.get("data", {}).get("input", {})
                if isinstance(state, dict):
                    messages = state.get("input", [])
                    for msg in reversed(messages):
                        if isinstance(msg, AIMessage):
                            code_match = re.search(
                                r"<execute>(.*?)</execute>", msg.content, re.DOTALL
                            )
                            if code_match:
                                return AgentEvent(
                                    event_type=EventType.CODE_EXECUTING,
                                    content=code_match.group(1).strip(),
                                    node_name=node_name,
                                )
                return None

        elif event_name == "on_chain_end":
            output = event.get("data", {}).get("output", {})
            if not isinstance(output, dict):
                return None
            messages = output.get("input", [])
            if not messages:
                return None

            if node_name == "retrieve":
                return AgentEvent(
                    event_type=EventType.RETRIEVAL_COMPLETE,
                    content="Resource retrieval complete",
                    node_name=node_name,
                )

            if node_name == "generate":
                # The generate node appends the raw LLM response as an AIMessage.
                # Find the most-recently appended AIMessage and parse its tags.
                for msg in reversed(messages):
                    if not isinstance(msg, AIMessage):
                        continue
                    content = msg.content

                    answer_match = re.search(
                        r"<solution>(.*?)</solution>", content, re.DOTALL
                    )
                    if answer_match:
                        return AgentEvent(
                            event_type=EventType.FINAL_ANSWER,
                            content=answer_match.group(1).strip(),
                            node_name=node_name,
                        )

                    think_match = re.search(
                        r"<think>(.*?)</think>", content, re.DOTALL
                    )
                    if think_match:
                        return AgentEvent(
                            event_type=EventType.THINKING,
                            content=think_match.group(1).strip(),
                            node_name=node_name,
                        )

                    # Parsing-error messages injected by the node itself
                    if "terminated due to" in content or "There are no tags" in content:
                        return AgentEvent(
                            event_type=EventType.ERROR,
                            content=content,
                            node_name=node_name,
                        )
                    break

            if node_name == "execute":
                # The execute node appends <observation>...</observation> as an AIMessage
                for msg in reversed(messages):
                    if not isinstance(msg, AIMessage):
                        continue
                    obs_match = re.search(
                        r"<observation>(.*?)</observation>", msg.content, re.DOTALL
                    )
                    if obs_match:
                        return AgentEvent(
                            event_type=EventType.CODE_RESULT,
                            content=obs_match.group(1).strip(),
                            node_name=node_name,
                        )
                    break

        return None

    def _clear_execution_plots(self):
        """Clear the per-instance plot capture buffer before a new execution."""
        try:
            self._plot_capture.clear()
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")


    def _inject_custom_functions_to_repl(self):
        """Inject custom tools into the per-instance REPL namespace.

        Makes custom tools added via ``add_tool()`` / ``add_mcp()`` callable
        inside ``<execute>`` blocks.  Uses ``self._repl_namespace`` so that
        injected functions are isolated to this agent instance.
        """
        custom_tools = self.resource_manager.collection.custom_tools
        custom_functions = {
            tool.name: tool.function
            for tool in custom_tools
            if tool.function is not None
        }
        inject_custom_functions_to_repl(custom_functions, namespace=self._repl_namespace)


    def _record_usage(self, usage):
        if usage is None:
            return
        self._usage_metrics.append(usage)