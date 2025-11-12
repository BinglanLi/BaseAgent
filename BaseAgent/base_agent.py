import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict, Literal, Callable
from dotenv import load_dotenv

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from BaseAgent.llm import SourceType, get_llm, extract_usage_metrics
from BaseAgent.prompts import (
    get_base_prompt_template,
    get_environment_resources_section,
    get_feedback_prompt,
    _PROMPT_CUSTOM_RESOURCES_SECTION,
    _CUSTOM_TOOLS_SECTION,
    _CUSTOM_DATA_SECTION,
    _CUSTOM_SOFTWARE_SECTION,
)
from BaseAgent.config import default_config
from BaseAgent.resource_manager import ResourceManager
from BaseAgent.retriever import ToolRetriever
from BaseAgent.tools.support_tools import run_python_repl
from BaseAgent.env_desc import data_lake_items, libraries
from BaseAgent.utils import (
    inject_custom_functions_to_repl,
    pretty_print,
    run_bash_script,
    run_r_code,
    run_with_timeout,
    function_to_api_schema,
)

if os.path.exists(".env"):
    load_dotenv(".env", override=False)
    print("Loaded environment variables from .env")

class AgentState(TypedDict):
    input: list[BaseMessage]
    next_step: str | None

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
    ):
        """
        Args:
            path: The path to the data.
            llm: The LLM to use.
            source: The source of the LLM, e.g., "OpenAI", "Anthropic", "Ollama", "Gemini", "Bedrock", "Groq", "Custom".
            timeout_seconds: The timeout in seconds.
            base_url: The base URL of the LLM.
            api_key: The API key of the LLM.
        """
        self.path = path if path is not None else default_config.path
        self.llm_model_name = llm if llm is not None else default_config.llm
        self.source = source if source is not None else default_config.source
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else default_config.timeout_seconds
        self.base_url = base_url if base_url is not None else default_config.base_url
        self.api_key = api_key if api_key is not None else default_config.api_key
        self.use_tool_retriever = use_tool_retriever if use_tool_retriever is not None else default_config.use_tool_retriever
        self.state = AgentState(input=[], next_step=None)
        self._usage_metrics = []
        #TODO: add self-critic mode
        self.self_critic = False

        # Initialize the LLM agent
        self.source, self.llm = get_llm(
            self.llm_model_name,
            stop_sequences=["</execute>", "</solution>"],
            source=self.source,
            base_url=self.base_url,
            api_key=self.api_key,
            config=default_config,
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

    def add_tool(self, func: Callable) -> None:
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

        except Exception as e:
            print(f"Error adding tool: {e}")
            import traceback
            traceback.print_exc()
            raise

    def add_mcp(self, config_path: str | Path = "./BaseAgent/test/mcp_config.yaml") -> None:
        """
        Add MCP (Model Context Protocol) tools from configuration file.

        This method dynamically registers MCP server tools as callable functions within
        the BaseAgent system. Each MCP server is loaded as an independent module
        with its tools exposed as synchronous wrapper functions.

        Supports both manual tool definitions and automatic tool discovery from MCP servers.

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

        def discover_mcp_tools_sync(server_params: StdioServerParameters) -> list[dict]:
            """Discover available tools from MCP server synchronously."""
            try:

                async def _discover_async():
                    async with stdio_client(server_params) as (reader, writer):
                        async with ClientSession(reader, writer) as session:
                            await session.initialize()

                            # Get available tools
                            tools_result = await session.list_tools()
                            tools = tools_result.tools if hasattr(tools_result, "tools") else tools_result

                            discovered_tools = []
                            for tool in tools:
                                if hasattr(tool, "name"):
                                    discovered_tools.append(
                                        {
                                            "name": tool.name,
                                            "description": tool.description,
                                            "inputSchema": tool.inputSchema,
                                        }
                                    )
                                else:
                                    print(f"Warning: Skipping tool with no name attribute: {tool}")

                            return discovered_tools

                return asyncio.run(_discover_async())
            except ExceptionGroup as eg:
                # Handle ExceptionGroup (TaskGroup errors) - Python 3.11+
                error_messages = []
                for exc in eg.exceptions:
                    error_messages.append(str(exc))
                    # Try to get more details from the exception
                    if hasattr(exc, '__cause__') and exc.__cause__:
                        error_messages.append(f"  Caused by: {exc.__cause__}")
                
                error_msg = " | ".join(error_messages) if error_messages else str(eg)
                print(f"Failed to discover tools: {error_msg}")
                
                # Provide diagnostic hints
                cmd = server_params.command if hasattr(server_params, 'command') else 'unknown'
                if 'docker' in str(cmd).lower():
                    print("  Hint: Check if Docker is running and the image is available.")
                    if hasattr(server_params, 'env') and server_params.env:
                        missing_env = [k for k, v in server_params.env.items() if not v]
                        if missing_env:
                            print(f"  Hint: Missing environment variables: {', '.join(missing_env)}")
                
                return []
            except Exception as e:
                print(f"Failed to discover tools: {e}")
                # Provide diagnostic hints for common issues
                cmd = server_params.command if hasattr(server_params, 'command') else 'unknown'
                if 'docker' in str(cmd).lower():
                    print("  Hint: Check if Docker is running and the image is available.")
                    if hasattr(server_params, 'env') and server_params.env:
                        missing_env = [k for k, v in server_params.env.items() if not v]
                        if missing_env:
                            print(f"  Hint: Missing environment variables: {', '.join(missing_env)}")
                return []

        def make_mcp_wrapper(cmd: str, args: list[str], tool_name: str, doc: str, env_vars: dict = None):
            """Create a synchronous wrapper for an async MCP tool call."""

            def sync_tool_wrapper(**kwargs):
                """Synchronous wrapper for MCP tool execution."""
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

                    try:
                        loop = asyncio.get_running_loop()
                        return loop.create_task(async_tool_call())
                    except RuntimeError:
                        return asyncio.run(async_tool_call())

                except Exception as e:
                    raise RuntimeError(f"MCP tool execution failed for '{tool_name}': {e}") from e

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

            # Validate command configuration
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
                    server_params = StdioServerParameters(command=cmd, args=args, env=env_vars)
                    tools_config = discover_mcp_tools_sync(server_params)

                    if tools_config:
                        print(f"Discovered {len(tools_config)} tools from {server_name} MCP server")
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

                # Create wrapper function
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

                # Create tool schema
                tool_schema = {
                    "name": tool_name,
                    "description": description,
                    "parameters": parameters,
                    "required_parameters": required_params,
                    "optional_parameters": optional_params,
                    "module": mcp_module_name,
                    "fn": wrapper_function,
                }

                # Register in resource manager as CustomTool
                from BaseAgent.resources import CustomTool, ToolParameter
                
                # Convert to CustomTool model (includes function attribute)
                required_params = [ToolParameter(**p) for p in required_params]
                optional_params = [ToolParameter(**p) for p in optional_params]
                
                custom_tool = CustomTool(
                    name=tool_schema["name"],
                    description=tool_schema["description"],
                    module=tool_schema["module"],
                    function=wrapper_function,  # Store the MCP wrapper function
                    required_parameters=required_params,
                    optional_parameters=optional_params
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
        # Base prompt
        prompt_modifier = get_base_prompt_template(self_critic=self_critic)
        prompt_format_dict = {}

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

        # Generate the system prompt (will be built automatically from ResourceManager)
        self.system_prompt = self._generate_system_prompt(
            self_critic=self_critic,
            is_retrieval=False,
        )

        # Define the nodes
        def generate(state: AgentState) -> AgentState:
            # Add the system prompt to the input to LLM 
            input = [SystemMessage(content=self.system_prompt)] + state["input"]
            output = self.llm.invoke(input)

            usage_metrics = extract_usage_metrics(self.source, output, model=getattr(self.llm, "model_name", None))
            if usage_metrics is not None:
                self._record_usage(usage_metrics)

            # Parse the response
            resp = str(output.content)

            # Check for incomplete tags and fix them
            if "<execute>" in resp and "</execute>" not in resp:
                resp += "</execute>"
            if "<solution>" in resp and "</solution>" not in resp:
                resp += "</solution>"
            if "<think>" in resp and "</think>" not in resp:
                resp += "</think>"

            # Parse the response
            think_match = re.search(r"<think>(.*?)</think>", resp, re.DOTALL)
            execute_match = re.search(r"<execute>(.*?)</execute>", resp, re.DOTALL)
            answer_match = re.search(r"<solution>(.*?)</solution>", resp, re.DOTALL)

            # Add the message to the state before checking for errors
            state["input"].append(AIMessage(content=resp.strip()))

            if answer_match:
                state["next_step"] = "end"
            elif execute_match:
                state["next_step"] = "execute"
            elif think_match:
                state["next_step"] = "generate"
            else:
                # Response doesn't contain required tags, will retry
                print("parsing error...")

                error_count = sum(
                    1 for _ in state["input"] if isinstance(_, AIMessage) and "There are no tags" in _.content
                )

                if error_count >= 2:
                    # If we've already tried to correct the model twice, just end the conversation
                    print("Detected repeated parsing errors, ending conversation")
                    state["next_step"] = "end"
                    # Add a final message explaining the termination
                    state["input"].append(
                        AIMessage(
                            content="Execution terminated due to repeated parsing errors. Please check your input prompt and try again."
                        )
                    )
                else:
                    # Try to correct it
                    state["input"].append(
                        HumanMessage(
                            content="Each response must include thinking process followed by either <execute> or <solution> tag. But there are no tags in the current response. Please follow the instruction, fix and regenerate the response again."
                        )
                    )
                    state["next_step"] = "generate"
            return state

        def execute(state: AgentState) -> AgentState:
            last_resp = state["input"][-1].content
            # Only add the closing tag if it's not already there
            if "<execute>" in last_resp and "</execute>" not in last_resp:
                last_resp += "</execute>"

            execute_match = re.search(r"<execute>(.*?)</execute>", last_resp, re.DOTALL)
            if execute_match:
                code = execute_match.group(1)

                # Set timeout duration (10 minutes = 600 seconds)
                timeout = self.timeout_seconds

                # Check if the code is R code
                if (
                    code.strip().startswith("#!R")
                    or code.strip().startswith("# R code")
                    or code.strip().startswith("# R script")
                ):
                    # Remove the R marker and run as R code
                    r_code = re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
                    result = run_with_timeout(run_r_code, [r_code], timeout=timeout)
                # Check if the code is a Bash script or CLI command
                elif (
                    code.strip().startswith("#!BASH")
                    or code.strip().startswith("# Bash script")
                    or code.strip().startswith("#!CLI")
                ):
                    # Handle both Bash scripts and CLI commands with the same function
                    if code.strip().startswith("#!CLI"):
                        # For CLI commands, extract the command and run it as a simple bash script
                        cli_command = re.sub(r"^#!CLI", "", code, count=1).strip()
                        # Remove any newlines to ensure it's a single command
                        cli_command = cli_command.replace("\n", " ")
                        result = run_with_timeout(run_bash_script, [cli_command], timeout=timeout)
                    else:
                        # For Bash scripts, remove the marker and run as a bash script
                        bash_script = re.sub(r"^#!BASH|^# Bash script", "", code, count=1).strip()
                        result = run_with_timeout(run_bash_script, [bash_script], timeout=timeout)
                # Otherwise, run as Python code
                else:
                    # Clear any previous plots before execution
                    self._clear_execution_plots()

                    # Inject custom functions into the Python execution environment
                    self._inject_custom_functions_to_repl()
                    result = run_with_timeout(run_python_repl, [code], timeout=timeout)

                    # Plots are now captured directly in the execution entry above

                if len(result) > 10000:
                    result = (
                        "The output is too long to be added to context. Here are the first 10K characters...\n"
                        + result[:10000]
                    )

                # Store the execution result with the triggering message
                if not hasattr(self, "_execution_results"):
                    self._execution_results = []

                # Get any plots that were generated during this execution
                execution_plots = []
                try:
                    from BaseAgent.tools.support_tools import get_captured_plots

                    current_plots = get_captured_plots()
                    execution_plots = current_plots.copy()
                except Exception as e:
                    print(f"Warning: Could not capture plots from execution: {e}")
                    execution_plots = []

                # Store the execution result with metadata
                execution_entry = {
                    "triggering_message": last_resp,  # The AI message that contained <execute>
                    "images": execution_plots,  # Base64 encoded images from this execution
                    "timestamp": datetime.now().isoformat(),
                }
                self._execution_results.append(execution_entry)

                observation = f"\n<observation>{result}</observation>"
                state["input"].append(AIMessage(content=observation.strip()))

            return state

        def routing_function(
            state: AgentState,
        ) -> Literal["execute", "generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "execute":
                return "execute"
            elif next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def routing_function_self_critic(
            state: AgentState,
        ) -> Literal["generate", "end"]:
            next_step = state.get("next_step")
            if next_step == "generate":
                return "generate"
            elif next_step == "end":
                return "end"
            else:
                raise ValueError(f"Unexpected next_step: {next_step}")

        def execute_self_critic(state: AgentState) -> AgentState:
            if self.critic_count < test_time_scale_round:
                # Generate feedback based on message history
                input = state["input"]
                feedback_prompt = get_feedback_prompt(self.user_task)
                feedback = self.llm.invoke(input + [HumanMessage(content=feedback_prompt)])

                usage_metrics = extract_usage_metrics(self.source, feedback, model=getattr(self.llm, "model_name", None))
                if usage_metrics is not None:
                    self._record_usage(usage_metrics)

                # Add feedback as a new message
                state["input"].append(
                    HumanMessage(
                        content=f"Wait... this is not enough to solve the task. Here are some feedbacks for improvement:\n{feedback.content}"
                    )
                )
                self.critic_count += 1
                state["next_step"] = "generate"
            else:
                state["next_step"] = "end"

            return state

        # Create the workflow
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("generate", generate)
        workflow.add_node("execute", execute)

        if self_critic:
            workflow.add_node("self_critic", execute_self_critic)
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={
                    "execute": "execute",
                    "generate": "generate",
                    "end": "self_critic",
                },
            )
            workflow.add_conditional_edges(
                "self_critic",
                routing_function_self_critic,
                path_map={"generate": "generate", "end": END},
            )
        else:
            # Add conditional edges
            workflow.add_conditional_edges(
                "generate",
                routing_function,
                path_map={"execute": "execute", "generate": "generate", "end": END},
            )
        workflow.add_edge("execute", "generate")
        workflow.add_edge(START, "generate")

        # Compile the workflow
        self.app = workflow.compile()
        self.checkpointer = MemorySaver()
        self.app.checkpointer = self.checkpointer
        # display(Image(self.app.get_graph().draw_mermaid_png()))

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

        # Use retrieval to get relevant resources
        all_resources = {
            "all_tools": all_tools,
            "all_data": all_data,
            "all_libraries": all_libraries,
        }

        # Use prompt-based retrieval with the agent's LLM
        print("Conducting prompt-based retrieval to select relevant resources...")
        selected_resources = self.retriever.prompt_based_retrieval(prompt, all_resources, llm=self.llm)
        
        # Update selection state in ResourceManager
        selected_tool_names = [tool["name"] for tool in selected_resources["selected_tools"]]
        selected_data_names = [data["name"] for data in selected_resources["selected_data"]]
        selected_lib_names = [lib["name"] for lib in selected_resources["selected_libraries"]]
        
        self.resource_manager.select_tools_by_names(selected_tool_names)
        self.resource_manager.select_data_by_names(selected_data_names)
        self.resource_manager.select_libraries_by_names(selected_lib_names)
        
        print(f"Selected {len(selected_tool_names)} tools, {len(selected_data_names)} data items, {len(selected_lib_names)} libraries")


    def go(self, prompt: str):
        """
        Execute the agent with the given prompt.

        Args:
            prompt: The user's query
        """
        self.critic_count = 0
        self.user_task = prompt

        # Select relevant resources using tool retriever if enabled
        if self.use_tool_retriever:
            # Mark relevant resources as selected in ResourceManager
            self._select_resources_for_prompt(prompt)
            # Regenerate system prompt with selected resources
            self.system_prompt = self._generate_system_prompt(
                self_critic=self.self_critic,
                is_retrieval=True,
            )

        inputs = {"input": [HumanMessage(content=prompt)], "next_step": None}
        config = {"recursion_limit": 500, "configurable": {"thread_id": 42}}
        self.log = []

        # Store the final conversation state for markdown generation
        final_state = None

        for state in self.app.stream(inputs, stream_mode="values", config=config):
            input = state["input"][-1]
            out = pretty_print(input)
            self.log.append(out)
            final_state = state  # Store the latest state

        # Store the conversation state for markdown generation
        self._conversation_state = final_state

        return self.log, input.content

    def _clear_execution_plots(self):
        """
        Clear execution plots before new execution.

        This function clears any previously captured plots from the execution environment
        before starting a new execution. This prevents old plots from appearing in
        new execution results.

        Note:
            This function calls the clear_captured_plots utility function and handles
            any exceptions gracefully to prevent execution failures.
        """
        try:
            from BaseAgent.tools.support_tools import clear_captured_plots

            clear_captured_plots()
        except Exception as e:
            print(f"Warning: Could not clear execution plots: {e}")


    def _inject_custom_functions_to_repl(self):
        """
        Inject custom functions into the Python REPL execution environment.
        This makes custom tools from ResourceManager available during code execution.
        """
        # Get custom tools from ResourceManager
        custom_tools = self.resource_manager.collection.custom_tools
        
        # Extract callable functions from CustomTool objects
        custom_functions = {}
        for tool in custom_tools:
            if tool.function is not None:
                custom_functions[tool.name] = tool.function
        
        inject_custom_functions_to_repl(custom_functions)


    def _record_usage(self, usage):
        if usage is None:
            return
        self._usage_metrics.append(usage)