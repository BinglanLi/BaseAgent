# BaseAgent Resource Management

Pure Pydantic-based resource management system for organizing tools, data, and libraries.

## Quick Start

### Installation
```bash
pip install pydantic>=2.11.7
```

### Basic Usage
```python
from BaseAgent.resource_manager import ResourceManager
from BaseAgent.resources import Tool, DataLakeItem, Library, ToolParameter

# Initialize
manager = ResourceManager()

# Add resources
tool = Tool(
    name="my_tool",
    description="Does something",
    required_parameters=[],
    module="my_module"
)
manager.add_tool(tool)

# Query
tool = manager.find_tool_by_name("my_tool")
summary = manager.get_summary()
```

## Core Files

- **`resources.py`** - Pydantic models (Tool, DataLakeItem, Library, etc.)
- **`resource_manager.py`** - Unified resource manager with query/filter methods and tool registry features
- **`base_agent_methods.py`** - add_tool and add_mcp implementations
- **`env_desc.py`** - Data lake and library definitions

## Pydantic Models

### Tool
```python
Tool(
    name: str,                           # Valid Python identifier
    description: str,
    required_parameters: list[ToolParameter],
    optional_parameters: list[ToolParameter],
    module: str,
    id: Optional[int] = None
)
```

### DataLakeItem
```python
DataLakeItem(
    filename: str,
    description: str,
    format: str,                         # e.g., "csv", "parquet"
    category: Optional[str] = None,      # e.g., "drug_discovery"
    size_mb: Optional[float] = None,     # Must be >= 0
    path: Optional[str] = None
)
```

### Library
```python
Library(
    name: str,
    description: str,
    type: Literal["Python", "R", "CLI"],
    version: Optional[str] = None,
    category: Optional[str] = None,
    installation_cmd: Optional[str] = None
)
```

## Integration with BaseAgent

```python
from BaseAgent.resource_manager import ResourceManager
from BaseAgent.resources import Tool, ToolParameter
from BaseAgent.base_agent_methods import BaseAgentResourceMethods

class BaseAgent(BaseAgentResourceMethods):
    def __init__(self):
        # Initialize resource manager
        self.resource_manager = ResourceManager()
        
        # Load default tools, data, libraries
        # See base_agent_methods.py for add_tool and add_mcp
```

## Add Custom Tool

```python
def my_function(query: str, limit: int = 10) -> list:
    return []

agent.add_tool(
    name="my_function",
    function=my_function,
    description="My custom function",
    required_parameters=[
        {"name": "query", "description": "Search query", "type": "str"}
    ],
    optional_parameters=[
        {"name": "limit", "description": "Max results", "type": "int", "default": 10}
    ]
)
```

## Add MCP Server

```python
# mcp_config.yaml:
# mcpServers:
#   filesystem:
#     command: "npx"
#     args: ["-y", "@modelcontextprotocol/server-filesystem", "/path"]

agent.add_mcp("./mcp_config.yaml")
```

## Resource Manager API

### Tools
```python
# Adding tools
add_tool(tool: Tool)                         # Automatically assigns IDs
add_custom_tool(custom_tool: CustomTool)
load_builtin_tools()                          # Load tools from tool_description modules

# Querying tools
get_all_tools() -> list[Tool]                # Includes custom tools
find_tool_by_name(name: str) -> Tool | None
find_tool_by_id(tool_id: int) -> Tool | None
get_tool_id_by_name(name: str) -> int | None
get_tool_name_by_id(tool_id: int) -> str | None
filter_tools_by_module(module: str) -> list[Tool]
list_all_tools() -> list[dict]               # [{name, id}, ...]

# Removing tools
remove_tool_by_id(tool_id: int) -> bool
remove_tool_by_name(name: str) -> bool

# DataFrame for retrieval (pandas)
tool_document_df: pd.DataFrame               # For tool retriever systems
```

### Data
```python
add_data_item(data: DataLakeItem)
add_custom_data(custom_data: CustomData)
get_all_data() -> list[DataLakeItem | CustomData]  # Includes custom data
find_data_by_filename(filename: str) -> DataLakeItem | None
filter_data_by_category(category: str) -> list[DataLakeItem]
filter_data_by_format(format: str) -> list[DataLakeItem]
```

### Libraries
```python
add_library(lib: Library)
add_custom_software(custom_software: CustomSoftware)
get_all_libraries() -> list[Library | CustomSoftware]  # Includes custom software
find_library_by_name(name: str) -> Library | None
filter_libraries_by_type(type: str) -> list[Library]
filter_libraries_by_category(category: str) -> list[Library]
```

### Resource Selection Management
```python
# Select/deselect all resources
select_all_resources() -> None     # Mark all resources as selected
deselect_all_resources() -> None   # Mark all resources as unselected

# Select specific resources by name
select_tools_by_names(tool_names: list[str]) -> None
select_data_by_names(data_names: list[str]) -> None
select_libraries_by_names(library_names: list[str]) -> None

# Get only selected resources
get_selected_tools() -> list[Tool]
get_selected_data() -> list[DataLakeItem | CustomData]
get_selected_libraries() -> list[Library | CustomSoftware]
```

### Utilities
```python
get_summary() -> dict
get_categories() -> dict
export_json(filepath: str)         # Note: Internal tool IDs are excluded
load_from_json(filepath: str)      # Tool IDs are reassigned on load
```

### Prompt Generation
```python
# Format tools grouped by module (comprehensive with detailed parameters)
ResourceManager.format_tools_by_module(tools_by_module: dict) -> str
# Input: {module_name: [Tool objects or dicts], ...}
# Output: Formatted string with tools organized by module/import file
# Shows detailed parameter specifications for each tool

# Format all tools (numbered list)
get_tools_description() -> str

# Format all data items (numbered list)
get_data_description() -> str

# Format all libraries (numbered list)
get_libraries_description() -> str
```

## Examples

### Query Resources
```python
# Find
tool = manager.find_tool_by_name("run_python_repl")

# Filter
drug_data = manager.filter_data_by_category("drug_discovery")
python_libs = manager.filter_libraries_by_type("Python")

# Summary
summary = manager.get_summary()
# {
#     "tools": {"standard": 5, "custom": 2, "total": 7},
#     "data": {"data_lake": 78, "custom": 3, "total": 81},
#     "libraries": {"standard": 50, "custom": 1, "total": 51}
# }
```

### Format Tools for LLM Prompts
```python
from collections import defaultdict

# Group tools by module for comprehensive formatting
tools_by_module = defaultdict(list)
for tool in manager.get_all_tools():
    tools_by_module[tool.module].append(tool)

# Format with detailed parameter information (organized by module)
formatted_prompt = ResourceManager.format_tools_by_module(tools_by_module)
# Output:
# Import file: BaseAgent.tools.support_tools
# ==========================================
# Method: run_python_repl
#   Description: Execute Python code...
#   Required Parameters:
#     - code (string): Python code to execute [Default: None]
#   Optional Parameters:
#     - timeout (int): Timeout in seconds [Default: 60]

# Simple numbered list (all tools)
all_tools = manager.get_tools_description()
```

### Resource Selection for Prompt Generation
```python
# By default, all resources are selected (selected=True)
all_tools = manager.get_selected_tools()  # Returns all tools

# Select only specific tools for a task
manager.select_tools_by_names(["run_python_repl", "fetch_data"])
manager.select_data_by_names(["BindingDB_All_202409.tsv"])
manager.select_libraries_by_names(["pandas", "numpy"])

# Now only selected resources will be included in prompts
selected_tools = manager.get_selected_tools()  # Returns only 2 tools

# In BaseAgent, tool retriever automatically manages selection:
# agent.go("Analyze protein binding data")
# -> Retriever selects relevant resources
# -> ResourceManager marks them as selected=True
# -> Prompt generation uses only selected resources
```

### Export/Import
```python
# Export
manager.export_json("resources.json")

# Import
manager = ResourceManager()
manager.load_from_json("resources.json")
```

## Benefits

✅ **Pure Pydantic** - No backward compatibility overhead  
✅ **Type-safe** - Full validation and type checking  
✅ **Clean API** - Simple, intuitive methods  
✅ **Well-organized** - Rich metadata for tracking  
✅ **Easy to use** - Straightforward integration  

## See Also

- `IMPROVEMENTS_SUMMARY.md` - Detailed improvements documentation
- `QUICK_REFERENCE.md` - Quick reference guide
- `base_agent_methods.py` - Complete add_tool and add_mcp implementations

