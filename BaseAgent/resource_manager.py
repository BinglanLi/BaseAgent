"""
Resource Manager for BaseAgent

This module provides a unified manager for organizing and accessing resources
using Pydantic models. Consolidates ToolRegistry functionality.
"""

import importlib
from typing import Optional
import pandas as pd
from BaseAgent.resources import (
    Tool,
    DataLakeItem,
    Library,
    CustomTool,
    CustomData,
    CustomSoftware,
    ResourceCollection,
    ToolParameter,
)


class ResourceManager:
    """Manager for all BaseAgent resources using Pydantic models.
    
    Consolidates functionality from ToolRegistry including:
    - Tool ID management
    - DataFrame for tool retrieval
    - Tool lookup by name/ID
    - Tool filtering by module
    """
    
    def __init__(self):
        """Initialize the resource manager."""
        self.collection = ResourceCollection()
        
        # Tool registry features
        self.next_tool_id: int = 0
        self.tool_document_df: pd.DataFrame = pd.DataFrame(columns=["docid", "document_content"])
    
    # ==========================================================================
    # Tool Management
    # ==========================================================================
    
    def add_tool(self, tool: Tool) -> None:
        """Add a tool to the collection with automatic ID assignment.
        
        The ID is an internal counter for this manager instance and is always
        auto-assigned sequentially, regardless of any existing ID on the tool.
        
        Args:
            tool: Tool object
        """
        # Always assign a fresh sequential ID (internal counter)
        tool.id = self.next_tool_id
        self.next_tool_id += 1
        
        self.collection.tools.append(tool)
        
        # Add to dataframe for retrieval
        new_row = pd.DataFrame([[tool.id, tool]], columns=["docid", "document_content"])
        self.tool_document_df = pd.concat([self.tool_document_df, new_row], ignore_index=True)
    
    def add_custom_tool(self, custom_tool: CustomTool) -> None:
        """Add a custom tool to the collection.
        
        Args:
            custom_tool: CustomTool object
        """
        self.collection.custom_tools.append(custom_tool)
    
    def get_all_tools(self) -> list[Tool]:
        """Get all tools including custom tools.
        
        Returns:
            List of Tool objects
        """
        return self.collection.get_all_tools()
    
    def get_tool_by_name(self, name: str) -> Optional[Tool]:
        """Find a tool by name (searches both standard and custom tools).
        
        Args:
            name: Tool name
            
        Returns:
            Tool object if found, None otherwise
        """
        for tool in self.get_all_tools():
            if tool.name == name:
                return tool
        return None
    
    def find_tool_by_id(self, tool_id: int) -> Optional[Tool]:
        """Find a tool by ID.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Tool object if found, None otherwise
        """
        for tool in self.collection.tools:
            if tool.id == tool_id:
                return tool
        return None
    
    def get_tool_id_by_name(self, name: str) -> Optional[int]:
        """Get tool ID by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool ID if found, None otherwise
        """
        tool = self.get_tool_by_name(name)
        return tool.id if tool else None
    
    def get_tool_name_by_id(self, tool_id: int) -> Optional[str]:
        """Get tool name by ID.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            Tool name if found, None otherwise
        """
        tool = self.find_tool_by_id(tool_id)
        return tool.name if tool else None
    
    def list_all_tools(self) -> list[dict[str, str | int]]:
        """List all tools with their names and IDs.
        
        Returns:
            List of dictionaries with tool name and ID
        """
        return [{"name": tool.name, "id": tool.id} for tool in self.collection.tools if tool.id is not None]
    
    def remove_tool_by_id(self, tool_id: int) -> bool:
        """Remove a tool by ID.
        
        Args:
            tool_id: Tool ID
            
        Returns:
            True if tool was removed, False otherwise
        """
        tool = self.find_tool_by_id(tool_id)
        if tool:
            self.collection.tools = [t for t in self.collection.tools if t.id != tool_id]
            # Remove from dataframe
            self.tool_document_df = self.tool_document_df[
                self.tool_document_df['docid'] != tool_id
            ].reset_index(drop=True)
            return True
        return False
    
    def remove_tool_by_name(self, name: str) -> bool:
        """Remove a tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            True if tool was removed, False otherwise
        """
        tool = self.get_tool_by_name(name)
        if tool and tool.id is not None:
            tool_id = tool.id
            self.collection.tools = [t for t in self.collection.tools if t.name != name]
            # Remove from dataframe
            self.tool_document_df = self.tool_document_df[
                self.tool_document_df['docid'] != tool_id
            ].reset_index(drop=True)
            return True
        return False
    
    def filter_tools_by_module(self, module: str) -> list[Tool]:
        """Filter tools by module name.
        
        Args:
            module: Module name or substring
            
        Returns:
            List of matching Tool objects
        """
        return [
            tool for tool in self.get_all_tools()
            if module in tool.module
        ]
    
    # ==========================================================================
    # Data Management
    # ==========================================================================
    
    def add_data_item(self, data_item: DataLakeItem) -> None:
        """Add a data lake item.
        
        Args:
            data_item: DataLakeItem object
        """
        self.collection.data_lake.append(data_item)
    
    def add_custom_data(self, custom_data: CustomData) -> None:
        """Add custom data.
        
        Args:
            custom_data: CustomData object
        """
        self.collection.custom_data.append(custom_data)
    
    def get_all_data(self) -> list[DataLakeItem | CustomData]:
        """Get all data items including custom data.
        
        Returns:
            List of data items
        """
        return self.collection.get_all_data()
    
    def find_data_by_filename(self, filename: str) -> Optional[DataLakeItem]:
        """Find a data item by filename.
        
        Args:
            filename: File name
            
        Returns:
            DataLakeItem if found, None otherwise
        """
        for item in self.get_all_data():
            if hasattr(item, 'filename') and item.filename == filename:
                return item
            elif hasattr(item, 'name') and item.name == filename:
                return item
        return None
    
    def filter_data_by_category(self, category: str) -> list[DataLakeItem]:
        """Filter data by category.
        
        Args:
            category: Category name
            
        Returns:
            List of matching data items
        """
        return [
            item for item in self.get_all_data()
            if hasattr(item, 'category') and item.category == category
        ]
    
    def filter_data_by_format(self, format: str) -> list[DataLakeItem]:
        """Filter data by format.
        
        Args:
            format: File format (e.g., 'csv', 'parquet')
            
        Returns:
            List of matching data items
        """
        return [
            item for item in self.get_all_data()
            if hasattr(item, 'format') and item.format == format.lower()
        ]
    
    # ==========================================================================
    # Library Management
    # ==========================================================================
    
    def add_library(self, library: Library) -> None:
        """Add a library.
        
        Args:
            library: Library object
        """
        self.collection.libraries.append(library)
    
    def add_custom_software(self, custom_software: CustomSoftware) -> None:
        """Add custom software.
        
        Args:
            custom_software: CustomSoftware object
        """
        self.collection.custom_software.append(custom_software)
    
    def get_all_libraries(self) -> list[Library | CustomSoftware]:
        """Get all libraries including custom software.
        
        Returns:
            List of libraries
        """
        return self.collection.get_all_libraries()
    
    def find_library_by_name(self, name: str) -> Optional[Library]:
        """Find a library by name.
        
        Args:
            name: Library name
            
        Returns:
            Library if found, None otherwise
        """
        for lib in self.get_all_libraries():
            if lib.name == name:
                return lib
        return None
    
    def filter_libraries_by_type(self, lib_type: str) -> list[Library]:
        """Filter libraries by type.
        
        Args:
            lib_type: Library type ('Python', 'R', or 'CLI')
            
        Returns:
            List of matching libraries
        """
        return [
            lib for lib in self.get_all_libraries()
            if hasattr(lib, 'type') and lib.type == lib_type
        ]
    
    def filter_libraries_by_category(self, category: str) -> list[Library]:
        """Filter libraries by category.
        
        Args:
            category: Category name
            
        Returns:
            List of matching libraries
        """
        return [
            lib for lib in self.get_all_libraries()
            if hasattr(lib, 'category') and lib.category == category
        ]
    
    # ==========================================================================
    # Bulk Operations
    # ==========================================================================
    
    def load_tools(self, tools: list[Tool]) -> None:
        """Load multiple tools at once with ID assignment and dataframe updates.
        
        Args:
            tools: List of Tool objects
        """
        for tool in tools:
            self.add_tool(tool)  # Use add_tool to ensure IDs and dataframe are updated
    
    def load_data_items(self, items: list[DataLakeItem]) -> None:
        """Load multiple data items at once.
        
        Args:
            items: List of DataLakeItem objects
        """
        self.collection.data_lake.extend(items)
    
    def load_libraries(self, libraries: list[Library]) -> None:
        """Load multiple libraries at once.
        
        Args:
            libraries: List of Library objects
        """
        self.collection.libraries.extend(libraries)
    
    # ==========================================================================
    # Summary and Statistics
    # ==========================================================================
    
    def get_summary(self) -> dict:
        """Get resource statistics.
        
        Returns:
            Dictionary with counts of each resource type
        """
        return {
            "tools": {
                "standard": len(self.collection.tools),
                "custom": len(self.collection.custom_tools),
                "total": len(self.get_all_tools())
            },
            "data": {
                "data_lake": len(self.collection.data_lake),
                "custom": len(self.collection.custom_data),
                "total": len(self.get_all_data())
            },
            "libraries": {
                "standard": len(self.collection.libraries),
                "custom": len(self.collection.custom_software),
                "total": len(self.get_all_libraries())
            }
        }
    
    def get_categories(self) -> dict[str, list[str]]:
        """Get all categories for data and libraries.
        
        Returns:
            Dictionary with data and library categories
        """
        data_categories = set()
        for item in self.get_all_data():
            if hasattr(item, 'category') and item.category:
                data_categories.add(item.category)
        
        lib_categories = set()
        for lib in self.get_all_libraries():
            if hasattr(lib, 'category') and lib.category:
                lib_categories.add(lib.category)
        
        return {
            "data_categories": sorted(list(data_categories)),
            "library_categories": sorted(list(lib_categories))
        }
    
    # ==========================================================================
    # Serialization
    # ==========================================================================
    
    def export_json(self, filepath: str) -> None:
        """Export all resources to JSON file.
        
        Args:
            filepath: Path to output JSON file
        """
        self.collection.export_json(filepath)
    
    def load_from_json(self, filepath: str) -> None:
        """Load resources from JSON file into this manager.
        
        Args:
            filepath: Path to JSON file
        """
        self.collection = ResourceCollection.import_json(filepath)
    
    def load_builtin_tools(self) -> None:
        """Load built-in tools directly from tool_description modules.
        
        This method directly imports tool definitions from BaseAgent.tools.tool_description
        and converts them to Pydantic Tool models.
        
        The tool descriptions are loaded from:
        - BaseAgent.tools.tool_description.support_tools
        (Add more tool description modules here as needed)
        """
        # List of tool description modules to load
        tool_modules = [
            "support_tools",
        ]
        
        for module_field in tool_modules:
            # Import the tool description module
            module_name = f"BaseAgent.tools.tool_description.{module_field}"
            module = importlib.import_module(module_name)
            
            # Get the tool descriptions from the module
            tool_list = module.description
            
            # Convert each tool dict to Pydantic Tool model
            full_module_name = f"BaseAgent.tools.{module_field}"
            for tool_dict in tool_list:
                # Convert parameter dicts to ToolParameter models
                required_params = [
                    ToolParameter(**p) for p in tool_dict.get("required_parameters", [])
                ]
                optional_params = [
                    ToolParameter(**p) for p in tool_dict.get("optional_parameters", [])
                ]
                
                # Create Tool model
                tool = Tool(
                    name=tool_dict["name"],
                    description=tool_dict["description"],
                    module=full_module_name,
                    required_parameters=required_params,
                    optional_parameters=optional_params
                )
                self.add_tool(tool)
    
    # ==========================================================================
    # Resource Selection Management
    # ==========================================================================
    
    def select_all_resources(self) -> None:
        """Mark all resources as selected for use in prompts."""
        for tool in self.collection.tools:
            tool.selected = True
        for tool in self.collection.custom_tools:
            tool.selected = True
        for data in self.collection.data_lake:
            data.selected = True
        for data in self.collection.custom_data:
            data.selected = True
        for lib in self.collection.libraries:
            lib.selected = True
        for lib in self.collection.custom_software:
            lib.selected = True
    
    def deselect_all_resources(self) -> None:
        """Mark all resources as unselected (not used in prompts)."""
        for tool in self.collection.tools:
            tool.selected = False
        for tool in self.collection.custom_tools:
            tool.selected = False
        for data in self.collection.data_lake:
            data.selected = False
        for data in self.collection.custom_data:
            data.selected = False
        for lib in self.collection.libraries:
            lib.selected = False
        for lib in self.collection.custom_software:
            lib.selected = False
    
    def select_tools_by_names(self, tool_names: list[str]) -> None:
        """
        Select specific tools by name, deselecting all others.
        
        Args:
            tool_names: List of tool names to select
        """
        # Deselect all tools first
        for tool in self.collection.tools + self.collection.custom_tools:
            tool.selected = tool.name in tool_names
    
    def select_data_by_names(self, data_names: list[str]) -> None:
        """
        Select specific data items by name, deselecting all others.
        
        Args:
            data_names: List of data item names (filenames) to select
        """
        # Deselect all data first
        for data in self.collection.data_lake + self.collection.custom_data:
            name = data.filename if hasattr(data, 'filename') else data.name
            data.selected = name in data_names
    
    def select_libraries_by_names(self, library_names: list[str]) -> None:
        """
        Select specific libraries by name, deselecting all others.
        
        Args:
            library_names: List of library names to select
        """
        # Deselect all libraries first
        for lib in self.collection.libraries + self.collection.custom_software:
            lib.selected = lib.name in library_names
    
    def get_selected_tools(self) -> list[Tool]:
        """Get only the tools currently marked as selected."""
        return [tool for tool in self.get_all_tools() if tool.selected]
    
    def get_selected_data(self) -> list[DataLakeItem | CustomData]:
        """Get only the data items currently marked as selected."""
        return [data for data in self.get_all_data() if data.selected]
    
    def get_selected_libraries(self) -> list[Library | CustomSoftware]:
        """Get only the libraries currently marked as selected."""
        return [lib for lib in self.get_all_libraries() if lib.selected]
    
    # ==========================================================================
    # Prompt Generation
    # ==========================================================================
    
    @staticmethod
    def format_tools_by_module(tools_by_module: dict) -> str:
        """
        Format tools grouped by module with detailed parameter information.
        
        This method provides a comprehensive, module-organized view of tools suitable
        for LLM prompts. Each module section includes all tools with their complete
        parameter specifications.
        
        Args:
            tools_by_module: Dictionary mapping module names to lists of tools
                           (Tool objects or dicts with tool definitions)
            
        Returns:
            Formatted string with tools organized by module/import file
            
        Example:
            >>> from collections import defaultdict
            >>> tools_by_module = defaultdict(list)
            >>> for tool in manager.get_all_tools():
            ...     tools_by_module[tool.module].append(tool)
            >>> formatted = ResourceManager.format_tools_by_module(tools_by_module)
            
        Output format:
            Import file: BaseAgent.tools.support_tools
            ==========================================
            Method: run_python_repl
              Description: Execute Python code...
              Required Parameters:
                - code (string): Python code to execute [Default: None]
        """
        lines = []
        
        for module_name, tools in tools_by_module.items():
            # Module header
            lines.append(f"Import file: {module_name}")
            lines.append("=" * (len("Import file: ") + len(module_name)))
            
            for tool in tools:
                # Convert dict to Tool-like structure if needed
                if isinstance(tool, dict):
                    tool_name = tool.get('name', 'N/A')
                    tool_desc = tool.get('description', 'No description provided.')
                    req_params = tool.get("required_parameters", [])
                    opt_params = tool.get("optional_parameters", [])
                else:
                    # Pydantic Tool model
                    tool_name = tool.name
                    tool_desc = tool.description
                    req_params = tool.required_parameters
                    opt_params = tool.optional_parameters
                
                # Format tool header
                lines.append(f"Method: {tool_name}")
                lines.append(f"  Description: {tool_desc}")

                # Format parameters using helper method
                if req_params:
                    lines.append("  Required Parameters:")
                    lines.extend(ResourceManager._format_parameters(req_params, indent="    "))

                if opt_params:
                    lines.append("  Optional Parameters:")
                    lines.extend(ResourceManager._format_parameters(opt_params, indent="    "))

                lines.append("")  # Empty line between tools
            
            lines.append("")  # Extra empty line after each module

        return "\n".join(lines)
    
    @staticmethod
    def _format_parameters(params: list, indent: str = "    ") -> list[str]:
        """
        Helper method to format parameter lists consistently.
        
        Args:
            params: List of parameters (dicts or ToolParameter objects)
            indent: Indentation string for each parameter line
            
        Returns:
            List of formatted parameter strings
        """
        lines = []
        for param in params:
            if isinstance(param, dict):
                param_name = param.get("name", "N/A")
                param_type = param.get("type", "N/A")
                param_desc = param.get("description", "No description")
                param_default = param.get("default", "None")
            else:
                # Pydantic ToolParameter model
                param_name = param.name
                param_type = param.type
                param_desc = param.description
                param_default = param.default if param.default is not None else "None"
            
            lines.append(f"{indent}- {param_name} ({param_type}): {param_desc} [Default: {param_default}]")
        
        return lines
    
    def get_tools_description(self) -> str:
        """Get formatted tool descriptions for prompts.
        
        Returns:
            Formatted string describing all tools
        """
        lines = []
        for i, tool in enumerate(self.get_all_tools()):
            lines.append(f"{i}. {tool.name}")
            lines.append(f"   Module: {tool.module}")
            lines.append(f"   Description: {tool.description}")
            if tool.required_parameters:
                params = ", ".join([p.name for p in tool.required_parameters])
                lines.append(f"   Required params: {params}")
            lines.append("")
        return "\n".join(lines)
    
    def get_data_description(self) -> str:
        """Get formatted data descriptions for prompts.
        
        Returns:
            Formatted string describing all data items
        """
        lines = []
        for i, item in enumerate(self.get_all_data()):
            name = item.filename if hasattr(item, 'filename') else item.name
            lines.append(f"{i}. {name}")
            lines.append(f"   Description: {item.description}")
            if hasattr(item, 'category') and item.category:
                lines.append(f"   Category: {item.category}")
            if hasattr(item, 'format'):
                lines.append(f"   Format: {item.format}")
            lines.append("")
        return "\n".join(lines)
    
    def get_libraries_description(self) -> str:
        """Get formatted library descriptions for prompts.
        
        Returns:
            Formatted string describing all libraries
        """
        lines = []
        for i, lib in enumerate(self.get_all_libraries()):
            lib_type = f" ({lib.type})" if hasattr(lib, 'type') else ""
            lines.append(f"{i}. {lib.name}{lib_type}")
            lines.append(f"   Description: {lib.description}")
            if hasattr(lib, 'category') and lib.category:
                lines.append(f"   Category: {lib.category}")
            if hasattr(lib, 'version') and lib.version:
                lines.append(f"   Version: {lib.version}")
            lines.append("")
        return "\n".join(lines)

