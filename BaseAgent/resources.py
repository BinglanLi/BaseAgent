"""
Resource Data Models for BaseAgent

This module defines Pydantic models for all internal resources used by the BaseAgent.
These models provide:
- Type safety and validation
- Consistent structure across all resource types
- Easy serialization/deserialization
- Better IDE support with autocomplete
- Self-documentation

Resource Types:
    - Tool: Functions/APIs available to the agent
    - DataLakeItem: Datasets in the data lake
    - Library: Software libraries (Python/R packages, CLI tools)
    - CustomTool: User-defined custom tools
    - CustomData: User-defined custom datasets
    - CustomSoftware: User-defined custom software/libraries
"""

from pathlib import Path
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator


# ==============================================================================
# Base Resource Models
# ==============================================================================

class ToolParameter(BaseModel):
    """Model for a tool function parameter.
    
    Attributes:
        name: Parameter name
        description: Description of what the parameter does
        type: Python type as string (e.g., 'str', 'int', 'List[str]')
        default: Optional default value
    """
    name: str = Field(..., description="Parameter name")
    description: str = Field(..., description="Parameter description")
    type: str = Field(..., description="Parameter type (e.g., 'str', 'int', 'bool', 'List[str]')")
    default: Optional[Any] = Field(None, description="Default value if parameter is optional")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "query",
                "description": "Search query string",
                "type": "str",
                "default": None
            }
        }
    )


class Tool(BaseModel):
    """Model for a tool (function/API) available to the agent.
    
    Tools represent callable functions that the agent can use to perform tasks.
    Each tool has a unique name, description, and parameter specification.
    
    Attributes:
        name: Unique tool name (used for invocation)
        description: Clear description of what the tool does
        required_parameters: List of required parameters
        optional_parameters: List of optional parameters with defaults
        module: Python module path where the tool is defined
        id: Optional unique identifier (auto-assigned by registry)
        selected: Whether this tool is currently selected for use in prompts (default: True)
    """
    name: str = Field(..., description="Unique tool name")
    description: str = Field(..., description="Tool description")
    required_parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Required parameters for the tool"
    )
    optional_parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Optional parameters with defaults"
    )
    module: str = Field(..., description="Module path (e.g., 'BaseAgent.tools.support_tools')")
    id: Optional[int] = Field(None, description="Unique identifier (auto-assigned)")
    selected: bool = Field(True, description="Whether this tool is selected for use in prompts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "run_python_repl",
                "description": "Execute Python code in the notebook environment",
                "required_parameters": [
                    {
                        "name": "command",
                        "description": "Python command to execute",
                        "type": "str",
                        "default": None
                    }
                ],
                "optional_parameters": [],
                "module": "BaseAgent.tools.support_tools",
                "id": 0
            }
        }
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure tool name is valid Python identifier."""
        if not v.replace('_', '').isalnum():
            raise ValueError(f"Tool name '{v}' must be a valid Python identifier")
        return v


class DataLakeItem(BaseModel):
    """Model for a data lake item (dataset).
    
    Data lake items represent datasets available to the agent for analysis.
    Each item has a filename and descriptive information about its contents.
    
    Attributes:
        filename: Name of the file in the data lake
        description: Description of dataset contents and structure
        format: File format (e.g., 'parquet', 'csv', 'tsv', 'pkl')
        category: Optional category for organization (e.g., 'protein', 'genomics', 'drug')
        size_mb: Optional file size in megabytes
        path: Optional full path to the file
        selected: Whether this dataset is currently selected for use in prompts (default: True)
    """
    filename: str = Field(..., description="Filename in the data lake")
    description: str = Field(..., description="Description of the dataset")
    format: str = Field(..., description="File format (e.g., 'parquet', 'csv', 'tsv')")
    category: Optional[str] = Field(None, description="Dataset category (e.g., 'protein', 'genomics')")
    size_mb: Optional[float] = Field(None, description="File size in megabytes", ge=0)
    path: Optional[str] = Field(None, description="Full path to the file")
    selected: bool = Field(True, description="Whether this dataset is selected for use in prompts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "filename": "BindingDB_All_202409.tsv",
                "description": "Measured binding affinities between proteins and small molecules for drug discovery",
                "format": "tsv",
                "category": "drug",
                "size_mb": 450.5,
                "path": "./data/data_lake/BindingDB_All_202409.tsv"
            }
        }
    )
    
    @field_validator('format')
    @classmethod
    def validate_format(cls, v: str) -> str:
        """Ensure format is lowercase."""
        return v.lower()
    


class Library(BaseModel):
    """Model for a software library (Python/R package, CLI tool).
    
    Libraries represent software packages available in the execution environment.
    
    Attributes:
        name: Library/package name
        description: Description of functionality
        type: Library type (Python, R, or CLI)
        version: Optional version string
        category: Optional category (e.g., 'bioinformatics', 'machine_learning')
        installation_cmd: Optional installation command
        selected: Whether this library is currently selected for use in prompts (default: True)
    """
    name: str = Field(..., description="Library or package name")
    description: str = Field(..., description="Description of library functionality")
    type: Literal["Python", "R", "CLI"] = Field(..., description="Library type")
    version: Optional[str] = Field(None, description="Version string (e.g., '1.2.3')")
    category: Optional[str] = Field(None, description="Category (e.g., 'bioinformatics')")
    installation_cmd: Optional[str] = Field(None, description="Installation command")
    selected: bool = Field(True, description="Whether this library is selected for use in prompts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "biopython",
                "description": "Tools for biological computation including parsers for bioinformatics files",
                "type": "Python",
                "version": "1.79",
                "category": "bioinformatics",
                "installation_cmd": "pip install biopython"
            }
        }
    )
    


# ==============================================================================
# Custom Resource Models
# ==============================================================================

class CustomTool(BaseModel):
    """Model for user-defined custom tools.
    
    Custom tools are functions added by users to extend the agent's capabilities.
    
    Attributes:
        name: Unique tool name
        description: Tool description
        module: Module path where the tool is defined
        function: Optional callable function object (not serializable)
        required_parameters: Optional list of required parameters
        optional_parameters: Optional list of optional parameters
        selected: Whether this tool is currently selected for use in prompts (default: True)
    """
    name: str = Field(..., description="Unique custom tool name")
    description: str = Field(..., description="Tool description")
    module: str = Field(default="custom_tools", description="Module path")
    function: Optional[Any] = Field(None, exclude=True, description="Callable function (not serializable)")
    required_parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Required parameters"
    )
    optional_parameters: list[ToolParameter] = Field(
        default_factory=list,
        description="Optional parameters"
    )
    selected: bool = Field(True, description="Whether this tool is selected for use in prompts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "custom_search",
                "description": "Search custom database",
                "module": "custom_tools",
                "required_parameters": []
            }
        }
    )
    
    @field_validator('name')
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure tool name is valid Python identifier."""
        if not v.replace('_', '').isalnum():
            raise ValueError(f"Tool name '{v}' must be a valid Python identifier")
        return v
    
    def to_tool(self) -> Tool:
        """Convert to standard Tool model, preserving selection state."""
        return Tool(
            name=self.name,
            description=self.description,
            required_parameters=self.required_parameters,
            optional_parameters=self.optional_parameters,
            module=self.module,
            selected=self.selected,
        )


class CustomData(BaseModel):
    """Model for user-defined custom datasets.
    
    Custom data represents datasets added by users for specific tasks.
    
    Attributes:
        name: Dataset name
        description: Description of the dataset
        path: Optional path to the dataset file
        format: Optional file format
        metadata: Optional additional metadata
        selected: Whether this dataset is currently selected for use in prompts (default: True)
    """
    name: str = Field(..., description="Dataset name")
    description: str = Field(..., description="Dataset description")
    path: Optional[str] = Field(None, description="Path to dataset file")
    format: Optional[str] = Field(None, description="File format")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    selected: bool = Field(True, description="Whether this dataset is selected for use in prompts")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "my_custom_dataset.csv",
                "description": "Custom experimental data from lab",
                "path": "./data/custom/my_custom_dataset.csv",
                "format": "csv",
                "metadata": {"experiments": 150, "samples": 1000}
            }
        }
    )


class CustomSoftware(BaseModel):
    """Model for user-defined custom software/libraries.

    Custom software represents libraries or tools added by users.

    Attributes:
        name: Software/library name
        description: Description of functionality
        type: Optional software type
        installation_info: Optional installation instructions
        usage_example: Optional usage example
        selected: Whether this software is currently selected for use in prompts (default: True)
    """
    name: str = Field(..., description="Software or library name")
    description: str = Field(..., description="Description of functionality")
    type: Optional[str] = Field(None, description="Software type (e.g., 'Python', 'CLI')")
    installation_info: Optional[str] = Field(None, description="Installation instructions")
    usage_example: Optional[str] = Field(None, description="Usage example")
    selected: bool = Field(True, description="Whether this software is selected for use in prompts")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "custom_analyzer",
                "description": "Custom analysis pipeline for genomic data",
                "type": "Python",
                "installation_info": "pip install custom_analyzer",
                "usage_example": "from custom_analyzer import analyze; analyze(data)"
            }
        }
    )


class Skill(BaseModel):
    """Model for an agent skill (behavioral instructions and domain workflows).

    Skills are distinct from tools: a tool is a callable function, while a skill
    is a markdown document providing domain knowledge, workflows, and usage patterns
    that get injected into the system prompt.

    Skills are typically loaded from SKILL.md files with YAML frontmatter under a
    conventional directory structure: ``{skills_directory}/{skill_name}/SKILL.md``.

    Attributes:
        name: Unique skill identifier
        description: Short description used for retrieval matching
        tools: Tool names this skill requires (validated at configure time; auto-selected
            when this skill is chosen by the retriever)
        instructions: Markdown body with behavioral instructions
        source_path: Filesystem path to the SKILL.md file, if loaded from disk
        source_dir: Parent directory of the SKILL.md file; used to locate bundled resources
        selected: Whether this skill is included in the system prompt (default: True)
    """
    name: str = Field(..., description="Unique skill name")
    description: str = Field(..., description="Short description for retrieval matching")
    tools: list[str] = Field(default_factory=list, description="Tool names this skill requires")
    instructions: str = Field("", description="Markdown body with behavioral instructions")
    source_path: Optional[str] = Field(None, description="Filesystem path to the SKILL.md file")
    source_dir: Optional[str] = Field(None, description="Directory containing SKILL.md and optional bundled resources")
    selected: bool = Field(True, description="Whether this skill is included in the system prompt")

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "protein-structure-analysis",
                "description": "Guidance for protein structure prediction using AlphaFold and PDB",
                "tools": ["run_python_repl"],
                "instructions": "## Workflow\n1. Retrieve protein sequence...",
                "source_path": "./skills/protein-structure-analysis/SKILL.md",
                "source_dir": "./skills/protein-structure-analysis",
            }
        }
    )

    @property
    def has_bundled_resources(self) -> bool:
        """True if the skill directory contains a ``references/``, ``scripts/``, or ``assets/`` subdirectory."""
        if not self.source_dir:
            return False
        d = Path(self.source_dir)
        return any((d / sub).is_dir() for sub in ("references", "scripts", "assets"))

    @property
    def bundled_resource_manifest(self) -> dict[str, list[str]]:
        """List files in each bundled resource subdirectory.

        Returns:
            Mapping of subdirectory name to sorted list of file names.
            Empty dict when ``source_dir`` is not set or no subdirectories exist.
        """
        if not self.source_dir:
            return {}
        manifest: dict[str, list[str]] = {}
        for sub in ("references", "scripts", "assets"):
            sub_dir = Path(self.source_dir) / sub
            if sub_dir.is_dir():
                manifest[sub] = [f.name for f in sorted(sub_dir.iterdir()) if f.is_file()]
        return manifest


# ==============================================================================
# Resource Collections
# ==============================================================================

class ResourceCollection(BaseModel):
    """Collection of all resources available to the agent.
    
    This model aggregates all resource types for easy management and serialization.
    
    Attributes:
        tools: List of standard tools
        data_lake: List of data lake items
        libraries: List of software libraries
        custom_tools: List of custom tools
        custom_data: List of custom datasets
        custom_software: List of custom software
    """
    tools: list[Tool] = Field(default_factory=list, description="Standard tools")
    data_lake: list[DataLakeItem] = Field(default_factory=list, description="Data lake items")
    libraries: list[Library] = Field(default_factory=list, description="Software libraries")
    custom_tools: list[CustomTool] = Field(default_factory=list, description="Custom tools")
    custom_data: list[CustomData] = Field(default_factory=list, description="Custom datasets")
    custom_software: list[CustomSoftware] = Field(default_factory=list, description="Custom software")
    skills: list[Skill] = Field(default_factory=list, description="Agent skills")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tools": [],
                "data_lake": [],
                "libraries": [],
                "custom_tools": [],
                "custom_data": [],
                "custom_software": []
            }
        }
    )
    
    def get_all_tools(self) -> list[Tool]:
        """Get all tools including custom tools converted to Tool format."""
        all_tools = self.tools.copy()
        all_tools.extend([ct.to_tool() for ct in self.custom_tools])
        return all_tools
    
    def get_all_data(self) -> list[DataLakeItem | CustomData]:
        """Get all data including custom data."""
        return self.data_lake + self.custom_data
    
    def get_all_libraries(self) -> list[Library | CustomSoftware]:
        """Get all libraries including custom software."""
        return self.libraries + self.custom_software
    
    def export_json(self, filepath: str) -> None:
        """Export resource collection to JSON file.
        
        Note: Tool IDs are excluded as they are internal counters that get
        reassigned when the manager is reloaded.
        """
        with open(filepath, 'w') as f:
            f.write(self.model_dump_json(indent=2, exclude={'tools': {'__all__': {'id'}}}))
    
    @classmethod
    def import_json(cls, filepath: str) -> "ResourceCollection":
        """Import resource collection from JSON file."""
        with open(filepath, 'r') as f:
            return cls.model_validate_json(f.read())



