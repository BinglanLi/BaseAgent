"""API schema generation: introspection + optional LLM enhancement."""

import inspect
import json
import re
from typing import Callable, get_type_hints

from langchain_core.language_models.chat_models import BaseChatModel

from BaseAgent.resources import ToolParameter, Tool, CustomTool


# ==============================================================================
# API Schema Generation - Hybrid Approach (Introspection + LLM)
# ==============================================================================
def _parse_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """
    Parse a docstring to extract description and parameter descriptions.

    Args:
        docstring: The function's docstring

    Returns:
        Tuple of (main_description, param_descriptions_dict)
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split('\n')
    description_lines = []
    param_descriptions = {}
    in_params_section = False

    for line in lines:
        line = line.strip()

        # Check for parameter section markers
        if line.lower() in ['parameters:', 'args:', 'arguments:']:
            in_params_section = True
            continue

        # Check for other section markers (stops param parsing)
        if line.lower() in ['returns:', 'return:', 'raises:', 'examples:', 'example:', 'note:', 'notes:']:
            in_params_section = False
            continue

        if in_params_section:
            # Parse parameter descriptions (supports various formats)
            # Format: param_name: description
            # Format: param_name (type): description
            match = re.match(r'(\w+)\s*(?:\([^)]+\))?\s*:\s*(.+)', line)
            if match:
                param_name, param_desc = match.groups()
                param_descriptions[param_name] = param_desc.strip()
        elif not in_params_section and line and not line.startswith('-'):
            # Collect description lines (before any section)
            description_lines.append(line)

    main_description = ' '.join(description_lines).strip()
    return main_description, param_descriptions


def _type_to_string(type_hint) -> str:
    """
    Convert a type hint to a string representation.

    Args:
        type_hint: Type hint object

    Returns:
        String representation of the type
    """
    if type_hint is inspect.Parameter.empty:
        return "Any"

    # Handle string type hints (already strings)
    if isinstance(type_hint, str):
        return type_hint

    # Get the string representation
    type_str = str(type_hint)

    # Clean up common type representations
    type_str = type_str.replace('typing.', '')
    type_str = type_str.replace('<class \'', '').replace('\'>', '')

    # Handle None type
    if type_hint is type(None):
        return "None"

    return type_str


def extract_schema_from_function(func: Callable) -> dict:
    """
    Extract API schema from a function using Python introspection.

    This function uses inspect module to extract function signature, type hints,
    and docstring to build a complete API schema without needing an LLM.

    Args:
        func: The function to analyze

    Returns:
        Dictionary containing the API schema with name, description,
        required_parameters, and optional_parameters
    """
    # Get function signature
    sig = inspect.signature(func)

    # Get type hints
    try:
        type_hints = get_type_hints(func)
    except Exception:
        # If type hints fail (e.g., forward references), fall back to annotations
        type_hints = {}

    # Parse docstring
    docstring = inspect.getdoc(func) or ""
    main_description, param_descriptions = _parse_docstring(docstring)

    # Extract function name
    function_name = func.__name__

    # Use first line of docstring as description, or create a default
    if main_description:
        description = main_description
    else:
        description = f"Function {function_name}"

    # Process parameters
    required_params = []
    optional_params = []

    for param_name, param in sig.parameters.items():
        # Skip self and cls parameters
        if param_name in ['self', 'cls']:
            continue

        # Get type
        param_type = type_hints.get(param_name, param.annotation)
        type_str = _type_to_string(param_type)

        # Get description from docstring or create default
        param_desc = param_descriptions.get(param_name, f"Parameter {param_name}")

        # Determine if required or optional
        has_default = param.default is not inspect.Parameter.empty
        default_value = param.default if has_default else None

        param_dict = {
            "name": param_name,
            "type": type_str,
            "description": param_desc,
            "default": default_value
        }

        if has_default:
            optional_params.append(param_dict)
        else:
            # Required parameters should have default=None in the schema
            param_dict["default"] = None
            required_params.append(param_dict)

    return {
        "name": function_name,
        "description": description,
        "required_parameters": required_params,
        "optional_parameters": optional_params
    }


def enhance_description_with_llm(function_code: str, schema: dict, llm: BaseChatModel | None = None) -> str:
    """
    Use LLM to enhance or generate a better description for the function.

    Args:
        function_code: Source code of the function
        schema: Current schema with basic info
        llm: Language model instance

    Returns:
        Enhanced description string
    """
    prompt = f"""Analyze this Python function and provide a clear, concise description (1-2 sentences) of what it does.
Focus on what the function accomplishes, not implementation details.

Function name: {schema['name']}
Current description: {schema['description']}

Function code:
{function_code}

Provide only the description text, nothing else."""

    try:
        # Use a simple string output for description enhancement
        response = llm.invoke(prompt)
        if hasattr(response, 'content'):
            return response.content.strip()
        return str(response).strip()
    except Exception as e:
        print(f"Warning: Could not enhance description with LLM: {e}")
        return schema['description']


def enhance_parameters_with_llm(function_code: str, schema: dict, llm: BaseChatModel | None = None) -> dict:
    """
    Use LLM to enhance parameter descriptions using structured output.

    Args:
        function_code: Source code of the function
        schema: Current schema with basic parameter info
        llm: Language model instance

    Returns:
        Enhanced schema with better parameter descriptions
    """
    prompt = f"""Analyze this Python function and provide clear, concise descriptions for each parameter.

Function code:
{function_code}

Current schema:
{json.dumps(schema, indent=2)}

Provide improved descriptions for all parameters. Keep descriptions clear and succinct (under 100 characters each).
Follow these rules:
- For variables without default values, set default to None, not null
- For boolean values, use capitalized True or False
- Be specific about what each parameter does
- Don't make up optional parameters that don't exist"""

    try:
        structured_llm = llm.with_structured_output(Tool)
        result = structured_llm.invoke(prompt)
        # Return dict without registry metadata (id will be None)
        return result.model_dump()
    except Exception as e:
        print(f"Warning: Could not enhance parameters with LLM: {e}")
        return schema


def function_to_api_schema(
    func: Callable,
    llm: BaseChatModel | None = None,
    enhance_description: bool = True,
    enhance_parameters: bool = False,
    module: str = None
) -> CustomTool:
    """
    Generate CustomTool from a function using hybrid approach (introspection + optional LLM).

    This is the main function that combines Python introspection with optional LLM
    enhancement. It returns a ready-to-use CustomTool Pydantic model.

    Strategy:
    1. Use introspection to extract structure (fast, reliable)
    2. Optionally enhance descriptions with LLM if available and requested
    3. Return CustomTool model with function reference

    Args:
        func: A callable Python function
        llm: Optional language model for description enhancement
        enhance_description: Whether to use LLM to improve the main description
        enhance_parameters: Whether to use LLM to improve parameter descriptions
        module: Module name for the tool (auto-detected if None)

    Returns:
        CustomTool: Pydantic model containing the function and its schema

    Example:
        >>> def add(x: int, y: int, verbose: bool = False) -> int:
        ...     '''Add two numbers together.'''
        ...     return x + y
        >>> custom_tool = function_to_api_schema(add)
        >>> custom_tool.name
        'add'
        >>> custom_tool.function(2, 3)
        5
    """
    if not callable(func):
        raise ValueError("func must be a callable")

    # Get function source code for LLM enhancement
    try:
        func_code = inspect.getsource(func)
    except Exception:
        func_code = str(func)

    # Extract schema using introspection
    schema = extract_schema_from_function(func)

    # Determine module name
    if module is None:
        if hasattr(func, '__module__') and func.__module__:
            module = func.__module__
        else:
            module = "custom_tools"

    # Enhance with LLM if available and requested
    if llm is not None and func_code is not None:
        # Enhance main description if it's generic or if requested
        if enhance_description and (
            schema['description'].startswith('Function ') or
            len(schema['description']) < 20
        ):
            schema['description'] = enhance_description_with_llm(func_code, schema, llm)

        # Enhance parameters if requested
        if enhance_parameters:
            enhanced = enhance_parameters_with_llm(func_code, schema, llm)
            # Keep the structure from introspection, but use enhanced descriptions
            if isinstance(enhanced, dict) and 'description' in enhanced:
                schema['description'] = enhanced.get('description', schema['description'])

                # Update parameter descriptions
                enhanced_req = {p['name']: p for p in enhanced.get('required_parameters', [])}
                enhanced_opt = {p['name']: p for p in enhanced.get('optional_parameters', [])}

                for param in schema['required_parameters']:
                    if param['name'] in enhanced_req:
                        param['description'] = enhanced_req[param['name']]['description']

                for param in schema['optional_parameters']:
                    if param['name'] in enhanced_opt:
                        param['description'] = enhanced_opt[param['name']]['description']

    # Convert parameter dicts to ToolParameter models
    required_params = [ToolParameter(**p) for p in schema['required_parameters']]
    optional_params = [ToolParameter(**p) for p in schema['optional_parameters']]

    # Create and return CustomTool model
    return CustomTool(
        name=schema['name'],
        description=schema['description'],
        module=module,
        function=func,
        required_parameters=required_params,
        optional_parameters=optional_params
    )


