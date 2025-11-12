"""
Unit tests for BaseAgent.add_tool() functionality.

Run with: pytest test_add_tool.py -v
"""

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING

import pytest

from BaseAgent.base_agent import BaseAgent
from BaseAgent.tools.support_tools import _persistent_namespace

if TYPE_CHECKING:
    from typing import Callable


class TestAddToolBasic:
    """Test basic add_tool() functionality."""
    
    def test_simple_function(self, base_agent: BaseAgent, sample_function: Callable):
        """Test adding a simple function with basic parameters."""
        # Add the tool
        schema = base_agent.add_tool(sample_function)
        
        # Verify schema returned
        assert schema is not None
        assert 'name' in schema
        assert schema['name'] == 'greet'
        
        # Verify it's in ResourceManager
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 1
        
        tool = custom_tools[0]
        assert tool.name == 'greet'
        assert 'greet someone' in tool.description.lower()
        assert tool.module == 'custom_tools'
        assert tool.function is not None
        assert tool.selected is True
        
        # Verify parameters
        req_params = [p.name for p in tool.required_parameters]
        opt_params = [p.name for p in tool.optional_parameters]
        assert 'name' in req_params
        assert 'greeting' in opt_params
        
        # Verify it's callable
        result = tool.function("Alice", "Hi")
        assert result == "Hi, Alice!"
    
    def test_complex_function(self, base_agent: BaseAgent, complex_function: Callable):
        """Test adding a function with multiple parameter types."""
        # Add the tool
        schema = base_agent.add_tool(complex_function)
        
        # Verify parameters were captured correctly
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 1
        
        tool = custom_tools[0]
        assert tool.name == 'analyze_data'
        assert 'analyze numerical data' in tool.description.lower()
        
        # Verify required parameters
        assert len(tool.required_parameters) == 1
        assert tool.required_parameters[0].name == 'data'
        assert tool.required_parameters[0].type == 'list'
        
        # Verify optional parameters
        opt_param_names = [p.name for p in tool.optional_parameters]
        assert 'threshold' in opt_param_names
        assert 'normalize' in opt_param_names
        assert 'top_k' in opt_param_names
        
        # Test the function
        test_data = [0.1, 0.5, 0.8, 1.0, 0.3]
        result = tool.function(test_data, threshold=0.3, top_k=3)
        assert isinstance(result, list)
        assert len(result) <= 3
    
    def test_function_without_types(self, base_agent: BaseAgent):
        """Test adding a function without type hints."""
        def untyped_func(a, b, c=10):
            """A function without type hints."""
            return a + b + c
        
        schema = base_agent.add_tool(untyped_func)
        
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 1
        
        tool = custom_tools[0]
        assert tool.name == 'untyped_func'
        assert len(tool.required_parameters) == 2
        assert len(tool.optional_parameters) == 1
        
        # Function should still be callable
        result = tool.function(5, 3, c=2)
        assert result == 10


class TestREPLInjection:
    """Test REPL namespace injection."""
    
    def test_repl_injection(self, base_agent: BaseAgent, clear_repl_namespace):
        """Test that custom tools are available in REPL execution."""
        # Add a custom function
        def multiply(a: float, b: float) -> float:
            """Multiply two numbers."""
            return a * b
        
        base_agent.add_tool(multiply)
        
        # Inject into REPL
        base_agent._inject_custom_functions_to_repl()
        
        # Check if it's in the persistent namespace
        assert 'multiply' in _persistent_namespace
        result = _persistent_namespace['multiply'](5, 7)
        assert result == 35
    
    def test_multiple_functions_injection(self, base_agent: BaseAgent, math_functions: tuple, clear_repl_namespace):
        """Test injecting multiple functions into REPL."""
        add_func, mult_func, pow_func = math_functions
        
        # Add all functions
        base_agent.add_tool(add_func)
        base_agent.add_tool(mult_func)
        base_agent.add_tool(pow_func)
        
        # Inject into REPL
        base_agent._inject_custom_functions_to_repl()
        
        # Verify all are in namespace
        assert 'add_numbers' in _persistent_namespace
        assert 'multiply_numbers' in _persistent_namespace
        assert 'power' in _persistent_namespace
        
        # Verify they're callable
        assert _persistent_namespace['add_numbers'](5, 3) == 8
        assert _persistent_namespace['multiply_numbers'](5, 3) == 15
        assert _persistent_namespace['power'](5, 2) == 25


class TestPromptGeneration:
    """Test system prompt generation with custom tools."""
    
    def test_prompt_includes_tool(self, base_agent: BaseAgent):
        """Test that custom tools appear in generated prompts."""
        def search_database(query: str, limit: int = 10) -> str:
            """Search the database for matching records."""
            return f"Found {limit} results for: {query}"
        
        base_agent.add_tool(search_database)
        
        # Check if it's in the system prompt
        assert 'search_database' in base_agent.system_prompt
    
    def test_prompt_includes_multiple_tools(self, base_agent: BaseAgent, math_functions: tuple):
        """Test that multiple tools appear in the system prompt."""
        add_func, mult_func, pow_func = math_functions
        
        base_agent.add_tool(add_func)
        base_agent.add_tool(mult_func)
        base_agent.add_tool(pow_func)
        
        # All tools should be in prompt
        assert 'add_numbers' in base_agent.system_prompt
        assert 'multiply_numbers' in base_agent.system_prompt
        assert 'power' in base_agent.system_prompt


class TestSelectionState:
    """Test tool selection state management."""
    
    def test_selection_state(self, base_agent: BaseAgent):
        """Test that custom tools respect the selection state."""
        # Add multiple custom functions
        def func1(x: int) -> int:
            """Function 1."""
            return x * 2
        
        def func2(x: int) -> int:
            """Function 2."""
            return x * 3
        
        base_agent.add_tool(func1)
        base_agent.add_tool(func2)
        
        # Check initial state
        all_tools = base_agent.resource_manager.collection.custom_tools
        assert len(all_tools) == 2
        assert all(tool.selected for tool in all_tools)
        
        # Test deselection
        for tool in all_tools:
            if tool.name == "func1":
                tool.selected = False
        
        # Get selected tools
        selected = base_agent.resource_manager.get_selected_tools()
        custom_selected = [t for t in selected if t in base_agent.resource_manager.collection.custom_tools]
        
        assert len(custom_selected) == 1
        assert custom_selected[0].name == "func2"
    
    def test_all_selected_by_default(self, base_agent: BaseAgent, math_functions: tuple):
        """Test that all added tools are selected by default."""
        add_func, mult_func, pow_func = math_functions
        
        base_agent.add_tool(add_func)
        base_agent.add_tool(mult_func)
        base_agent.add_tool(pow_func)
        
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 3
        assert all(tool.selected for tool in custom_tools)


class TestSchemaGeneration:
    """Test API schema generation from functions."""
    
    def test_schema_structure(self, base_agent: BaseAgent, typed_function: Callable):
        """Test that generated schema has correct structure."""
        schema = base_agent.add_tool(typed_function)
        
        # Verify schema structure
        assert 'name' in schema
        assert 'description' in schema
        assert 'required_parameters' in schema
        assert 'optional_parameters' in schema
        
        assert schema['name'] == 'calculate_sum'
        assert len(schema['required_parameters']) == 2
        assert len(schema['optional_parameters']) == 1
    
    def test_parameter_details(self, base_agent: BaseAgent, typed_function: Callable):
        """Test that parameter details are captured correctly."""
        schema = base_agent.add_tool(typed_function)
        
        # Check required parameters
        req_params = {p['name']: p for p in schema['required_parameters']}
        assert 'x' in req_params
        assert 'y' in req_params
        assert req_params['x']['type'] == 'int'
        assert req_params['y']['type'] == 'int'
        
        # Check optional parameters
        opt_params = {p['name']: p for p in schema['optional_parameters']}
        assert 'verbose' in opt_params
        assert opt_params['verbose']['type'] == 'bool'
        assert opt_params['verbose']['default'] is False

