"""
Unit tests for hybrid API schema generation.

This test verifies that the introspection-based schema extraction works correctly
with and without LLM enhancement.

Run with: pytest test_hybrid_schema.py -v
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from BaseAgent.utils import _parse_docstring, _type_to_string, function_to_api_schema


class TestSchemaExtraction:
    """Test schema extraction from various function types."""
    
    def test_introspection_with_typed_function(self):
        """Test schema extraction from a well-typed function."""
        def calculate_sum(x: int, y: int, verbose: bool = False) -> int:
            """
            Calculate the sum of two numbers.
            
            Args:
                x: First number to add
                y: Second number to add
                verbose: Whether to print debug information
                
            Returns:
                The sum of x and y
            """
            if verbose:
                print(f"Adding {x} + {y}")
            return x + y
        
        schema = function_to_api_schema(calculate_sum, llm=None)
        
        # Verify basic structure
        assert schema['name'] == 'calculate_sum'
        assert 'sum of two numbers' in schema['description'].lower()
        
        # Verify required parameters
        assert len(schema['required_parameters']) == 2
        req_params = {p['name']: p for p in schema['required_parameters']}
        assert 'x' in req_params
        assert 'y' in req_params
        assert req_params['x']['type'] == 'int'
        assert req_params['y']['type'] == 'int'
        assert 'first number' in req_params['x']['description'].lower()
        assert 'second number' in req_params['y']['description'].lower()
        
        # Verify optional parameters
        assert len(schema['optional_parameters']) == 1
        opt_params = {p['name']: p for p in schema['optional_parameters']}
        assert 'verbose' in opt_params
        assert opt_params['verbose']['type'] == 'bool'
        assert opt_params['verbose']['default'] is False


    def test_introspection_without_types(self):
        """Test schema extraction from a function without type hints."""
        def multiply(a, b, factor=1):
            """Multiply two numbers with an optional factor."""
            return a * b * factor
        
        schema = function_to_api_schema(multiply, llm=None)
        
        # Verify basic structure
        assert schema['name'] == 'multiply'
        assert len(schema['required_parameters']) == 2
        assert len(schema['optional_parameters']) == 1
        
        # Without type hints, types should be 'Any'
        req_params = {p['name']: p for p in schema['required_parameters']}
        assert req_params['a']['type'] == 'Any'
        assert req_params['b']['type'] == 'Any'
    
    def test_complex_types(self):
        """Test handling of complex type hints."""
        def process_data(
            items: List[str],
            mapping: Dict[str, int],
            config: Optional[Dict] = None
        ):
            """Process data with complex types."""
            pass
        
        schema = function_to_api_schema(process_data, llm=None)
        
        req_params = {p['name']: p for p in schema['required_parameters']}
        opt_params = {p['name']: p for p in schema['optional_parameters']}
        
        # Verify complex types are converted to strings correctly
        assert 'list' in req_params['items']['type'].lower()
        assert 'dict' in req_params['mapping']['type'].lower()
        assert 'config' in opt_params
    
    def test_string_input(self):
        """Test schema extraction from source code string."""
        function_code = """
def greet(name: str, greeting: str = "Hello"):
    '''Greet someone with a custom greeting.'''
    return f"{greeting}, {name}!"
"""
        
        schema = function_to_api_schema(function_code, llm=None)
        
        assert schema['name'] == 'greet'
        assert len(schema['required_parameters']) == 1
        assert len(schema['optional_parameters']) == 1
        assert schema['required_parameters'][0]['name'] == 'name'
        assert schema['optional_parameters'][0]['name'] == 'greeting'
        assert schema['optional_parameters'][0]['default'] == "Hello"
    
    def test_minimal_function(self):
        """Test with minimal function (no docstring, no types)."""
        def minimal(x):
            return x * 2
        
        schema = function_to_api_schema(minimal, llm=None)
        
        assert schema['name'] == 'minimal'
        assert len(schema['required_parameters']) == 1
        assert schema['required_parameters'][0]['name'] == 'x'
        # Should have a default description
        assert 'description' in schema


class TestDocstringParsing:
    """Test docstring parsing utilities."""
    
    def test_docstring_parsing(self):
        """Test docstring parsing functionality."""
        docstring = """
        This is a function that does something useful.
        It has multiple lines in the description.
        
        Args:
            param1: First parameter description
            param2 (str): Second parameter with type
            param3: Third parameter
            
        Returns:
            Something useful
        """
        
        description, params = _parse_docstring(docstring)
        
        assert "useful" in description
        assert "param1" in params
        assert "param2" in params
        assert "param3" in params
        assert "First parameter" in params["param1"]
        assert "Second parameter" in params["param2"]
    
    def test_empty_docstring(self):
        """Test parsing an empty docstring."""
        description, params = _parse_docstring("")
        
        assert description == ""
        assert params == {}
    
    def test_docstring_without_args(self):
        """Test parsing a docstring without Args section."""
        docstring = "Simple function description."
        
        description, params = _parse_docstring(docstring)
        
        assert "Simple function" in description
        assert params == {}


class TestTypeConversion:
    """Test type conversion utilities."""
    
    def test_basic_types(self):
        """Test conversion of basic Python types."""
        assert _type_to_string(int) == 'int'
        assert _type_to_string(str) == 'str'
        assert _type_to_string(float) == 'float'
        assert _type_to_string(bool) == 'bool'
    
    def test_list_type(self):
        """Test conversion of List types."""
        result = _type_to_string(List[str])
        assert 'list' in result.lower()
    
    def test_dict_type(self):
        """Test conversion of Dict types."""
        result = _type_to_string(Dict[str, int])
        assert 'dict' in result.lower()
    
    def test_optional_type(self):
        """Test conversion of Optional types."""
        result = _type_to_string(Optional[str])
        # Optional types should be handled gracefully
        assert result is not None


@pytest.mark.parametrize("func_name,has_types,expected_params", [
    ("simple_func", True, 2),
    ("no_types_func", False, 1),
])
class TestParametrizedSchemas:
    """Parametrized tests for schema generation."""
    
    def test_parameter_count(self, func_name: str, has_types: bool, expected_params: int):
        """Test that parameter counts are correct."""
        if func_name == "simple_func":
            def simple_func(x: int, y: int) -> int:
                """Simple function."""
                return x + y
            func = simple_func
        else:
            def no_types_func(x):
                """No types function."""
                return x
            func = no_types_func
        
        schema = function_to_api_schema(func, llm=None)
        assert len(schema['required_parameters']) == expected_params

