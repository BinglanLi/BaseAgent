"""Unit tests for BaseAgent.utils.schema module."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest

from BaseAgent.utils.schema import (
    _parse_docstring,
    _type_to_string,
    extract_schema_from_function,
    function_to_api_schema,
)


class TestParseDocstring:
    """Tests for _parse_docstring()."""

    def test_google_style_args(self):
        docstring = """
        Do something useful.

        Args:
            x: First value
            y: Second value

        Returns:
            Result
        """
        description, params = _parse_docstring(docstring)
        assert "useful" in description
        assert "x" in params
        assert "y" in params
        assert "First" in params["x"]
        assert "Second" in params["y"]

    def test_empty_docstring(self):
        description, params = _parse_docstring("")
        assert description == ""
        assert params == {}

    def test_no_args_section(self):
        description, params = _parse_docstring("Just a simple description.")
        assert "simple description" in description
        assert params == {}

    def test_param_with_type_annotation(self):
        docstring = """
        Function description.

        Args:
            name (str): The name parameter
        """
        description, params = _parse_docstring(docstring)
        assert "name" in params
        assert "name parameter" in params["name"]

    def test_multiline_description(self):
        docstring = """
        First line of description.
        Second line of description.

        Args:
            x: A value
        """
        description, params = _parse_docstring(docstring)
        assert "First line" in description


class TestTypeToString:
    """Tests for _type_to_string()."""

    def test_int(self):
        assert _type_to_string(int) == "int"

    def test_str(self):
        assert _type_to_string(str) == "str"

    def test_float(self):
        assert _type_to_string(float) == "float"

    def test_bool(self):
        assert _type_to_string(bool) == "bool"

    def test_list_of_str(self):
        result = _type_to_string(List[str])
        assert "list" in result.lower()

    def test_dict(self):
        result = _type_to_string(Dict[str, int])
        assert "dict" in result.lower()

    def test_optional(self):
        result = _type_to_string(Optional[str])
        assert result is not None

    def test_none_type(self):
        result = _type_to_string(type(None))
        assert result is not None


class TestExtractSchemaFromFunction:
    """Tests for extract_schema_from_function()."""

    def test_typed_function(self):
        def add(x: int, y: int, verbose: bool = False) -> int:
            """Add two numbers."""
            return x + y

        schema = extract_schema_from_function(add)
        assert schema["name"] == "add"
        assert len(schema["required_parameters"]) == 2
        assert len(schema["optional_parameters"]) == 1

    def test_untyped_function(self):
        def greet(name, greeting="Hello"):
            """Greet someone."""
            return f"{greeting}, {name}"

        schema = extract_schema_from_function(greet)
        assert schema["name"] == "greet"
        req_params = {p["name"]: p for p in schema["required_parameters"]}
        assert req_params["name"]["type"] == "Any"

    def test_no_params(self):
        def noop():
            """Do nothing."""
            pass

        schema = extract_schema_from_function(noop)
        assert schema["name"] == "noop"
        assert len(schema["required_parameters"]) == 0
        assert len(schema["optional_parameters"]) == 0

    def test_complex_types(self):
        def process(items: List[str], mapping: Dict[str, int]):
            """Process data."""
            pass

        schema = extract_schema_from_function(process)
        req = {p["name"]: p for p in schema["required_parameters"]}
        assert "list" in req["items"]["type"].lower()
        assert "dict" in req["mapping"]["type"].lower()


class TestFunctionToApiSchema:
    """Tests for function_to_api_schema() — returns CustomTool Pydantic model."""

    def test_returns_custom_tool(self):
        def my_func(x: int) -> int:
            """Double a number."""
            return x * 2

        schema = function_to_api_schema(my_func, llm=None)
        assert schema.name == "my_func"
        assert schema.description is not None

    def test_required_and_optional(self):
        def calc(a: int, b: int, scale: float = 1.0):
            """Calculate something."""
            return (a + b) * scale

        schema = function_to_api_schema(calc, llm=None)
        assert len(schema.required_parameters) == 2
        assert len(schema.optional_parameters) == 1
        opt = schema.optional_parameters[0]
        assert opt.name == "scale"
        assert opt.default == 1.0

    def test_docstring_descriptions(self):
        def search(query: str, limit: int = 10):
            """
            Search for items matching the query.

            Args:
                query: The search term
                limit: Maximum results to return
            """
            pass

        schema = function_to_api_schema(search, llm=None)
        req = {p.name: p for p in schema.required_parameters}
        assert "search term" in req["query"].description.lower()

    def test_non_callable_raises(self):
        with pytest.raises((ValueError, TypeError, AttributeError)):
            function_to_api_schema("not a function", llm=None)
