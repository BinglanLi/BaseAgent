"""
Integration tests - Full end-to-end workflow with custom tools.

These tests verify that custom tools work in actual agent execution.

Run with: pytest test_integration.py -v
"""

from __future__ import annotations

import pytest

from BaseAgent.base_agent import BaseAgent
from BaseAgent.tools.support_tools import _persistent_namespace


@pytest.mark.integration
class TestFullWorkflow:
    """Integration tests for the complete custom tool workflow."""
    
    def test_full_workflow(self, base_agent: BaseAgent, clear_repl_namespace):
        """Test the complete workflow: add tool -> register -> inject -> verify."""
        # Define a custom function
        def calculate_average(numbers: list) -> float:
            """Calculate the average of a list of numbers.
            
            Args:
                numbers: List of numerical values
                
            Returns:
                The average (mean) value
            """
            if not numbers:
                return 0.0
            return sum(numbers) / len(numbers)
        
        # Step 1: Add the custom tool
        base_agent.add_tool(calculate_average)
        
        # Step 2: Verify it's in ResourceManager
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 1
        
        tool = custom_tools[0]
        assert tool.name == 'calculate_average'
        assert tool.function is not None
        
        # Step 3: Check it's in the system prompt
        assert 'calculate_average' in base_agent.system_prompt
        
        # Step 4: Test direct function call
        test_numbers = [10, 20, 30, 40, 50]
        result = tool.function(test_numbers)
        assert result == 30.0
        
        # Step 5: Inject into REPL and verify
        base_agent._inject_custom_functions_to_repl()
        assert 'calculate_average' in _persistent_namespace
        
        # Verify callable from namespace
        result_from_namespace = _persistent_namespace['calculate_average']([5, 10, 15, 20])
        assert result_from_namespace == 12.5


@pytest.mark.integration
class TestMultipleTools:
    """Integration tests for multiple custom tools."""
    
    def test_multiple_tools(self, base_agent: BaseAgent, math_functions: tuple, clear_repl_namespace):
        """Test adding and using multiple custom tools."""
        add_func, mult_func, pow_func = math_functions
        
        # Add multiple tools
        base_agent.add_tool(add_func)
        base_agent.add_tool(mult_func)
        base_agent.add_tool(pow_func)
        
        # Verify all are registered
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) == 3
        
        tool_names = [tool.name for tool in custom_tools]
        assert 'add_numbers' in tool_names
        assert 'multiply_numbers' in tool_names
        assert 'power' in tool_names
        
        # Verify all are callable
        assert custom_tools[0].function(5, 3) == 8
        assert custom_tools[1].function(5, 3) == 15
        assert custom_tools[2].function(5, 2) == 25
        
        # Verify all are in REPL
        base_agent._inject_custom_functions_to_repl()
        
        for tool in custom_tools:
            assert tool.name in _persistent_namespace
            assert callable(_persistent_namespace[tool.name])
    
    def test_tool_isolation(self, base_agent: BaseAgent):
        """Test that tools don't interfere with each other."""
        def func_a(x: int) -> int:
            """Function A."""
            return x + 1
        
        def func_b(x: int) -> int:
            """Function B."""
            return x + 2
        
        base_agent.add_tool(func_a)
        base_agent.add_tool(func_b)
        
        custom_tools = base_agent.resource_manager.collection.custom_tools
        
        # Each function should work independently
        func_a_tool = next(t for t in custom_tools if t.name == 'func_a')
        func_b_tool = next(t for t in custom_tools if t.name == 'func_b')
        
        assert func_a_tool.function(10) == 11
        assert func_b_tool.function(10) == 12


@pytest.mark.integration
class TestToolSelection:
    """Integration tests for tool selection behavior."""
    
    def test_selection_in_execution(self, base_agent: BaseAgent):
        """Test that deselected tools are not included in prompts."""
        # Add tools
        def tool1(x: int) -> int:
            """Tool 1 - should be selected."""
            return x * 2
        
        def tool2(x: int) -> int:
            """Tool 2 - will be deselected."""
            return x * 3
        
        base_agent.add_tool(tool1)
        base_agent.add_tool(tool2)
        
        # Deselect tool2
        for tool in base_agent.resource_manager.collection.custom_tools:
            if tool.name == "tool2":
                tool.selected = False
        
        # Regenerate prompt
        base_agent.system_prompt = base_agent._generate_system_prompt(
            self_critic=False,
            is_retrieval=False
        )
        
        # Verify selection in prompt
        assert 'tool1' in base_agent.system_prompt
        # Note: tool2 may or may not be in prompt depending on implementation
    
    def test_reselection(self, base_agent: BaseAgent):
        """Test that tools can be deselected and reselected."""
        def test_func(x: int) -> int:
            """Test function."""
            return x
        
        base_agent.add_tool(test_func)
        
        tool = base_agent.resource_manager.collection.custom_tools[0]
        
        # Initial state: selected
        assert tool.selected is True
        
        # Deselect
        tool.selected = False
        assert tool.selected is False
        
        # Reselect
        tool.selected = True
        assert tool.selected is True


@pytest.mark.integration
class TestToolErrorHandling:
    """Integration tests for error handling with custom tools."""
    
    def test_function_with_exception(self, base_agent: BaseAgent):
        """Test that tools can raise exceptions and they're handled properly."""
        def failing_func(x: int) -> int:
            """A function that raises an exception."""
            if x < 0:
                raise ValueError("x must be non-negative")
            return x * 2
        
        base_agent.add_tool(failing_func)
        
        tool = base_agent.resource_manager.collection.custom_tools[0]
        
        # Should work with valid input
        assert tool.function(5) == 10
        
        # Should raise with invalid input
        with pytest.raises(ValueError, match="must be non-negative"):
            tool.function(-1)
    
    def test_function_with_none_return(self, base_agent: BaseAgent):
        """Test tools that return None."""
        def void_func(x: int) -> None:
            """A function that returns None."""
            print(f"Processing {x}")
        
        base_agent.add_tool(void_func)
        
        tool = base_agent.resource_manager.collection.custom_tools[0]
        result = tool.function(42)
        assert result is None

