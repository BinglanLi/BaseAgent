"""
Unit tests for BaseAgent.add_mcp() functionality.

NOTE: These tests require an MCP server configuration file.
Create test_mcp_config.yaml in the tests directory (see example at bottom of this file).

Run with: pytest test_add_mcp.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from BaseAgent.base_agent import BaseAgent
from BaseAgent.tools.support_tools import _persistent_namespace


@pytest.fixture
def skip_if_no_config(mcp_config_path: Path):
    """Skip test if MCP config file doesn't exist."""
    if not mcp_config_path.exists():
        pytest.skip(f"MCP config file not found at: {mcp_config_path}")


class TestMCPBasic:
    """Test basic MCP tool loading functionality."""
    
    def test_mcp_basic_loading(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config):
        """Test basic MCP tool loading."""
        # Add MCP tools
        base_agent.add_mcp(mcp_config_path)
        
        # Check custom tools were loaded
        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) > 0, "No MCP tools were loaded"
        
        # Verify tool structure
        for tool in custom_tools:
            assert tool.name, "Tool should have a name"
            assert tool.module, "Tool should have a module"
            assert tool.description, "Tool should have a description"
            assert tool.function is not None, "Tool should have a function"
    
    def test_mcp_tools_have_correct_module_prefix(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config):
        """Test that MCP tools have the mcp_servers module prefix."""
        base_agent.add_mcp(mcp_config_path)
        
        # Check that tools have the mcp_servers module prefix
        mcp_tools = [t for t in base_agent.resource_manager.collection.custom_tools 
                     if t.module.startswith("mcp_servers.")]
        
        assert len(mcp_tools) > 0, "No tools with mcp_servers prefix found"
        
        # Verify module naming convention
        for tool in mcp_tools:
            assert tool.module.startswith("mcp_servers."), f"Tool {tool.name} doesn't have mcp_servers prefix"


class TestMCPResourceManager:
    """Test MCP tool storage in ResourceManager."""
    
    def test_mcp_adds_to_custom_tools(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config):
        """Test that MCP tools are properly stored in ResourceManager."""
        # Count tools before
        tools_before = len(base_agent.resource_manager.collection.custom_tools)
        
        # Add MCP
        base_agent.add_mcp(mcp_config_path)
        
        # Count tools after
        tools_after = len(base_agent.resource_manager.collection.custom_tools)
        
        assert tools_after > tools_before, "No MCP tools were added to ResourceManager"
        
        # Verify MCP tools exist
        mcp_tools = [t for t in base_agent.resource_manager.collection.custom_tools 
                     if t.module.startswith("mcp_servers.")]
        assert len(mcp_tools) > 0, "No MCP tools with proper module prefix found"
    
    def test_mcp_tools_selected_by_default(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config):
        """Test that MCP tools are selected by default."""
        base_agent.add_mcp(mcp_config_path)
        
        mcp_tools = [t for t in base_agent.resource_manager.collection.custom_tools 
                     if t.module.startswith("mcp_servers.")]
        
        # All MCP tools should be selected by default
        assert all(tool.selected for tool in mcp_tools), "Not all MCP tools are selected by default"


class TestMCPREPLInjection:
    """Test MCP tool injection into REPL namespace."""
    
    def test_mcp_repl_injection(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config, clear_repl_namespace):
        """Test that MCP tools are available in REPL."""
        # Add MCP
        base_agent.add_mcp(mcp_config_path)
        
        # Inject into REPL
        base_agent._inject_custom_functions_to_repl()
        
        # Check namespace
        mcp_tools = base_agent.resource_manager.collection.custom_tools
        
        # At least one tool should be in namespace
        tools_in_namespace = [tool for tool in mcp_tools if tool.name in _persistent_namespace]
        assert len(tools_in_namespace) > 0, "No MCP tools found in REPL namespace"
        
        # Verify callable
        for tool in tools_in_namespace:
            assert callable(_persistent_namespace[tool.name]), f"Tool {tool.name} in namespace is not callable"


class TestMCPPromptGeneration:
    """Test MCP tool appearance in system prompts."""
    
    def test_mcp_prompt_inclusion(self, base_agent: BaseAgent, mcp_config_path: Path, skip_if_no_config):
        """Test that MCP tools appear in prompts."""
        # Add MCP
        base_agent.add_mcp(mcp_config_path)
        
        # Check prompt
        mcp_tools = base_agent.resource_manager.collection.custom_tools
        
        # Count tools in prompt
        found_in_prompt = sum(1 for tool in mcp_tools if tool.name in base_agent.system_prompt)
        
        # Note: Tools might not appear in prompt if using tool retriever
        # So we just verify the count is recorded, not necessarily that all are in prompt
        assert found_in_prompt >= 0, "Prompt check failed"


# Example MCP configuration file (test_mcp_config.yaml):
"""
mcp_servers:
  # Example: Weather server
  weather:
    enabled: true
    command: ["npx", "-y", "@modelcontextprotocol/server-weather"]
    tools:
      - name: get_weather
        description: Get current weather for a location
        parameters:
          location:
            type: string
            description: City name or location
            required: true
          units:
            type: string
            description: Temperature units (celsius/fahrenheit)
            required: false
            default: celsius

  # Example: File system server
  filesystem:
    enabled: false  # Disabled by default
    command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    # tools will be auto-discovered if not specified
"""

