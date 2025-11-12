"""
MCP Server Integration Example
===============================

This example shows how to integrate Model Context Protocol (MCP) servers.
"""

from BaseAgent import BaseAgent

# First, create a config file: mcp_config.yaml
# 
# mcpServers:
#   filesystem:
#     command: "npx"
#     args: ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/files"]
#   
#   sqlite:
#     command: "npx"
#     args: ["-y", "@modelcontextprotocol/server-sqlite", "/path/to/database.db"]

# Initialize agent
agent = BaseAgent()

# Add MCP server tools from config
agent.add_mcp("./examples/mcp_config.yaml")

# Print all tools
print("Here are all the tools you've added:")
for tool in agent.resource_manager.get_all_tools():
    print(tool.name)

# Now you can use MCP server tools
# result = agent.go("List all files in the directory and show their sizes")
# print(result)

