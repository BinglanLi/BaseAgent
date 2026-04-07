"""
MCP Server Integration Example
===============================

This example shows how to integrate Model Context Protocol (MCP) servers.
BaseAgent supports both local (stdio) and remote (Streamable HTTP) transports.
"""

from BaseAgent import BaseAgent

# --- YAML config format ---
#
# mcp_servers:
#   # Local server (stdio transport) - launches a subprocess
#   filesystem:
#     command: ["npx", "-y", "@modelcontextprotocol/server-filesystem", "./"]
#
#   # Remote server (Streamable HTTP transport) - connects to a URL
#   biocontext:
#     url: https://mcp.biocontext.ai/mcp/
#     type: remote
#
#   # Remote server with auth headers (${ENV_VAR} interpolation)
#   private_api:
#     url: https://mcp.example.com/mcp/
#     headers:
#       Authorization: "Bearer ${MY_API_KEY}"

# Initialize agent
agent = BaseAgent()

# Add MCP server tools from config (supports both local and remote servers)
agent.add_mcp("./examples/mcp_config.yaml")

# Print all discovered tools
print("Discovered MCP tools:")
for tool in agent.resource_manager.get_all_tools():
    print(f"  - {tool.name} ({tool.module})")

# Now you can use MCP server tools in agent tasks
# log, answer = agent.run("List all files in the directory and show their sizes")

