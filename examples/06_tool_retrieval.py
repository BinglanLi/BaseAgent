"""
Tool Retrieval Example
=======================

This example shows how to use automatic tool selection based on task requirements.
"""

from BaseAgent import BaseAgent

# Enable automatic tool selection: the agent embeds and ranks tools
# by relevance before each run instead of passing the full tool list.
agent = BaseAgent(
    llm="claude-sonnet-4-20250514",
    use_tool_retriever=True,
)

log, result = agent.run(
    "Analyze protein sequences, calculate binding affinities, "
    "and create a visualization of the results"
)
print(result)

# You can also manually select a subset of tools for a specific task.
agent.resource_manager.select_tools_by_names([
    "run_python_repl",
    "fetch_data",
])

log, result = agent.run("Run a Python analysis on the data")
print(result)
