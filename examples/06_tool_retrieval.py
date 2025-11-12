"""
Tool Retrieval Example
=======================

This example shows how to use automatic tool selection based on task requirements.
"""

from BaseAgent import BaseAgent

# Enable automatic tool selection/retrieval
agent = BaseAgent(
    llm="gpt-4",
    enable_retrieval=True  # Agent will automatically select relevant tools
)

# The agent will automatically determine which tools are needed
result = agent.go(
    "Analyze protein sequences, calculate binding affinities, "
    "and create a visualization of the results"
)

print(result)

# You can also manually select tools
agent.resource_manager.select_tools_by_names([
    "run_python_repl",
    "fetch_data"
])

# Only selected tools will be available for this task
result = agent.go("Run a Python analysis on the data")
print(result)

