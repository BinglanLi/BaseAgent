"""
Basic Usage Example
===================

This example shows how to initialize and use BaseAgent with a simple task.
"""

from BaseAgent import BaseAgent

# Initialize the agent with your preferred LLM
agent = BaseAgent(
    llm="gpt-4",  # or "claude-3-5-sonnet-20241022", "gemini-pro", etc.
    path="./data"
)

# Run a task
result = agent.go("Analyze the dataset and create a visualization")
print(result)

# Access usage metrics
print(f"\nUsage Metrics:")
print(f"Input tokens: {agent.total_input_tokens}")
print(f"Output tokens: {agent.total_output_tokens}")
print(f"Total cost: ${agent.total_cost:.4f}")

