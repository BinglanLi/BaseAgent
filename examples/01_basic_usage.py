"""
Basic Usage Example
===================

This example shows how to initialize and use BaseAgent with a simple task.
"""

from BaseAgent import BaseAgent

# Initialize the agent
agent = BaseAgent(llm="claude-sonnet-4-20250514")

# run() returns (log, content)
log, result = agent.run("Analyze the dataset and create a visualization")
print(result)

# Access usage metrics — _usage_metrics accumulates across all runs
total_cost = sum(u.cost for u in agent._usage_metrics if u.cost is not None)
total_input = sum(u.input_tokens for u in agent._usage_metrics if u.input_tokens is not None)
total_output = sum(u.output_tokens for u in agent._usage_metrics if u.output_tokens is not None)
print(f"\nUsage Metrics:")
print(f"Input tokens:  {total_input}")
print(f"Output tokens: {total_output}")
print(f"Total cost:    ${total_cost:.4f}")
