"""
Basic Usage Example
===================

This example shows how to initialize and use BaseAgent with a simple task.
"""

from BaseAgent import BaseAgent

# Initialize the agent
agent = BaseAgent(llm="claude-sonnet-4-20250514")

# run() returns (log, content)
log, result = agent.run("Compute the sum of 1 through 10 in Python.")
print(result)

# Access usage metrics — agent.usage_metrics accumulates across all runs.
# Note: cost may be None for providers that do not return it in streaming events (e.g. Anthropic).
total_cost = sum(u.cost for u in agent.usage_metrics if u.cost is not None)
total_input = sum(u.input_tokens for u in agent.usage_metrics if u.input_tokens is not None)
total_output = sum(u.output_tokens for u in agent.usage_metrics if u.output_tokens is not None)
total_cache_creation = sum(u.cache_creation_tokens for u in agent.usage_metrics if u.cache_creation_tokens is not None)
total_cache_read = sum(u.cache_read_tokens for u in agent.usage_metrics if u.cache_read_tokens is not None)
print(f"\nUsage Metrics:")
print(f"Input tokens:         {total_input}")
print(f"Output tokens:        {total_output}")
print(f"Cache creation tokens:{total_cache_creation}")
print(f"Cache read tokens:    {total_cache_read}")
print(f"Total cost:           ${total_cost:.4f} (may not reflect actual expense for streaming)")
