"""
Custom Tools Example
====================

This example shows how to add custom tools to your agent.
"""

from BaseAgent import BaseAgent


def search_database(query: str, limit: int = 10) -> list:
    """Search the database for relevant entries."""
    results = [
        {"id": 1, "name": "Result 1", "relevance": 0.95},
        {"id": 2, "name": "Result 2", "relevance": 0.87},
    ]
    return results[:limit]


def calculate_metrics(data: list, metric_type: str = "mean") -> float:
    """Calculate statistical metrics from data."""
    if metric_type == "mean":
        return sum(data) / len(data) if data else 0
    elif metric_type == "sum":
        return sum(data)
    else:
        return 0


agent = BaseAgent()

# add_tool() takes a callable; name, description, and schema are derived
# automatically from the function's name, docstring, and type hints.
agent.add_tool(search_database)
agent.add_tool(calculate_metrics)

log, result = agent.run("Search for proteins related to cancer and calculate the mean relevance")
print(result)
