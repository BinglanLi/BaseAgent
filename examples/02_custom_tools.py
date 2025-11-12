"""
Custom Tools Example
====================

This example shows how to add custom tools to your agent.
"""

from BaseAgent import BaseAgent


def search_database(query: str, limit: int = 10) -> list:
    """Search the database for relevant entries."""
    # Your implementation here
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


# Create agent
agent = BaseAgent()

# Add first custom tool
agent.add_tool(
    name="search_database",
    function=search_database,
    description="Search the database for relevant entries",
    required_parameters=[
        {"name": "query", "description": "Search query", "type": "str"}
    ],
    optional_parameters=[
        {"name": "limit", "description": "Maximum results", "type": "int", "default": 10}
    ]
)

# Add second custom tool
agent.add_tool(
    name="calculate_metrics",
    function=calculate_metrics,
    description="Calculate statistical metrics from numerical data",
    required_parameters=[
        {"name": "data", "description": "List of numbers", "type": "list"}
    ],
    optional_parameters=[
        {"name": "metric_type", "description": "Type of metric (mean, sum)", "type": "str", "default": "mean"}
    ]
)

# Use the agent with your custom tools
result = agent.go("Search for proteins related to cancer and calculate the mean relevance")
print(result)

