"""
Resource Management and Tool Retrieval
=======================================

BaseAgent exposes a ResourceManager that tracks data sources, libraries, and
tools.  You can register custom data and libraries so the agent can reference
them when planning tasks, and optionally enable automatic tool selection.

DataLakeItem  — a dataset the agent can read (CSV, Parquet, etc.)
Library       — a Python/R package or CLI tool the agent can use
use_tool_retriever — when True, the agent embeds available tools and selects
                     relevant ones by semantic similarity before each run,
                     instead of passing the full tool list.
"""

from BaseAgent import BaseAgent
from BaseAgent.resources import DataLakeItem, Library

# ---------------------------------------------------------------------------
# 1. Register data sources and libraries
# ---------------------------------------------------------------------------
agent = BaseAgent()

agent.resource_manager.add_data_item(
    DataLakeItem(
        filename="my_dataset.csv",
        description="Custom dataset for protein analysis",
        format="csv",
        category="research",
        path="/path/to/my_dataset.csv",
    )
)

agent.resource_manager.add_data_item(
    DataLakeItem(
        filename="experiment_results.parquet",
        description="Results from experiment batch #42",
        format="parquet",
        category="experiments",
        size_mb=125.5,
        path="/path/to/experiment_results.parquet",
    )
)

agent.resource_manager.add_library(
    Library(
        name="custom_analysis_lib",
        description="Custom library for advanced protein analysis",
        type="Python",
        version="2.1.0",
        category="analysis",
        installation_cmd="pip install custom-analysis-lib",
    )
)

# Query the registered resources
print("=== Available Data Sources ===")
for data in agent.resource_manager.get_all_data():
    print(f"- {data.filename}: {data.description}")

print("\n=== Python Libraries ===")
for lib in agent.resource_manager.filter_libraries_by_type("Python")[:5]:
    print(f"- {lib.name} ({lib.version}): {lib.description}")

summary = agent.resource_manager.get_summary()
print(f"\n=== Resource Summary ===")
print(f"Tools: {summary['tools']['total']}")
print(f"Data Sources: {summary['data']['total']}")
print(f"Libraries: {summary['libraries']['total']}")

_, result = agent.run("Analyse my_dataset.csv using custom_analysis_lib")
print(f"\n=== Agent Result ===\n{result}")

# ---------------------------------------------------------------------------
# 2. Automatic tool selection (tool retriever)
# ---------------------------------------------------------------------------
# With use_tool_retriever=True the agent ranks tools by semantic similarity
# to the task and passes only the top matches to the LLM — useful when the
# full tool list is large.
retriever_agent = BaseAgent(
    llm="claude-sonnet-4-20250514",
    use_tool_retriever=True,
)

_, result = retriever_agent.run(
    "Analyze protein sequences and calculate binding affinities."
)
print(result)

# You can also pin a specific tool subset for a single run.
retriever_agent.resource_manager.select_tools_by_names(["run_python_repl"])
_, result = retriever_agent.run("Run a quick Python calculation.")
print(result)
