"""
Resource Management Example
============================

This example shows how to manage tools, data sources, and libraries.
"""

from BaseAgent import BaseAgent
from BaseAgent.resources import DataLakeItem, Library

# Initialize agent
agent = BaseAgent()

# Add custom data source
agent.resource_manager.add_data_item(
    DataLakeItem(
        filename="my_dataset.csv",
        description="Custom dataset for protein analysis",
        format="csv",
        category="research",
        path="/path/to/my_dataset.csv"
    )
)

# Add another data source
agent.resource_manager.add_data_item(
    DataLakeItem(
        filename="experiment_results.parquet",
        description="Results from experiment batch #42",
        format="parquet",
        category="experiments",
        size_mb=125.5,
        path="/path/to/experiment_results.parquet"
    )
)

# Add custom library
agent.resource_manager.add_library(
    Library(
        name="custom_analysis_lib",
        description="Custom library for advanced protein analysis",
        type="Python",
        version="2.1.0",
        category="analysis",
        installation_cmd="pip install custom-analysis-lib"
    )
)

# Query resources
print("=== Available Data Sources ===")
all_data = agent.resource_manager.get_all_data()
for data in all_data:
    print(f"- {data.filename}: {data.description}")

print("\n=== Research Category Data ===")
research_data = agent.resource_manager.filter_data_by_category("research")
for data in research_data:
    print(f"- {data.filename}")

print("\n=== Python Libraries ===")
python_libs = agent.resource_manager.filter_libraries_by_type("Python")
for lib in python_libs[:5]:  # Show first 5
    print(f"- {lib.name} ({lib.version}): {lib.description}")

# Get summary
summary = agent.resource_manager.get_summary()
print(f"\n=== Resource Summary ===")
print(f"Tools: {summary['tools']['total']}")
print(f"Data Sources: {summary['data']['total']}")
print(f"Libraries: {summary['libraries']['total']}")

# Use the agent with custom resources
result = agent.go("Analyze my_dataset.csv using custom_analysis_lib")
print(f"\n=== Agent Result ===")
print(result)

