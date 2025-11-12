"""
Custom Configuration Example
=============================

This example shows how to customize agent configuration.
"""

from BaseAgent import BaseAgent
from BaseAgent.config import default_config

# View default configuration
print("Default Configuration:")
print(default_config)

# Customize configuration
custom_config = default_config.copy()
custom_config["max_iterations"] = 20
custom_config["temperature"] = 0.7
custom_config["verbose"] = True

# Initialize agent with custom config
agent = BaseAgent(
    llm="gpt-4",
    path="./workspace",
    config=custom_config
)

# Run a task with custom configuration
result = agent.go("Perform a complex multi-step analysis")
print(result)

