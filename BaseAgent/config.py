"""
BabyAgent Configuration Management

Simple configuration class for centralizing common settings.
Maintains full backward compatibility with existing code.

Modified from the biomni package
https://github.com/openbmb/Biomni
https://github.com/openbmb/Biomni/blob/main/biomni/config.py
"""

import os
from dataclasses import dataclass


@dataclass
class BaseAgentConfig:
    """Central configuration for the agent.

    All settings are optional and have sensible defaults.
    API keys are still read from environment variables to maintain
    compatibility with existing .env file structure.

    Usage:
        # Create config with defaults
        config = BaseAgentConfig()

        # Override specific settings
        config = BaseAgentConfig(llm="gpt-4", timeout_seconds=1200)

        # Modify after creation
        config.path = "./custom_data"
    """

    # Data and execution settings
    path: str = "./data"
    timeout_seconds: int = 600

    # LLM settings (API keys still from environment)
    llm: str = "claude-sonnet-4-20250514"
    temperature: float = 0.7

    # Tool settings
    use_tool_retriever: bool = False

    # Custom model settings (for custom LLM serving)
    base_url: str | None = None
    api_key: str | None = None  # Only for custom models, not provider API keys

    # LLM source (auto-detected if None)
    source: str | None = None

    def __post_init__(self):
        """Load any environment variable overrides if they exist."""
        # Check for environment variable overrides (optional)
        # Support both old and new names for backwards compatibility
        if os.getenv("BASE_AGENT_PATH") or os.getenv("BASE_AGENT_DATA_PATH"):
            self.path = os.getenv("BASE_AGENT_PATH") or os.getenv("BASE_AGENT_DATA_PATH")
        if os.getenv("BASE_AGENT_TIMEOUT_SECONDS"):
            self.timeout_seconds = int(os.getenv("BASE_AGENT_TIMEOUT_SECONDS"))
        if os.getenv("BASE_AGENT_LLM") or os.getenv("BASE_AGENT_LLM_MODEL"):
            self.llm = os.getenv("BASE_AGENT_LLM") or os.getenv("BASE_AGENT_LLM_MODEL")
        if os.getenv("BASE_AGENT_USE_TOOL_RETRIEVER"):
            self.use_tool_retriever = os.getenv("BASE_AGENT_USE_TOOL_RETRIEVER").lower() == "true"
        if os.getenv("BASE_AGENT_COMMERCIAL_MODE"):
            self.commercial_mode = os.getenv("BASE_AGENT_COMMERCIAL_MODE").lower() == "true"
        if os.getenv("BASE_AGENT_TEMPERATURE"):
            self.temperature = float(os.getenv("BASE_AGENT_TEMPERATURE"))
        if os.getenv("BASE_AGENT_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("BASE_AGENT_CUSTOM_BASE_URL")
        if os.getenv("BASE_AGENT_CUSTOM_API_KEY"):
            self.api_key = os.getenv("BASE_AGENT_CUSTOM_API_KEY")
        if os.getenv("BASE_AGENT_SOURCE"):
            self.source = os.getenv("BASE_AGENT_SOURCE")

    def to_dict(self) -> dict:
        """Convert config to dictionary for easy access."""
        return {
            "path": self.path,
            "timeout_seconds": self.timeout_seconds,
            "llm": self.llm,
            "temperature": self.temperature,
            "use_tool_retriever": self.use_tool_retriever,
            "base_url": self.base_url,
            "api_key": self.api_key,
            "source": self.source,
        }


# Global default config instance (optional, for convenience)
default_config = BaseAgentConfig()