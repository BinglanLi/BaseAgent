"""
BaseAgent Configuration Management

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

    # Checkpointing
    checkpoint_db_path: str = ":memory:"  # ":memory:" preserves current ephemeral behavior

    # Interrupt/approval policy
    require_approval: str = "always"  # "always" | "never"

    # Skills
    skills_directory: str | None = None  # directory of SKILL.md files to load on startup

    # Context window management
    max_context_messages: int | None = None  # None = disabled; int >= 2 = sliding window size

    # Termination conditions (all None = disabled, no behavior change)
    max_iterations: int | None = None       # max think/execute cycles; None or >= 1
    max_cost: float | None = None           # per-run USD budget; None or > 0
                                            # has no effect when the LLM provider does not
                                            # return cost metadata (e.g. Anthropic returns None)
    max_consecutive_errors: int | None = None  # infra failures before forced stop; None or >= 2

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
        if os.getenv("BASE_AGENT_TEMPERATURE"):
            self.temperature = float(os.getenv("BASE_AGENT_TEMPERATURE"))
        if os.getenv("BASE_AGENT_CUSTOM_BASE_URL"):
            self.base_url = os.getenv("BASE_AGENT_CUSTOM_BASE_URL")
        if os.getenv("BASE_AGENT_CUSTOM_API_KEY"):
            self.api_key = os.getenv("BASE_AGENT_CUSTOM_API_KEY")
        if os.getenv("BASE_AGENT_SOURCE"):
            self.source = os.getenv("BASE_AGENT_SOURCE")
        if os.getenv("BASE_AGENT_CHECKPOINT_DB_PATH"):
            self.checkpoint_db_path = os.getenv("BASE_AGENT_CHECKPOINT_DB_PATH")
        if os.getenv("BASE_AGENT_REQUIRE_APPROVAL"):
            val = os.getenv("BASE_AGENT_REQUIRE_APPROVAL").lower()
            if val not in ("always", "never"):
                raise ValueError(
                    f"BASE_AGENT_REQUIRE_APPROVAL must be 'always' or 'never', got '{val}'"
                )
            self.require_approval = val
        if os.getenv("BASE_AGENT_SKILLS_DIRECTORY"):
            self.skills_directory = os.getenv("BASE_AGENT_SKILLS_DIRECTORY")
        if os.getenv("BASE_AGENT_MAX_CONTEXT_MESSAGES"):
            try:
                val = int(os.getenv("BASE_AGENT_MAX_CONTEXT_MESSAGES"))
            except ValueError:
                raise ValueError("BASE_AGENT_MAX_CONTEXT_MESSAGES must be an integer")
            if val < 2:
                raise ValueError("BASE_AGENT_MAX_CONTEXT_MESSAGES must be >= 2")
            self.max_context_messages = val
        if self.max_context_messages is not None and self.max_context_messages < 2:
            raise ValueError("max_context_messages must be None or >= 2")
        if os.getenv("BASE_AGENT_MAX_ITERATIONS"):
            try:
                val = int(os.getenv("BASE_AGENT_MAX_ITERATIONS"))
            except ValueError:
                raise ValueError("BASE_AGENT_MAX_ITERATIONS must be an integer")
            if val < 1:
                raise ValueError("BASE_AGENT_MAX_ITERATIONS must be >= 1")
            self.max_iterations = val
        if self.max_iterations is not None and self.max_iterations < 1:
            raise ValueError("max_iterations must be None or >= 1")
        if os.getenv("BASE_AGENT_MAX_COST"):
            try:
                val = float(os.getenv("BASE_AGENT_MAX_COST"))
            except ValueError:
                raise ValueError("BASE_AGENT_MAX_COST must be a number")
            if val <= 0:
                raise ValueError("BASE_AGENT_MAX_COST must be > 0")
            self.max_cost = val
        if self.max_cost is not None and self.max_cost <= 0:
            raise ValueError("max_cost must be None or > 0")
        if os.getenv("BASE_AGENT_MAX_CONSECUTIVE_ERRORS"):
            try:
                val = int(os.getenv("BASE_AGENT_MAX_CONSECUTIVE_ERRORS"))
            except ValueError:
                raise ValueError("BASE_AGENT_MAX_CONSECUTIVE_ERRORS must be an integer")
            if val < 2:
                raise ValueError("BASE_AGENT_MAX_CONSECUTIVE_ERRORS must be >= 2")
            self.max_consecutive_errors = val
        if self.max_consecutive_errors is not None and self.max_consecutive_errors < 2:
            raise ValueError("max_consecutive_errors must be None or >= 2")

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
            "checkpoint_db_path": self.checkpoint_db_path,
            "require_approval": self.require_approval,
            "skills_directory": self.skills_directory,
            "max_context_messages": self.max_context_messages,
            "max_iterations": self.max_iterations,
            "max_cost": self.max_cost,
            "max_consecutive_errors": self.max_consecutive_errors,
        }


# Global default config instance (optional, for convenience)
default_config = BaseAgentConfig()