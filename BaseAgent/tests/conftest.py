"""
Pytest configuration and shared fixtures for BaseAgent tests.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable
from dotenv import load_dotenv

import pytest

# Load environment variables first
env_path = Path(__file__).resolve().parents[2] / ".env"
if env_path.exists():
    load_dotenv(env_path, override=True)

# Add parent directory to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BaseAgent.base_agent import BaseAgent
from BaseAgent.tools.support_tools import _persistent_namespace


@pytest.fixture
def base_agent() -> BaseAgent:
    """Create a fresh BaseAgent instance for testing."""
    return BaseAgent()


@pytest.fixture
def sample_function() -> Callable:
    """Provide a simple test function."""
    def greet(name: str, greeting: str = "Hello") -> str:
        """Greet someone by name."""
        return f"{greeting}, {name}!"
    return greet


@pytest.fixture
def typed_function() -> Callable:
    """Provide a well-typed function for testing."""
    def calculate_sum(x: int, y: int, verbose: bool = False) -> int:
        """
        Calculate the sum of two numbers.
        
        Args:
            x: First number to add
            y: Second number to add
            verbose: Whether to print debug information
            
        Returns:
            The sum of x and y
        """
        if verbose:
            print(f"Adding {x} + {y}")
        return x + y
    return calculate_sum


@pytest.fixture
def complex_function() -> Callable:
    """Provide a function with multiple parameter types."""
    def analyze_data(
        data: list,
        threshold: float = 0.5,
        normalize: bool = True,
        top_k: int = 10
    ):
        """Analyze numerical data with various options.
        
        Args:
            data: List of numerical values to analyze
            threshold: Minimum value threshold for filtering
            normalize: Whether to normalize the data
            top_k: Number of top values to return
        """
        if normalize and data:
            max_val = max(data)
            data = [x / max_val for x in data]
        
        filtered = [x for x in data if x > threshold]
        return sorted(filtered, reverse=True)[:top_k]
    return analyze_data


@pytest.fixture
def mcp_config_path() -> Path:
    """Return path to MCP test config file."""
    return Path(__file__).parent / "test_mcp_config.yaml"


@pytest.fixture
def clear_repl_namespace():
    """Clear REPL namespace before and after tests."""
    # Clear before test
    _persistent_namespace.clear()
    
    yield
    
    # Clear after test
    _persistent_namespace.clear()


@pytest.fixture
def math_functions() -> tuple[Callable, Callable, Callable]:
    """Provide multiple math functions for testing."""
    def add_numbers(a: float, b: float) -> float:
        """Add two numbers."""
        return a + b
    
    def multiply_numbers(a: float, b: float) -> float:
        """Multiply two numbers."""
        return a * b
    
    def power(base: float, exponent: float = 2.0) -> float:
        """Raise base to the power of exponent."""
        return base ** exponent
    
    return add_numbers, multiply_numbers, power


@pytest.fixture
def temp_file(tmp_path: Path):
    """Provide a temporary file path for testing."""
    def _create_file(filename: str, content: str = "") -> Path:
        file_path = tmp_path / filename
        file_path.write_text(content)
        return file_path
    return _create_file

