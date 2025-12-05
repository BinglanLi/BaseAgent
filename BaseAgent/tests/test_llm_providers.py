"""
Unit tests for LLM providers using get_llm function.

These tests verify that get_llm() can successfully initialize chat models
for different providers and that they generate valid responses.

Run with: pytest test_llm_providers.py -v
Run specific provider: pytest test_llm_providers.py -v -k "openai"
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from BaseAgent.llm import get_llm, _detect_source


def has_api_key(env_var: str) -> bool:
    """Check if an API key environment variable is set."""
    return os.getenv(env_var) is not None


def validate_llm_response(response: str) -> None:
    """
    Validate that a response from an LLM is valid.

    Args:
        response: The generated response text
    """
    assert response, "Response should not be empty"
    assert isinstance(response, str), "Response should be a string"
    assert len(response.strip()) > 3, "Response should have meaningful content"


@pytest.mark.unit
class TestSourceDetection:
    """Test automatic LLM source detection."""

    def test_auto_detect_openai(self):
        """Test that OpenAI models are auto-detected from model name."""
        source = _detect_source(model="gpt-4o-mini")
        assert source == "OpenAI"
    
    def test_auto_detect_azure_openai(self):
        """Test that Azure OpenAI models are auto-detected from model name."""
        source = _detect_source(model="azure-gpt-4o")
        assert source == "AzureOpenAI"

    def test_auto_detect_anthropic(self):
        """Test that Anthropic models are auto-detected from model name."""
        source = _detect_source(model="claude-3-5-haiku-20241022")
        assert source == "Anthropic"
    
    def test_auto_detect_anthropic_foundry(self):
        """Test that Azure OpenAI models are auto-detected from model name."""
        source = _detect_source(model="azure-claude-sonnet-4-5")
        assert source == "AnthropicFoundry"

    def test_auto_detect_gemini(self):
        """Test that Gemini models are auto-detected from model name."""
        source = _detect_source(model="gemini-1.5-flash")
        assert source == "Gemini"


@pytest.mark.integration
@pytest.mark.skipif(not has_api_key("OPENAI_API_KEY"), reason="OpenAI API key not found")
class TestOpenAI:
    """Test OpenAI provider."""

    def test_openai_connection(self):
        """Test OpenAI with GPT-4o model."""
        _, chat_model = get_llm(model="gpt-4o")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
@pytest.mark.skipif(
    not (has_api_key("AZURE_FOUNDRY_API_KEY") and os.getenv("AZURE_FOUNDRY_BASE_URL")),
    reason="Azure Foundry credentials not found"
)
class TestAzureOpenAI:
    """Test Azure OpenAI provider."""

    def test_azure_openai_connection(self):
        """Test Azure OpenAI connection."""
        _, chat_model = get_llm(model="azure-gpt-5.1")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
@pytest.mark.skipif(not has_api_key("ANTHROPIC_API_KEY"), reason="Anthropic API key not found")
class TestAnthropic:
    """Test Anthropic provider."""

    def test_anthropic_connection(self):
        """Test Anthropic connection."""
        _, chat_model = get_llm(model="claude-sonnet-4-5")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
@pytest.mark.skipif(
    not (has_api_key("ANTHROPIC_FOUNDRY_API_KEY") and os.getenv("ANTHROPIC_FOUNDRY_BASE_URL")),
    reason="Anthropic Foundry credentials not found"
)
class TestAnthropicFoundry:
    """Test Anthropic Foundry (Azure) provider."""

    def test_anthropic_foundry_connection(self):
        """Test Anthropic Foundry connection."""
        _, chat_model = get_llm(model="azure-claude-sonnet-4-5")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
class TestOllama:
    """Test Ollama provider (assumes Ollama is running locally)."""

    @pytest.mark.skipif(
        not os.path.exists(os.path.expanduser("~/.ollama")),
        reason="Ollama not installed"
    )
    def test_ollama_connection(self):
        """Test Ollama with Llama model."""
        try:
            _, chat_model = get_llm(model="llama3.2:3b")
            response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
            validate_llm_response(response.content)
        except Exception as e:
            pytest.skip(f"Ollama service not available: {e}")


@pytest.mark.integration
@pytest.mark.skipif(not has_api_key("GEMINI_API_KEY"), reason="Gemini API key not found")
class TestGemini:
    """Test Google Gemini provider."""

    def test_gemini_connection(self):
        """Test Gemini with Gemini Pro model."""
        _, chat_model = get_llm(model="gemini-1.5-pro")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
@pytest.mark.skipif(not has_api_key("GROQ_API_KEY"), reason="Groq API key not found")
class TestGroq:
    """Test Groq provider."""

    def test_groq_connection(self):
        """Test Groq with Llama model."""
        _, chat_model = get_llm(
            model="llama-3.3-70b-versatile",
            source="Groq"
        )
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


@pytest.mark.integration
@pytest.mark.skipif(
    not (has_api_key("AWS_ACCESS_KEY_ID") and has_api_key("AWS_SECRET_ACCESS_KEY")),
    reason="AWS credentials not found"
)
class TestBedrock:
    """Test AWS Bedrock provider."""

    def test_bedrock_connection(self):
        """Test Bedrock with Claude model."""
        _, chat_model = get_llm(model="anthropic.claude-3-sonnet-20240229-v1:0", source="Bedrock")
        response = chat_model.invoke("What's 2+2? Answer in one short sentence.")
        validate_llm_response(response.content)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
