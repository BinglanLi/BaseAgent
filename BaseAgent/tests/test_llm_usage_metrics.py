from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from BaseAgent.llm import UsageMetrics, extract_usage_metrics


@dataclass
class DummyResponse:
    response_metadata: dict | None = None
    metadata: dict | None = None


def test_extract_usage_metrics_returns_none_when_metadata_missing() -> None:
    response = DummyResponse()

    metrics = extract_usage_metrics("OpenAI", response)

    assert metrics is None


def test_extract_usage_metrics_openai_family_uses_token_usage_mapping() -> None:
    metadata = {
        "model": "gpt-4o",
        "token_usage": {
            "prompt_tokens": "10",
            "completion_tokens": 5,
            "total_tokens": "15",
            "cost": "0.02",
            "currency": "USD",
        },
    }
    response = {"response_metadata": metadata}

    metrics = extract_usage_metrics("OpenAI", response, model="gpt-4o-mini")

    assert isinstance(metrics, UsageMetrics)
    assert metrics.provider == "OpenAI"
    assert metrics.model == "gpt-4o"
    assert metrics.input_tokens == 10
    assert metrics.output_tokens == 5
    assert metrics.total_tokens == 15
    assert metrics.cost == pytest.approx(0.02)
    assert metrics.currency == "USD"
    assert metrics.details["response_metadata"] == metadata


def test_extract_usage_metrics_anthropic_computes_missing_total_tokens() -> None:
    metadata = {
        "model": "claude-3-opus",
        "usage": {
            "input_tokens": "11",
            "output_tokens": "9",
            "total_cost": "0.055",
        },
    }
    response = DummyResponse(response_metadata=metadata)

    metrics = extract_usage_metrics("Anthropic", response)

    assert isinstance(metrics, UsageMetrics)
    assert metrics.provider == "Anthropic"
    assert metrics.model == "claude-3-opus"
    assert metrics.input_tokens == 11
    assert metrics.output_tokens == 9
    assert metrics.total_tokens == 20
    assert metrics.cost == pytest.approx(0.055)
    assert metrics.details["response_metadata"] == metadata


def test_extract_usage_metrics_bedrock_accepts_camel_case_usage_keys() -> None:
    metadata = {
        "modelId": "anthropic.claude-v2",
        "usage": {
            "inputTokens": "8",
            "outputTokens": "4",
        },
        "cost": "0.12",
        "currency": "USD",
    }
    response = DummyResponse(response_metadata=metadata)

    metrics = extract_usage_metrics("Bedrock", response)

    assert isinstance(metrics, UsageMetrics)
    assert metrics.provider == "Bedrock"
    assert metrics.model == "anthropic.claude-v2"
    assert metrics.input_tokens == 8
    assert metrics.output_tokens == 4
    assert metrics.total_tokens == 12
    assert metrics.cost == pytest.approx(0.12)
    assert metrics.currency == "USD"
    assert metrics.details["response_metadata"] == metadata


def test_extract_usage_metrics_ollama_sums_tokens_when_total_missing() -> None:
    metadata = {
        "model": "mistral",
        "prompt_eval_count": "3",
        "eval_count": 7,
    }
    response = DummyResponse(response_metadata=metadata)

    metrics = extract_usage_metrics("Ollama", response)

    assert isinstance(metrics, UsageMetrics)
    assert metrics.provider == "Ollama"
    assert metrics.model == "mistral"
    assert metrics.input_tokens == 3
    assert metrics.output_tokens == 7
    assert metrics.total_tokens == 10
    assert metrics.cost is None
    assert metrics.currency is None
    assert metrics.details["response_metadata"] == metadata

