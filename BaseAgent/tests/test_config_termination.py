"""Unit tests for Feature 6 config fields: max_iterations, max_cost, max_consecutive_errors."""

from __future__ import annotations

import os

import pytest

from BaseAgent.config import BaseAgentConfig


class TestMaxIterations:
    def test_none_by_default(self):
        cfg = BaseAgentConfig()
        assert cfg.max_iterations is None

    def test_valid_value(self):
        cfg = BaseAgentConfig(max_iterations=10)
        assert cfg.max_iterations == 10

    def test_minimum_value(self):
        cfg = BaseAgentConfig(max_iterations=1)
        assert cfg.max_iterations == 1

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="max_iterations"):
            BaseAgentConfig(max_iterations=0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="max_iterations"):
            BaseAgentConfig(max_iterations=-5)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_ITERATIONS", "20")
        cfg = BaseAgentConfig()
        assert cfg.max_iterations == 20

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_ITERATIONS", "not_a_number")
        with pytest.raises(ValueError):
            BaseAgentConfig()

    def test_env_var_zero_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_ITERATIONS", "0")
        with pytest.raises(ValueError, match="MAX_ITERATIONS"):
            BaseAgentConfig()


class TestMaxCost:
    def test_none_by_default(self):
        cfg = BaseAgentConfig()
        assert cfg.max_cost is None

    def test_valid_value(self):
        cfg = BaseAgentConfig(max_cost=1.5)
        assert cfg.max_cost == 1.5

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="max_cost"):
            BaseAgentConfig(max_cost=0.0)

    def test_negative_raises(self):
        with pytest.raises(ValueError, match="max_cost"):
            BaseAgentConfig(max_cost=-0.01)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_COST", "2.5")
        cfg = BaseAgentConfig()
        assert cfg.max_cost == 2.5

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_COST", "not_a_float")
        with pytest.raises(ValueError):
            BaseAgentConfig()

    def test_env_var_zero_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_COST", "0")
        with pytest.raises(ValueError, match="MAX_COST"):
            BaseAgentConfig()


class TestMaxConsecutiveErrors:
    def test_none_by_default(self):
        cfg = BaseAgentConfig()
        assert cfg.max_consecutive_errors is None

    def test_valid_value(self):
        cfg = BaseAgentConfig(max_consecutive_errors=3)
        assert cfg.max_consecutive_errors == 3

    def test_minimum_value(self):
        cfg = BaseAgentConfig(max_consecutive_errors=2)
        assert cfg.max_consecutive_errors == 2

    def test_one_raises(self):
        with pytest.raises(ValueError, match="max_consecutive_errors"):
            BaseAgentConfig(max_consecutive_errors=1)

    def test_zero_raises(self):
        with pytest.raises(ValueError, match="max_consecutive_errors"):
            BaseAgentConfig(max_consecutive_errors=0)

    def test_env_var_override(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONSECUTIVE_ERRORS", "4")
        cfg = BaseAgentConfig()
        assert cfg.max_consecutive_errors == 4

    def test_env_var_invalid_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONSECUTIVE_ERRORS", "abc")
        with pytest.raises(ValueError):
            BaseAgentConfig()

    def test_env_var_one_raises(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_MAX_CONSECUTIVE_ERRORS", "1")
        with pytest.raises(ValueError, match="MAX_CONSECUTIVE_ERRORS"):
            BaseAgentConfig()


class TestToDict:
    def test_new_fields_in_to_dict(self):
        cfg = BaseAgentConfig(max_iterations=5, max_cost=2.0, max_consecutive_errors=3)
        d = cfg.to_dict()
        assert d["max_iterations"] == 5
        assert d["max_cost"] == 2.0
        assert d["max_consecutive_errors"] == 3

    def test_none_values_in_to_dict(self):
        cfg = BaseAgentConfig()
        d = cfg.to_dict()
        assert d["max_iterations"] is None
        assert d["max_cost"] is None
        assert d["max_consecutive_errors"] is None
