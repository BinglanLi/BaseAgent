"""Unit tests for persistent checkpointing — Phase 1 feature."""

from __future__ import annotations

import os
import uuid
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from BaseAgent.config import BaseAgentConfig


# ---------------------------------------------------------------------------
# Config field tests
# ---------------------------------------------------------------------------


class TestBaseAgentConfigCheckpointing:
    def test_default_checkpoint_db_path(self):
        cfg = BaseAgentConfig()
        assert cfg.checkpoint_db_path == ":memory:"

    def test_default_require_approval(self):
        cfg = BaseAgentConfig()
        assert cfg.require_approval == "always"

    def test_explicit_checkpoint_db_path(self):
        cfg = BaseAgentConfig(checkpoint_db_path="my.db")
        assert cfg.checkpoint_db_path == "my.db"

    def test_explicit_require_approval_always(self):
        cfg = BaseAgentConfig(require_approval="always")
        assert cfg.require_approval == "always"

    def test_to_dict_includes_new_fields(self):
        cfg = BaseAgentConfig(checkpoint_db_path="x.db", require_approval="always")
        d = cfg.to_dict()
        assert d["checkpoint_db_path"] == "x.db"
        assert d["require_approval"] == "always"

    def test_env_var_checkpoint_db_path(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_CHECKPOINT_DB_PATH", "env.db")
        cfg = BaseAgentConfig()
        assert cfg.checkpoint_db_path == "env.db"

    def test_env_var_require_approval_valid(self, monkeypatch):
        for val in ("always", "never"):
            monkeypatch.setenv("BASE_AGENT_REQUIRE_APPROVAL", val)
            cfg = BaseAgentConfig()
            assert cfg.require_approval == val

    def test_env_var_require_approval_invalid(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_REQUIRE_APPROVAL", "bogus")
        with pytest.raises(ValueError, match="BASE_AGENT_REQUIRE_APPROVAL"):
            BaseAgentConfig()

    def test_env_var_require_approval_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("BASE_AGENT_REQUIRE_APPROVAL", "ALWAYS")
        cfg = BaseAgentConfig()
        assert cfg.require_approval == "always"


# ---------------------------------------------------------------------------
# _create_checkpointer tests
# ---------------------------------------------------------------------------


class TestCreateCheckpointer:
    def test_returns_sqlite_saver_when_available(self, base_agent, tmp_path):
        pytest.importorskip("langgraph.checkpoint.sqlite", reason="langgraph-checkpoint-sqlite not installed")
        from langgraph.checkpoint.sqlite import SqliteSaver

        base_agent.checkpoint_db_path = str(tmp_path / "test.db")
        cp = base_agent._create_checkpointer()
        assert isinstance(cp, SqliteSaver)

    def test_memory_path_uses_in_memory_sqlite(self, base_agent):
        pytest.importorskip("langgraph.checkpoint.sqlite", reason="langgraph-checkpoint-sqlite not installed")
        from langgraph.checkpoint.sqlite import SqliteSaver

        base_agent.checkpoint_db_path = ":memory:"
        cp = base_agent._create_checkpointer()
        assert isinstance(cp, SqliteSaver)

    def test_fallback_warns_when_sqlite_missing_and_file_path(self, base_agent):
        """When sqlite package absent AND a file path is given, a warning is emitted."""
        base_agent.checkpoint_db_path = "some.db"
        with patch("BaseAgent.base_agent._HAS_SQLITE_SAVER", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cp = base_agent._create_checkpointer()
            assert any("langgraph-checkpoint-sqlite" in str(w.message) for w in caught)

    def test_fallback_no_warning_for_memory_path(self, base_agent):
        """When sqlite package absent but path is :memory:, no warning."""
        base_agent.checkpoint_db_path = ":memory:"
        with patch("BaseAgent.base_agent._HAS_SQLITE_SAVER", False):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cp = base_agent._create_checkpointer()
            assert not any("langgraph-checkpoint-sqlite" in str(w.message) for w in caught)

    def test_checkpointer_passed_to_compile(self, tmp_path):
        """compile() receives the checkpointer as a keyword argument."""
        mock_llm = MagicMock()
        mock_llm.model_name = "mock-model"
        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        mock_llm.invoke.return_value = mock_resp

        with patch("BaseAgent.base_agent.get_llm", return_value=("Anthropic", mock_llm)):
            from BaseAgent.base_agent import BaseAgent

            agent = BaseAgent(checkpoint_db_path=":memory:")

        # The compiled app should have a checkpointer attached
        assert agent.app.checkpointer is agent.checkpointer


# ---------------------------------------------------------------------------
# Thread ID tests
# ---------------------------------------------------------------------------


class TestThreadId:
    def test_thread_id_none_before_run(self, base_agent):
        assert base_agent.thread_id is None

    def test_thread_id_auto_generated_on_run(self, base_agent):
        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        base_agent.llm.invoke.return_value = mock_resp

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            base_agent.run("test prompt")

        assert base_agent.thread_id is not None
        # Must be a valid UUID
        uuid.UUID(base_agent.thread_id)

    def test_thread_id_caller_provided(self, base_agent):
        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        base_agent.llm.invoke.return_value = mock_resp

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            base_agent.run("test prompt", thread_id="my-session")

        assert base_agent.thread_id == "my-session"

    def test_thread_ids_differ_across_runs(self, base_agent):
        mock_resp = MagicMock()
        mock_resp.content = "<solution>done</solution>"
        base_agent.llm.invoke.return_value = mock_resp

        with patch("BaseAgent.nodes.extract_usage_metrics", return_value=None):
            base_agent.run("first")
            tid1 = base_agent.thread_id
            base_agent.run("second")
            tid2 = base_agent.thread_id

        assert tid1 != tid2


# ---------------------------------------------------------------------------
# close() / resource cleanup
# ---------------------------------------------------------------------------


class TestClose:
    def test_close_calls_conn_close(self, base_agent):
        mock_conn = MagicMock()
        base_agent.checkpointer = MagicMock()
        base_agent.checkpointer.conn = mock_conn
        base_agent.close()
        mock_conn.close.assert_called_once()

    def test_close_tolerates_missing_conn(self, base_agent):
        """close() should not raise when checkpointer has no .conn attribute."""
        base_agent.checkpointer = MagicMock(spec=[])  # no .conn
        base_agent.close()  # must not raise

    def test_close_tolerates_missing_checkpointer(self, base_agent):
        del base_agent.checkpointer
        base_agent.close()  # must not raise
