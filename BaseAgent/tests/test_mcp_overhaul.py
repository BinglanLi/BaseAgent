"""
Unit tests for MCP Overhaul (Feature 1):
  Phase 1: async/sync bridge fix in make_mcp_wrapper
  Phase 2: remote transport via streamablehttp_client
  Phase 3: auth headers with ${ENV_VAR} interpolation

Run with: pytest test_mcp_overhaul.py -v
"""

from __future__ import annotations

import asyncio
import textwrap
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from BaseAgent.base_agent import BaseAgent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_yaml(tmp_path: Path, cfg: dict) -> Path:
    """Write a YAML config dict to a temp file and return its path."""
    p = tmp_path / "mcp_config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


def _make_mock_tool(name="test_tool", description="A test tool",
                    input_schema=None):
    """Return a mock MCP tool object as returned by session.list_tools()."""
    tool = SimpleNamespace(
        name=name,
        description=description,
        inputSchema=input_schema or {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
            },
            "required": ["query"],
        },
    )
    return tool


def _make_tool_result(text="result text"):
    """Return a mock MCP tool call result."""
    content = SimpleNamespace(text=text)
    # No .json attribute -> falls through to .text
    return SimpleNamespace(content=[content])


# ---------------------------------------------------------------------------
# Phase 1: Async/Sync bridge fix
# ---------------------------------------------------------------------------

class TestAsyncSyncBridge:
    """Verify make_mcp_wrapper uses asyncio.run (not create_task)."""

    def test_stdio_wrapper_returns_value_not_coroutine(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Wrapper must return the result string, not a Task/coroutine."""
        mock_tools = [_make_mock_tool()]
        mock_result = _make_tool_result("hello")

        # Patch stdio_client to avoid real subprocess
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        # Build a context-manager mock for stdio_client
        mock_stdio_ctx = AsyncMock()
        mock_stdio_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock())
        )
        mock_stdio_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "test_server": {
                    "enabled": True,
                    "command": ["echo", "test"],
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_ctx),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        custom_tools = base_agent.resource_manager.collection.custom_tools
        assert len(custom_tools) >= 1
        tool = custom_tools[0]
        assert tool.name == "test_tool"

        # Now call the wrapper — it should return a string, not a Task
        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_ctx),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            result = tool.function(query="test")

        assert isinstance(result, str)
        assert result == "hello"
        assert not asyncio.isfuture(result)


# ---------------------------------------------------------------------------
# Phase 2: Remote transport
# ---------------------------------------------------------------------------

class TestRemoteTransport:
    """Verify remote MCP servers are discovered and wrapped via streamablehttp_client."""

    def test_remote_server_tools_are_registered(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Remote servers (url-based) should no longer be skipped."""
        mock_tools = [
            _make_mock_tool("remote_tool_a", "Tool A"),
            _make_mock_tool("remote_tool_b", "Tool B"),
        ]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), None)
        )
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "biocontext": {
                    "enabled": True,
                    "url": "https://mcp.biocontext.ai/mcp/",
                    "type": "remote",
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        names = [t.name for t in base_agent.resource_manager.collection.custom_tools]
        assert "remote_tool_a" in names
        assert "remote_tool_b" in names

    def test_remote_server_url_without_type_field(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """A server with 'url' but no 'type: remote' should still be treated as remote."""
        mock_tools = [_make_mock_tool("url_only_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), None)
        )
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "my_remote": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                    # No "type" field
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        names = [t.name for t in base_agent.resource_manager.collection.custom_tools]
        assert "url_only_tool" in names

    def test_remote_wrapper_returns_value(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Remote MCP wrapper should return the actual result, not a coroutine."""
        mock_tools = [_make_mock_tool("remote_exec")]
        mock_result = _make_tool_result("remote result")

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )
        mock_session.call_tool = AsyncMock(return_value=mock_result)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), None)
        )
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "remote_srv": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                    "type": "remote",
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        tool = base_agent.resource_manager.collection.custom_tools[0]
        assert tool.name == "remote_exec"

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            result = tool.function(query="test")

        assert result == "remote result"
        assert not asyncio.isfuture(result)

    def test_remote_server_missing_url_skipped(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Remote server with type=remote but no url should be skipped with a warning."""
        cfg = {
            "mcp_servers": {
                "bad_remote": {
                    "enabled": True,
                    "type": "remote",
                    # No "url" field
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)
        base_agent.add_mcp(str(config_path))

        assert len(base_agent.resource_manager.collection.custom_tools) == 0

    def test_remote_module_prefix(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Remote MCP tools should use the mcp_servers.<name> module prefix."""
        mock_tools = [_make_mock_tool("remote_mod_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), None)
        )
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "my_remote_srv": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        tool = base_agent.resource_manager.collection.custom_tools[0]
        assert tool.module == "mcp_servers.my_remote_srv"


# ---------------------------------------------------------------------------
# Phase 3: Auth headers with env var interpolation
# ---------------------------------------------------------------------------

class TestAuthHeaders:
    """Verify headers are processed and threaded into remote transport."""

    def test_headers_interpolated_from_env(
        self, base_agent: BaseAgent, tmp_path: Path, monkeypatch
    ):
        """${ENV_VAR} in headers should be replaced with actual env values."""
        monkeypatch.setenv("MY_API_KEY", "secret-key-123")

        mock_tools = [_make_mock_tool("authed_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        captured_headers = {}

        def fake_streamablehttp_client(url, headers=None):
            captured_headers.update(headers or {})
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(
                return_value=(AsyncMock(), AsyncMock(), None)
            )
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "authed_server": {
                    "enabled": True,
                    "url": "https://mcp.example.com/mcp/",
                    "type": "remote",
                    "headers": {
                        "Authorization": "Bearer ${MY_API_KEY}",
                        "X-Custom": "static-value",
                    },
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=fake_streamablehttp_client,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        # Verify headers were interpolated
        assert captured_headers["Authorization"] == "Bearer secret-key-123"
        assert captured_headers["X-Custom"] == "static-value"

    def test_missing_env_var_becomes_empty_string(
        self, base_agent: BaseAgent, tmp_path: Path, monkeypatch
    ):
        """${NONEXISTENT_VAR} should resolve to empty string."""
        monkeypatch.delenv("NONEXISTENT_MCP_KEY", raising=False)

        mock_tools = [_make_mock_tool("nokey_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        captured_headers = {}

        def fake_streamablehttp_client(url, headers=None):
            captured_headers.update(headers or {})
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(
                return_value=(AsyncMock(), AsyncMock(), None)
            )
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "nokey_server": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                    "headers": {
                        "Authorization": "Bearer ${NONEXISTENT_MCP_KEY}",
                    },
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=fake_streamablehttp_client,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        assert captured_headers["Authorization"] == "Bearer "

    def test_no_headers_produces_none(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Remote server without headers should pass None to transport."""
        mock_tools = [_make_mock_tool("noheader_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        captured_args = {}

        def fake_streamablehttp_client(url, headers=None):
            captured_args["headers"] = headers
            ctx = AsyncMock()
            ctx.__aenter__ = AsyncMock(
                return_value=(AsyncMock(), AsyncMock(), None)
            )
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "no_header_srv": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                    # No "headers" field
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                side_effect=fake_streamablehttp_client,
            ),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        assert captured_args["headers"] is None


# ---------------------------------------------------------------------------
# Backwards compatibility: stdio servers still work
# ---------------------------------------------------------------------------

class TestStdioBackwardsCompat:
    """Ensure existing stdio-based servers still work unchanged."""

    def test_stdio_server_still_registered(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Stdio servers should still be discovered and registered normally."""
        mock_tools = [_make_mock_tool("stdio_tool")]

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=mock_tools)
        )

        mock_stdio_ctx = AsyncMock()
        mock_stdio_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock())
        )
        mock_stdio_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_session_ctx = AsyncMock()
        mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session_ctx.__aexit__ = AsyncMock(return_value=False)

        cfg = {
            "mcp_servers": {
                "local_srv": {
                    "enabled": True,
                    "command": ["python", "-m", "some_server"],
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_ctx),
            patch("mcp.ClientSession", return_value=mock_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        names = [t.name for t in base_agent.resource_manager.collection.custom_tools]
        assert "stdio_tool" in names

    def test_disabled_server_still_skipped(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Servers with enabled: false should be skipped."""
        cfg = {
            "mcp_servers": {
                "disabled_srv": {
                    "enabled": False,
                    "command": ["echo", "test"],
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)
        base_agent.add_mcp(str(config_path))

        assert len(base_agent.resource_manager.collection.custom_tools) == 0

    def test_mixed_stdio_and_remote_servers(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Both stdio and remote servers in the same config should be processed."""
        mock_stdio_tool = _make_mock_tool("local_tool")
        mock_remote_tool = _make_mock_tool("cloud_tool")

        mock_stdio_session = AsyncMock()
        mock_stdio_session.initialize = AsyncMock()
        mock_stdio_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[mock_stdio_tool])
        )

        mock_remote_session = AsyncMock()
        mock_remote_session.initialize = AsyncMock()
        mock_remote_session.list_tools = AsyncMock(
            return_value=SimpleNamespace(tools=[mock_remote_tool])
        )

        mock_stdio_ctx = AsyncMock()
        mock_stdio_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock())
        )
        mock_stdio_ctx.__aexit__ = AsyncMock(return_value=False)

        mock_http_ctx = AsyncMock()
        mock_http_ctx.__aenter__ = AsyncMock(
            return_value=(AsyncMock(), AsyncMock(), None)
        )
        mock_http_ctx.__aexit__ = AsyncMock(return_value=False)

        # Track which session is used per call
        session_call_count = {"count": 0}
        sessions = [mock_stdio_session, mock_remote_session]

        def make_session_ctx(*args, **kwargs):
            ctx = AsyncMock()
            s = sessions[min(session_call_count["count"], len(sessions) - 1)]
            session_call_count["count"] += 1
            ctx.__aenter__ = AsyncMock(return_value=s)
            ctx.__aexit__ = AsyncMock(return_value=False)
            return ctx

        cfg = {
            "mcp_servers": {
                "local_srv": {
                    "enabled": True,
                    "command": ["echo", "test"],
                },
                "remote_srv": {
                    "enabled": True,
                    "url": "https://example.com/mcp/",
                    "type": "remote",
                },
            }
        }
        config_path = _write_yaml(tmp_path, cfg)

        with (
            patch("mcp.client.stdio.stdio_client", return_value=mock_stdio_ctx),
            patch(
                "mcp.client.streamable_http.streamablehttp_client",
                return_value=mock_http_ctx,
            ),
            patch("mcp.ClientSession", side_effect=make_session_ctx),
        ):
            base_agent.add_mcp(str(config_path))

        names = [t.name for t in base_agent.resource_manager.collection.custom_tools]
        assert "local_tool" in names
        assert "cloud_tool" in names

    def test_manual_tool_definitions_still_work(
        self, base_agent: BaseAgent, tmp_path: Path
    ):
        """Manual tool definitions in config should still be registered without discovery."""
        cfg = {
            "mcp_servers": {
                "manual_srv": {
                    "enabled": True,
                    "command": ["echo", "test"],
                    "tools": [
                        {
                            "name": "manual_tool",
                            "description": "A manually defined tool",
                            "parameters": {
                                "input": {
                                    "type": "string",
                                    "description": "Input data",
                                    "required": True,
                                }
                            },
                        }
                    ],
                }
            }
        }
        config_path = _write_yaml(tmp_path, cfg)
        base_agent.add_mcp(str(config_path))

        tools = base_agent.resource_manager.collection.custom_tools
        assert len(tools) == 1
        assert tools[0].name == "manual_tool"
        assert len(tools[0].required_parameters) == 1
        assert tools[0].required_parameters[0].name == "input"
