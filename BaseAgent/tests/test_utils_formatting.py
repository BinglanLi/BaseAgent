"""Unit tests for BaseAgent.utils.formatting module."""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from BaseAgent.utils.formatting import (
    clean_message_content,
    color_print,
    extract_agent_result,
    langchain_to_gradio_message,
    pretty_print,
    wrap_text,
)

pytestmark = pytest.mark.unit


class TestWrapText:
    """Tests for wrap_text()."""

    def test_short_text_unchanged(self):
        text = "Short text."
        assert wrap_text(text, max_line_length=80) == text

    def test_long_text_wrapped(self):
        text = "word " * 30  # ~150 chars
        result = wrap_text(text.strip(), max_line_length=40)
        for line in result.splitlines():
            assert len(line) <= 40

    def test_exactly_at_limit_unchanged(self):
        text = "a" * 80
        assert wrap_text(text, max_line_length=80) == text

    def test_custom_width(self):
        text = "one two three four five six seven eight nine ten"
        result = wrap_text(text, max_line_length=20)
        for line in result.splitlines():
            assert len(line) <= 20


class TestCleanMessageContent:
    """Tests for clean_message_content()."""

    def test_removes_ansi_color_codes(self):
        raw = "\x1b[31mred text\x1b[0m"
        assert clean_message_content(raw) == "red text"

    def test_plain_text_unchanged(self):
        text = "Hello, world!"
        assert clean_message_content(text) == text

    def test_multiple_ansi_codes(self):
        raw = "\x1b[36;1mblue bold\x1b[0m and \x1b[32;1mgreen\x1b[0m"
        result = clean_message_content(raw)
        assert "\x1b" not in result
        assert "blue bold" in result
        assert "green" in result


class TestColorPrint:
    """Tests for color_print()."""

    def test_does_not_raise(self, capsys):
        color_print("hello", color="blue")
        captured = capsys.readouterr()
        assert "hello" in captured.out

    def test_all_colors(self, capsys):
        for color in ("blue", "yellow", "pink", "green", "red"):
            color_print("test", color=color)


class TestPrettyPrint:
    """Tests for pretty_print()."""

    def test_human_message(self, capsys):
        msg = HumanMessage(content="Hello agent")
        title = pretty_print(msg, printout=True)
        assert "Hello agent" in title

    def test_ai_message(self, capsys):
        msg = AIMessage(content="I will help you.")
        title = pretty_print(msg, printout=True)
        assert "I will help you." in title

    def test_returns_string(self):
        msg = HumanMessage(content="test")
        result = pretty_print(msg, printout=False)
        assert isinstance(result, str)


class TestExtractAgentResult:
    """Tests for extract_agent_result()."""

    def test_extracts_solution_content(self):
        raw = "<think>reasoning</think><solution>the answer</solution>"
        assert extract_agent_result(raw) == "the answer"

    def test_strips_think_when_no_solution(self):
        raw = "<think>some reasoning</think>plain text"
        assert extract_agent_result(raw) == "plain text"

    def test_passthrough_for_plain_text(self):
        assert extract_agent_result("ERROR: something went wrong") == "ERROR: something went wrong"

    def test_strips_execute_block(self):
        raw = "<execute>print('hi')</execute>result text"
        assert extract_agent_result(raw) == "result text"

    def test_truncates_long_solution(self):
        raw = f"<solution>{'x' * 3000}</solution>"
        assert len(extract_agent_result(raw)) == 2000

    def test_multiline_solution(self):
        raw = "<think>step 1\nstep 2</think><solution>line1\nline2</solution>"
        assert extract_agent_result(raw) == "line1\nline2"


class TestLangchainToGradioMessage:
    """Tests for langchain_to_gradio_message()."""

    def test_human_message_role(self):
        msg = HumanMessage(content="hi")
        result = langchain_to_gradio_message(msg)
        assert isinstance(result, list)
        assert result[0]["role"] == "user"

    def test_ai_message_role(self):
        msg = AIMessage(content="hello")
        result = langchain_to_gradio_message(msg)
        assert isinstance(result, list)
        assert result[0]["role"] == "assistant"

    def test_content_included(self):
        msg = HumanMessage(content="test content")
        result = langchain_to_gradio_message(msg)
        assert "test content" in result[0]["content"]

    def test_think_tags_replaced(self):
        msg = AIMessage(content="<think>reasoning</think>answer")
        result = langchain_to_gradio_message(msg)
        assert "<think>" not in result[0]["content"]
