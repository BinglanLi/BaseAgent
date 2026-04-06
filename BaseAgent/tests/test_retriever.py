"""Unit tests for BaseAgent.retriever.ToolRetriever."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from BaseAgent.retriever import ToolRetriever


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_resources(n_tools=2, n_data=2, n_libs=2) -> dict:
    return {
        "all_tools": [
            {"name": f"tool_{i}", "description": f"Tool number {i}"}
            for i in range(n_tools)
        ],
        "all_data": [
            {"name": f"data_{i}.csv", "description": f"Dataset {i}"}
            for i in range(n_data)
        ],
        "all_libraries": [
            {"name": f"lib_{i}", "description": f"Library {i}"}
            for i in range(n_libs)
        ],
    }


def make_llm(response_text: str) -> MagicMock:
    llm = MagicMock()
    resp = MagicMock()
    resp.content = response_text
    llm.invoke.return_value = resp
    return llm


# ===========================================================================
# _format_resources_for_prompt
# ===========================================================================

class TestFormatResourcesForPrompt:

    def setup_method(self):
        self.retriever = ToolRetriever()

    def test_empty_list_returns_none_available(self):
        result = self.retriever._format_resources_for_prompt([])
        assert result == "None available"

    def test_single_resource_formatted(self):
        resources = [{"name": "blast", "description": "Sequence alignment tool"}]
        result = self.retriever._format_resources_for_prompt(resources)
        assert result == "0. blast: Sequence alignment tool"

    def test_multiple_resources_numbered(self):
        resources = [
            {"name": "alpha", "description": "First"},
            {"name": "beta", "description": "Second"},
            {"name": "gamma", "description": "Third"},
        ]
        result = self.retriever._format_resources_for_prompt(resources)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        assert lines[0].startswith("0.")
        assert lines[1].startswith("1.")
        assert lines[2].startswith("2.")

    def test_resource_name_and_description_present(self):
        resources = [{"name": "my_tool", "description": "Does something useful"}]
        result = self.retriever._format_resources_for_prompt(resources)
        assert "my_tool" in result
        assert "Does something useful" in result


# ===========================================================================
# _parse_llm_response
# ===========================================================================

class TestParseLlmResponse:

    def setup_method(self):
        self.retriever = ToolRetriever()

    def test_parses_tools_indices(self):
        response = "TOOLS: [0, 2]\nDATA: []\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == [0, 2]

    def test_parses_data_indices(self):
        response = "TOOLS: []\nDATA: [1, 3]\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_data"] == [1, 3]

    def test_parses_libraries_indices(self):
        response = "TOOLS: []\nDATA: []\nLIBRARIES: [0, 1, 2]"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_libraries"] == [0, 1, 2]

    def test_parses_all_categories_together(self):
        response = "TOOLS: [0]\nDATA: [2]\nLIBRARIES: [1]"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == [0]
        assert result["selected_data"] == [2]
        assert result["selected_libraries"] == [1]

    def test_empty_brackets_return_empty_list(self):
        response = "TOOLS: []\nDATA: []\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == []
        assert result["selected_data"] == []
        assert result["selected_libraries"] == []

    def test_case_insensitive_matching(self):
        response = "tools: [0]\ndata: [1]\nlibraries: [2]"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == [0]
        assert result["selected_data"] == [1]
        assert result["selected_libraries"] == [2]

    def test_malformed_response_returns_empty_lists(self):
        response = "I don't know what to do."
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == []
        assert result["selected_data"] == []
        assert result["selected_libraries"] == []

    def test_non_integer_indices_suppressed(self):
        """Non-integer tokens inside brackets are silently ignored."""
        response = "TOOLS: [a, b]\nDATA: []\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == []

    def test_mixed_valid_and_invalid_indices_suppressed(self):
        """If any token is non-integer, the entire list is suppressed by contextlib.suppress."""
        response = "TOOLS: [0, x, 2]\nDATA: []\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        # contextlib.suppress(ValueError) means the whole list comprehension is skipped on error
        assert result["selected_tools"] == []

    def test_whitespace_around_indices_handled(self):
        response = "TOOLS: [ 0 , 1 ]\nDATA: []\nLIBRARIES: []"
        result = self.retriever._parse_llm_response(response)
        assert result["selected_tools"] == [0, 1]

    def test_all_keys_always_present_in_output(self):
        """Output always contains all three keys even if only some are matched."""
        response = "TOOLS: [0]"
        result = self.retriever._parse_llm_response(response)
        assert "selected_tools" in result
        assert "selected_data" in result
        assert "selected_libraries" in result


# ===========================================================================
# prompt_based_retrieval (integration with mocked LLM)
# ===========================================================================

class TestPromptBasedRetrieval:

    def setup_method(self):
        self.retriever = ToolRetriever()

    def test_returns_selected_tools(self):
        resources = make_resources(n_tools=3, n_data=0, n_libs=0)
        llm = make_llm("TOOLS: [0, 2]\nDATA: []\nLIBRARIES: []")
        result = self.retriever.prompt_based_retrieval("find proteins", resources, llm=llm)
        assert len(result["selected_tools"]) == 2
        assert result["selected_tools"][0]["name"] == "tool_0"
        assert result["selected_tools"][1]["name"] == "tool_2"

    def test_returns_selected_data(self):
        resources = make_resources(n_tools=0, n_data=3, n_libs=0)
        llm = make_llm("TOOLS: []\nDATA: [1]\nLIBRARIES: []")
        result = self.retriever.prompt_based_retrieval("analyze dataset", resources, llm=llm)
        assert len(result["selected_data"]) == 1
        assert result["selected_data"][0]["name"] == "data_1.csv"

    def test_returns_selected_libraries(self):
        resources = make_resources(n_tools=0, n_data=0, n_libs=4)
        llm = make_llm("TOOLS: []\nDATA: []\nLIBRARIES: [0, 3]")
        result = self.retriever.prompt_based_retrieval("need libraries", resources, llm=llm)
        assert len(result["selected_libraries"]) == 2
        assert result["selected_libraries"][0]["name"] == "lib_0"
        assert result["selected_libraries"][1]["name"] == "lib_3"

    def test_out_of_bounds_indices_are_ignored(self):
        """Indices >= length of resource list are silently dropped."""
        resources = make_resources(n_tools=2, n_data=0, n_libs=0)
        llm = make_llm("TOOLS: [0, 5, 99]\nDATA: []\nLIBRARIES: []")
        result = self.retriever.prompt_based_retrieval("query", resources, llm=llm)
        # Only index 0 is valid (len == 2, so 5 and 99 are out of bounds)
        assert len(result["selected_tools"]) == 1
        assert result["selected_tools"][0]["name"] == "tool_0"

    def test_empty_resources_returns_empty_selections(self):
        resources = {"all_tools": [], "all_data": [], "all_libraries": []}
        llm = make_llm("TOOLS: []\nDATA: []\nLIBRARIES: []")
        result = self.retriever.prompt_based_retrieval("anything", resources, llm=llm)
        assert result["selected_tools"] == []
        assert result["selected_data"] == []
        assert result["selected_libraries"] == []

    def test_output_keys_are_correct(self):
        resources = make_resources()
        llm = make_llm("TOOLS: [0]\nDATA: [0]\nLIBRARIES: [0]")
        result = self.retriever.prompt_based_retrieval("task", resources, llm=llm)
        assert set(result.keys()) == {"selected_tools", "selected_data", "selected_libraries", "selected_skills"}

    def test_llm_is_invoked_once(self):
        resources = make_resources()
        llm = make_llm("TOOLS: []\nDATA: []\nLIBRARIES: []")
        self.retriever.prompt_based_retrieval("task", resources, llm=llm)
        llm.invoke.assert_called_once()

    def test_missing_resource_keys_default_to_empty(self):
        """resources dict with missing keys defaults to empty lists."""
        resources = {}  # no keys at all
        llm = make_llm("TOOLS: []\nDATA: []\nLIBRARIES: []")
        result = self.retriever.prompt_based_retrieval("task", resources, llm=llm)
        assert result["selected_tools"] == []
        assert result["selected_data"] == []
        assert result["selected_libraries"] == []
