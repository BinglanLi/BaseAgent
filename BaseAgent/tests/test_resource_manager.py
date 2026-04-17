"""Unit tests for BaseAgent.resource_manager.ResourceManager."""

from __future__ import annotations

import json
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

from BaseAgent.resource_manager import ResourceManager
from BaseAgent.resources import (
    CustomData,
    CustomSoftware,
    CustomTool,
    DataLakeItem,
    Library,
    Tool,
    ToolParameter,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_tool(name="my_tool", module="test.module") -> Tool:
    return Tool(name=name, description=f"Description of {name}", module=module)


def make_tool_with_params(name="paramtool", module="test.module") -> Tool:
    return Tool(
        name=name,
        description="A parameterised tool",
        module=module,
        required_parameters=[
            ToolParameter(name="x", description="First arg", type="int")
        ],
        optional_parameters=[
            ToolParameter(name="verbose", description="Verbose flag", type="bool", default=False)
        ],
    )


def make_data(filename="data.csv", fmt="csv", category=None) -> DataLakeItem:
    return DataLakeItem(
        filename=filename,
        description=f"Dataset {filename}",
        format=fmt,
        category=category,
    )


def make_library(name="numpy", lib_type="Python", category=None) -> Library:
    return Library(
        name=name,
        description=f"Library {name}",
        type=lib_type,
        category=category,
    )


def make_custom_tool(name="custom_fn") -> CustomTool:
    return CustomTool(name=name, description="A custom tool")


def make_custom_data(name="custom.csv") -> CustomData:
    return CustomData(name=name, description="Custom dataset")


def make_custom_software(name="mycli") -> CustomSoftware:
    return CustomSoftware(name=name, description="Custom CLI tool")


# ===========================================================================
# Tool management
# ===========================================================================

class TestToolManagement:

    def test_add_tool_assigns_sequential_ids(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("tool_a"))
        rm.add_tool(make_tool("tool_b"))
        assert rm.collection.tools[0].id == 0
        assert rm.collection.tools[1].id == 1

    def test_add_tool_ignores_existing_id(self):
        """Any id already set on the tool is overwritten by the manager."""
        rm = ResourceManager()
        t = make_tool()
        t.id = 99
        rm.add_tool(t)
        assert t.id == 0

    def test_get_tool_id_by_name(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("alpha"))
        rm.add_tool(make_tool("beta"))
        assert rm.get_tool_id_by_name("beta") == 1

    def test_get_tool_name_by_id(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("gamma"))
        assert rm.get_tool_name_by_id(0) == "gamma"

    def test_list_all_tools(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("t1"))
        rm.add_tool(make_tool("t2"))
        listing = rm.list_all_tools()
        assert len(listing) == 2
        assert {"name": "t1", "id": 0} in listing
        assert {"name": "t2", "id": 1} in listing

    def test_remove_tool_by_id_success(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("removeme"))
        removed = rm.remove_tool_by_id(0)
        assert removed is True
        assert rm.get_tool_by_name("removeme") is None

    def test_remove_tool_by_name_success(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("byname"))
        removed = rm.remove_tool_by_name("byname")
        assert removed is True
        assert rm.get_tool_by_name("byname") is None

    def test_filter_tools_by_module(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("a", module="pkg.utils"))
        rm.add_tool(make_tool("b", module="pkg.core"))
        rm.add_tool(make_tool("c", module="pkg.utils"))
        result = rm.filter_tools_by_module("pkg.utils")
        assert len(result) == 2
        assert all("pkg.utils" in t.module for t in result)

    def test_add_custom_tool_appears_in_get_all_tools(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("std_tool"))
        rm.add_custom_tool(make_custom_tool("custom_fn"))
        all_tools = rm.get_all_tools()
        names = [t.name for t in all_tools]
        assert "std_tool" in names
        assert "custom_fn" in names

    def test_load_tools_bulk(self):
        rm = ResourceManager()
        tools = [make_tool(f"t{i}") for i in range(5)]
        rm.load_tools(tools)
        assert len(rm.collection.tools) == 5
        # IDs are sequential
        assert [t.id for t in rm.collection.tools] == list(range(5))


# ===========================================================================
# Data management
# ===========================================================================

class TestDataManagement:

    def test_get_all_data_combines_both(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("lake.tsv", "tsv"))
        rm.add_custom_data(make_custom_data("custom.csv"))
        all_data = rm.get_all_data()
        assert len(all_data) == 2

    def test_filter_data_by_category(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("a.csv", category="genomics"))
        rm.add_data_item(make_data("b.csv", category="drug"))
        rm.add_data_item(make_data("c.csv", category="genomics"))
        result = rm.filter_data_by_category("genomics")
        assert len(result) == 2

    def test_filter_data_by_format(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("a.parquet", "parquet"))
        rm.add_data_item(make_data("b.csv", "csv"))
        rm.add_data_item(make_data("c.parquet", "parquet"))
        result = rm.filter_data_by_format("parquet")
        assert len(result) == 2

    def test_filter_data_by_format_case_insensitive(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("a.csv", "csv"))
        result = rm.filter_data_by_format("CSV")
        assert len(result) == 1

    def test_load_data_items_bulk(self):
        rm = ResourceManager()
        items = [make_data(f"f{i}.csv") for i in range(3)]
        rm.load_data_items(items)
        assert len(rm.collection.data_lake) == 3


# ===========================================================================
# Library management
# ===========================================================================

class TestLibraryManagement:

    def test_get_all_libraries_combines_both(self):
        rm = ResourceManager()
        rm.add_library(make_library("biopython"))
        rm.add_custom_software(make_custom_software("blast"))
        assert len(rm.get_all_libraries()) == 2

    def test_filter_libraries_by_type(self):
        rm = ResourceManager()
        rm.add_library(make_library("numpy", "Python"))
        rm.add_library(make_library("ggplot2", "R"))
        rm.add_library(make_library("torch", "Python"))
        result = rm.filter_libraries_by_type("Python")
        assert len(result) == 2

    def test_filter_libraries_by_category(self):
        rm = ResourceManager()
        rm.add_library(make_library("biopython", category="bioinformatics"))
        rm.add_library(make_library("numpy", category="numerical"))
        rm.add_library(make_library("pysam", category="bioinformatics"))
        result = rm.filter_libraries_by_category("bioinformatics")
        assert len(result) == 2

    def test_load_libraries_bulk(self):
        rm = ResourceManager()
        libs = [make_library(f"lib{i}") for i in range(4)]
        rm.load_libraries(libs)
        assert len(rm.collection.libraries) == 4


# ===========================================================================
# Selection management
# ===========================================================================

class TestSelectionManagement:

    def _populated_rm(self) -> ResourceManager:
        rm = ResourceManager()
        rm.add_tool(make_tool("t1"))
        rm.add_tool(make_tool("t2"))
        rm.add_custom_tool(make_custom_tool("ct1"))
        rm.add_data_item(make_data("d1.csv"))
        rm.add_custom_data(make_custom_data("cd1.csv"))
        rm.add_library(make_library("lib1"))
        rm.add_custom_software(make_custom_software("cs1"))
        return rm

    def test_all_resources_selected_by_default(self):
        rm = self._populated_rm()
        assert all(t.selected for t in rm.get_all_tools())
        assert all(d.selected for d in rm.get_all_data())
        assert all(l.selected for l in rm.get_all_libraries())

    def test_deselect_all_resources(self):
        rm = self._populated_rm()
        rm.deselect_all_resources()
        assert all(not t.selected for t in rm.get_all_tools())
        assert all(not d.selected for d in rm.get_all_data())
        assert all(not l.selected for l in rm.get_all_libraries())

    def test_select_all_resources_after_deselect(self):
        rm = self._populated_rm()
        rm.deselect_all_resources()
        rm.select_all_resources()
        assert all(t.selected for t in rm.get_all_tools())

    def test_select_tools_by_names(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("alpha"))
        rm.add_tool(make_tool("beta"))
        rm.add_tool(make_tool("gamma"))
        rm.select_tools_by_names(["alpha", "gamma"])
        names_selected = {t.name for t in rm.collection.tools if t.selected}
        assert names_selected == {"alpha", "gamma"}
        assert not rm.collection.tools[1].selected  # beta

    def test_select_data_by_names(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("a.csv"))
        rm.add_data_item(make_data("b.csv"))
        rm.select_data_by_names(["a.csv"])
        assert rm.collection.data_lake[0].selected is True
        assert rm.collection.data_lake[1].selected is False

    def test_select_libraries_by_names(self):
        rm = ResourceManager()
        rm.add_library(make_library("numpy"))
        rm.add_library(make_library("pandas"))
        rm.select_libraries_by_names(["pandas"])
        assert rm.collection.libraries[0].selected is False
        assert rm.collection.libraries[1].selected is True

    def test_get_selected_tools(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("a"))
        rm.add_tool(make_tool("b"))
        rm.select_tools_by_names(["a"])
        selected = rm.get_selected_tools()
        assert len(selected) == 1
        assert selected[0].name == "a"

    def test_get_selected_data(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("keep.csv"))
        rm.add_data_item(make_data("drop.csv"))
        rm.select_data_by_names(["keep.csv"])
        selected = rm.get_selected_data()
        assert len(selected) == 1

    def test_get_selected_libraries(self):
        rm = ResourceManager()
        rm.add_library(make_library("keep_lib"))
        rm.add_library(make_library("drop_lib"))
        rm.select_libraries_by_names(["keep_lib"])
        selected = rm.get_selected_libraries()
        assert len(selected) == 1


# ===========================================================================
# Summary and categories
# ===========================================================================

class TestSummaryAndCategories:

    def test_get_summary_empty(self):
        rm = ResourceManager()
        summary = rm.get_summary()
        assert summary["tools"]["total"] == 0
        assert summary["data"]["total"] == 0
        assert summary["libraries"]["total"] == 0

    def test_get_summary_with_resources(self):
        rm = ResourceManager()
        rm.add_tool(make_tool("t1"))
        rm.add_tool(make_tool("t2"))
        rm.add_custom_tool(make_custom_tool("ct1"))
        rm.add_data_item(make_data("d.csv"))
        rm.add_library(make_library("lib"))
        summary = rm.get_summary()
        assert summary["tools"]["standard"] == 2
        assert summary["tools"]["custom"] == 1
        assert summary["tools"]["total"] == 3
        assert summary["data"]["data_lake"] == 1
        assert summary["libraries"]["standard"] == 1

    def test_get_categories_returns_sorted_unique(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("a.csv", category="genomics"))
        rm.add_data_item(make_data("b.csv", category="drug"))
        rm.add_data_item(make_data("c.csv", category="genomics"))
        rm.add_library(make_library("bio", category="bioinformatics"))
        cats = rm.get_categories()
        assert cats["data_categories"] == ["drug", "genomics"]
        assert cats["library_categories"] == ["bioinformatics"]

    def test_get_categories_empty(self):
        rm = ResourceManager()
        cats = rm.get_categories()
        assert cats["data_categories"] == []
        assert cats["library_categories"] == []


# ===========================================================================
# Description formatting
# ===========================================================================

class TestDescriptionFormatting:

    def test_format_tools_by_module_with_tool_objects(self):
        tools_by_module = defaultdict(list)
        t = make_tool_with_params("my_func", "pkg.utils")
        tools_by_module["pkg.utils"].append(t)
        output = ResourceManager.format_tools_by_module(tools_by_module)
        assert "Import file: pkg.utils" in output
        assert "Method: my_func" in output
        assert "Required Parameters:" in output
        assert "Optional Parameters:" in output

    def test_format_tools_by_module_with_dict_tools(self):
        tools_by_module = {
            "my.module": [
                {
                    "name": "dict_tool",
                    "description": "A dict-style tool",
                    "required_parameters": [
                        {"name": "q", "type": "str", "description": "query", "default": None}
                    ],
                    "optional_parameters": [],
                }
            ]
        }
        output = ResourceManager.format_tools_by_module(tools_by_module)
        assert "dict_tool" in output
        assert "A dict-style tool" in output

    def test_get_tools_description(self):
        rm = ResourceManager()
        rm.add_tool(make_tool_with_params("my_tool"))
        desc = rm.get_tools_description()
        assert "my_tool" in desc
        assert "test.module" in desc

    def test_get_data_description(self):
        rm = ResourceManager()
        rm.add_data_item(make_data("binding.tsv", "tsv", category="drug"))
        desc = rm.get_data_description()
        assert "binding.tsv" in desc
        assert "drug" in desc
        assert "tsv" in desc

    def test_get_libraries_description(self):
        rm = ResourceManager()
        rm.add_library(Library(
            name="biopython",
            description="Biological tools",
            type="Python",
            version="1.79",
            category="bioinformatics",
        ))
        desc = rm.get_libraries_description()
        assert "biopython" in desc
        assert "1.79" in desc
        assert "bioinformatics" in desc


# ===========================================================================
# Serialization round-trip
# ===========================================================================

class TestSerialization:

    def test_export_and_import_json_round_trip(self, tmp_path):
        rm = ResourceManager()
        rm.add_tool(make_tool("round_trip_tool"))
        rm.add_data_item(make_data("data.parquet", "parquet", category="protein"))
        rm.add_library(make_library("torch", "Python", category="ml"))

        filepath = str(tmp_path / "resources.json")
        rm.export_json(filepath)

        rm2 = ResourceManager()
        rm2.load_from_json(filepath)

        assert len(rm2.collection.tools) == 1
        assert rm2.collection.tools[0].name == "round_trip_tool"
        assert len(rm2.collection.data_lake) == 1
        assert rm2.collection.data_lake[0].format == "parquet"
        assert len(rm2.collection.libraries) == 1

    def test_exported_json_excludes_tool_ids(self, tmp_path):
        """Tool IDs are internal counters and should not appear in exported JSON."""
        rm = ResourceManager()
        rm.add_tool(make_tool("no_id_tool"))
        filepath = str(tmp_path / "resources.json")
        rm.export_json(filepath)

        with open(filepath) as f:
            data = json.load(f)

        # The exported tool should not have an 'id' field
        assert "id" not in data["tools"][0]


# ===========================================================================
# Built-in tools
# ===========================================================================

class TestLoadBuiltinTools:

    def test_load_builtin_tools_populates_collection(self):
        rm = ResourceManager()
        rm.load_builtin_tools()
        tool_names = [t.name for t in rm.collection.tools]
        assert "run_python_repl" in tool_names

    def test_load_builtin_tools_assigns_ids(self):
        rm = ResourceManager()
        rm.load_builtin_tools()
        for tool in rm.collection.tools:
            assert tool.id is not None

    def test_load_builtin_tools_sets_correct_module(self):
        rm = ResourceManager()
        rm.load_builtin_tools()
        repl_tool = rm.get_tool_by_name("run_python_repl")
        assert repl_tool is not None
        assert "support_tools" in repl_tool.module
