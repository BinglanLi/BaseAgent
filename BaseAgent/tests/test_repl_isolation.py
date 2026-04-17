"""Tests for REPL namespace isolation between BaseAgent instances.

Covers:
- Per-instance namespace isolation (variable set in one agent is invisible to another)
- Variable persistence within a single agent across execute blocks
- Backward-compat global fallback when namespace=None
- Per-instance PlotCapture isolation
- inject_custom_functions_to_repl namespace parameter
"""
from __future__ import annotations

import pytest

from BaseAgent.tools.support_tools import (
    PlotCapture,
    _persistent_namespace,
    run_python_repl,
)
from BaseAgent.utils.tool_bridge import inject_custom_functions_to_repl
from helpers.node_helpers import make_base_agent as make_agent

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# run_python_repl — namespace parameter
# ---------------------------------------------------------------------------

class TestRunPythonReplNamespace:
    """run_python_repl uses provided namespace, not global."""

    def test_provided_namespace_receives_vars(self):
        ns = {}
        run_python_repl("x = 42", namespace=ns)
        assert ns.get("x") == 42

    def test_global_namespace_unchanged_when_ns_provided(self):
        _persistent_namespace.pop("_isolation_probe", None)
        ns = {}
        run_python_repl("_isolation_probe = 'hello'", namespace=ns)
        assert "_isolation_probe" not in _persistent_namespace

    def test_variables_persist_across_calls_in_same_namespace(self):
        ns = {}
        run_python_repl("counter = 0", namespace=ns)
        run_python_repl("counter += 1", namespace=ns)
        run_python_repl("counter += 1", namespace=ns)
        assert ns["counter"] == 2

    def test_two_namespaces_are_isolated(self):
        ns1, ns2 = {}, {}
        run_python_repl("shared = 'from_ns1'", namespace=ns1)
        run_python_repl("shared = 'from_ns2'", namespace=ns2)
        assert ns1["shared"] == "from_ns1"
        assert ns2["shared"] == "from_ns2"

    def test_output_captured_correctly_with_namespace(self):
        ns = {}
        output = run_python_repl("print('hello namespace')", namespace=ns)
        assert "hello namespace" in output

    def test_error_in_namespace_does_not_raise(self):
        ns = {}
        output = run_python_repl("1 / 0", namespace=ns)
        assert "Error" in output


# ---------------------------------------------------------------------------
# BaseAgent._repl_namespace isolation
# ---------------------------------------------------------------------------

class TestBaseAgentNamespaceIsolation:
    """Per-instance namespace on BaseAgent instances is isolated."""

    def test_each_instance_has_own_namespace(self):
        agent1 = make_agent()
        agent2 = make_agent()
        assert agent1._repl_namespace is not agent2._repl_namespace

    def test_variable_in_agent1_invisible_to_agent2(self):
        agent1 = make_agent()
        agent2 = make_agent()
        run_python_repl("secret = 'agent1_only'", namespace=agent1._repl_namespace)
        assert "secret" not in agent2._repl_namespace

    def test_variable_persists_within_same_agent(self):
        agent = make_agent()
        run_python_repl("my_var = 10", namespace=agent._repl_namespace)
        run_python_repl("my_var += 5", namespace=agent._repl_namespace)
        assert agent._repl_namespace["my_var"] == 15

    def test_namespace_starts_empty(self):
        agent = make_agent()
        assert agent._repl_namespace == {}


# ---------------------------------------------------------------------------
# inject_custom_functions_to_repl — namespace parameter
# ---------------------------------------------------------------------------

class TestInjectCustomFunctions:
    """inject_custom_functions_to_repl respects the namespace parameter."""

    def test_inject_into_provided_namespace(self):
        ns = {}
        inject_custom_functions_to_repl({"my_fn": lambda: "result"}, namespace=ns)
        assert "my_fn" in ns
        assert ns["my_fn"]() == "result"

    def test_global_namespace_unchanged_when_ns_provided(self):
        _persistent_namespace.pop("_probe_fn", None)
        ns = {}
        inject_custom_functions_to_repl({"_probe_fn": lambda: None}, namespace=ns)
        assert "_probe_fn" not in _persistent_namespace

    def test_empty_functions_dict_is_noop(self):
        ns = {}
        inject_custom_functions_to_repl({}, namespace=ns)
        assert ns == {}

    def test_injected_function_callable_in_repl(self):
        ns = {}
        inject_custom_functions_to_repl({"double": lambda x: x * 2}, namespace=ns)
        output = run_python_repl("print(double(7))", namespace=ns)
        assert "14" in output


# ---------------------------------------------------------------------------
# PlotCapture
# ---------------------------------------------------------------------------

class TestPlotCapture:
    """PlotCapture provides isolated per-instance plot buffering."""

    def test_starts_empty(self):
        cap = PlotCapture()
        assert cap.get_plots() == []

    def test_clear_empties_list(self):
        cap = PlotCapture()
        cap._plots = ["data:image/png;base64,abc"]
        cap.clear()
        assert cap.get_plots() == []

    def test_get_plots_returns_copy(self):
        cap = PlotCapture()
        cap._plots = ["plot1"]
        result = cap.get_plots()
        result.append("extra")
        assert cap._plots == ["plot1"]  # internal list not mutated

    def test_two_instances_have_independent_buffers(self):
        cap1 = PlotCapture()
        cap2 = PlotCapture()
        cap1._plots.append("plot_from_cap1")
        assert cap2.get_plots() == []

    def test_patched_flag_set_after_apply_patches(self):
        pytest.importorskip("matplotlib")
        cap = PlotCapture()
        assert not cap._patched
        cap.apply_patches()
        assert cap._patched

    def test_apply_patches_noop_without_matplotlib(self, monkeypatch):
        """apply_patches() should not raise even if matplotlib is absent."""
        import sys
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
        cap = PlotCapture()
        cap.apply_patches()  # must not raise
        assert not cap._patched

    def test_clear_does_not_affect_other_instance(self):
        cap1 = PlotCapture()
        cap2 = PlotCapture()
        cap1._plots.append("plot")
        cap2.clear()
        assert cap1.get_plots() == ["plot"]

    def test_base_agent_has_own_plot_capture(self):
        agent1 = make_agent()
        agent2 = make_agent()
        assert agent1._plot_capture is not agent2._plot_capture
        agent1._plot_capture._plots.append("agent1_plot")
        assert agent2._plot_capture.get_plots() == []
