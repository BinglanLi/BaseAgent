"""Tool bridge utilities: REPL injection for custom tools."""


def inject_custom_functions_to_repl(custom_functions: dict, namespace: dict | None = None):
    """Inject custom tools/functions into the Python REPL execution environment.

    This function makes custom tools from ResourceManager available during code execution
    by injecting them into the target execution namespace.  This allows the agent to call
    custom functions that users have added via ``agent.add_tool()`` when executing Python
    code in ``<execute>`` blocks.

    Args:
        custom_functions: Dictionary mapping function names to their callable objects.
            Typically extracted from CustomTool objects in ResourceManager where
            ``tool.function`` contains the actual callable.
        namespace: Target execution namespace.  When provided, functions are injected
            into this dict (per-instance isolation).  When ``None``, falls back to the
            module-level ``_persistent_namespace`` for backward compatibility.

    Note:
        - When ``namespace`` is provided, only that namespace is modified.
        - When ``namespace`` is ``None``, also updates ``builtins._BaseAgent_custom_functions``
          for maximum backward-compatible access.
        - Custom tools are sourced from ``ResourceManager.collection.custom_tools``.
        - Each ``CustomTool.function`` (if not None) is made available by its ``tool.name``.

    Example:
        >>> agent.add_tool(my_custom_function)
        >>> # When agent executes Python code, my_custom_function is available:
        >>> # <execute>
        >>> # result = my_custom_function(param1, param2)
        >>> # </execute>
    """
    if not custom_functions:
        return

    if namespace is not None:
        # Per-instance mode: inject only into the provided namespace
        for name, func in custom_functions.items():
            namespace[name] = func
    else:
        # Global fallback mode (backward compat)
        from BaseAgent.tools.support_tools import _persistent_namespace

        for name, func in custom_functions.items():
            _persistent_namespace[name] = func

        import builtins

        if not hasattr(builtins, "_BaseAgent_custom_functions"):
            builtins._BaseAgent_custom_functions = {}
        builtins._BaseAgent_custom_functions.update(custom_functions)
