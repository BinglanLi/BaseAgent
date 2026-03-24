"""Tool bridge utilities: REPL injection for custom tools."""


def inject_custom_functions_to_repl(custom_functions: dict):
    """Inject custom tools/functions into the Python REPL execution environment.

    This function makes custom tools from ResourceManager available during code execution
    by injecting them into both the persistent execution namespace and the builtins module.
    This allows the agent to call custom functions that users have added via
    agent.add_tool() when executing Python code in <execute> blocks.

    Args:
        custom_functions: Dictionary mapping function names to their callable objects.
                        Typically extracted from CustomTool objects in ResourceManager
                        where tool.function contains the actual callable.

    Note:
        - Modifies the persistent namespace used by run_python_repl
        - Also adds functions to builtins for maximum compatibility
        - Custom tools are sourced from ResourceManager.collection.custom_tools
        - Each CustomTool.function (if not None) is made available by its tool.name

    Example:
        >>> # Custom tools in ResourceManager are automatically injected
        >>> agent.add_tool(my_custom_function)
        >>> # When agent executes Python code:
        >>> # <execute>
        >>> # result = my_custom_function(param1, param2)  # Available!
        >>> # </execute>
    """
    if custom_functions:
        # Access the persistent namespace used by run_python_repl
        from BaseAgent.tools.support_tools import _persistent_namespace

        # Inject all custom functions into the execution namespace
        for name, func in custom_functions.items():
            _persistent_namespace[name] = func

        # Also make them available in builtins for broader access
        import builtins

        if not hasattr(builtins, "_BaseAgent_custom_functions"):
            builtins._BaseAgent_custom_functions = {}
        builtins._BaseAgent_custom_functions.update(custom_functions)
