import base64
import io
import sys
from io import StringIO

# Create a persistent namespace that will be shared across all executions
_persistent_namespace = {}

# Global list to store captured plots (module-level fallback for backward compat)
_captured_plots = []


class PlotCapture:
    """Per-agent matplotlib plot capture with isolated state.

    Replaces the module-level ``_captured_plots`` and ``_base_agent_patched``
    globals so that multiple ``BaseAgent`` instances capture plots independently
    without corrupting each other.

    Usage::

        capture = PlotCapture()
        capture.apply_patches()          # activate before code execution
        # ... exec user code ...
        plots = capture.get_plots()      # list of base64 data URIs
        capture.clear()                  # reset between runs
    """

    def __init__(self):
        self._plots: list[str] = []
        self._patched: bool = False

    def apply_patches(self) -> None:
        """Patch ``plt.show`` / ``plt.savefig`` to capture into this instance.

        Always replaces the current patches so the most recently activated
        ``PlotCapture`` is the active capture target.  The original matplotlib
        functions are preserved on the ``plt`` module the first time any
        ``PlotCapture`` patches, ensuring clean unwrapping across instances.
        """
        try:
            import matplotlib.pyplot as plt

            # Snapshot originals once (shared across all PlotCapture instances)
            if not hasattr(plt, "_base_agent_orig_show"):
                plt._base_agent_orig_show = plt.show
            if not hasattr(plt, "_base_agent_orig_savefig"):
                plt._base_agent_orig_savefig = plt.savefig

            capture = self
            orig_show = plt._base_agent_orig_show
            orig_savefig = plt._base_agent_orig_savefig

            def _show_with_capture(*args, **kwargs):
                capture._capture_current_figures()
                print("Plot generated and displayed")
                return orig_show(*args, **kwargs)

            def _savefig_with_capture(*args, **kwargs):
                filename = args[0] if args else kwargs.get("fname", "unknown")
                result = orig_savefig(*args, **kwargs)
                capture._capture_current_figures()
                print(f"Plot saved to: {filename}")
                return result

            plt.show = _show_with_capture
            plt.savefig = _savefig_with_capture
            self._patched = True

        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not apply matplotlib patches: {e}")

    def _capture_current_figures(self) -> None:
        """Capture all open matplotlib figures into ``self._plots``."""
        try:
            import matplotlib.pyplot as plt

            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                image_data = base64.b64encode(buf.getvalue()).decode("utf-8")
                plot_data = f"data:image/png;base64,{image_data}"
                if plot_data not in self._plots:
                    self._plots.append(plot_data)
                plt.close(fig)
        except ImportError:
            pass
        except Exception as e:
            print(f"Warning: Could not capture matplotlib plots: {e}")

    def get_plots(self) -> list[str]:
        """Return a copy of captured plots."""
        return self._plots.copy()

    def clear(self) -> None:
        """Clear captured plots."""
        self._plots = []


def run_python_repl(command: str, namespace: dict | None = None) -> str:
    """
    Executes the provided Python command in a persistent environment and returns the output.
    Variables defined in one execution will be available in subsequent executions.

    Args:
        command (str): The Python command to execute.
        namespace (dict | None): Execution namespace. When provided, ``exec`` uses this
            dict and matplotlib patches are *not* applied (the caller is responsible for
            plot capture via ``PlotCapture``).  When ``None``, falls back to the
            module-level ``_persistent_namespace`` and applies global matplotlib patches
            for backward compatibility.

    Returns:
        str: The output of the command
    """
    _ns = namespace if namespace is not None else _persistent_namespace

    def execute_in_repl(command: str) -> str:
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()

        try:
            # Apply global matplotlib patches only when using the global namespace
            # (backward compat). Per-instance callers apply PlotCapture.apply_patches()
            # themselves before invoking run_python_repl.
            if namespace is None:
                _apply_matplotlib_patches()

            exec(command, _ns)
            output = mystdout.getvalue()

        except Exception as e:
            output = f"Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
        return output

    command = command.strip("```").strip()
    return execute_in_repl(command)


def _capture_matplotlib_plots():
    """
    Capture any matplotlib plots that might have been generated during execution.
    """
    global _captured_plots
    try:
        import matplotlib.pyplot as plt

        # Check if there are any active figures
        if plt.get_fignums():
            for fig_num in plt.get_fignums():
                fig = plt.figure(fig_num)

                # Save figure to base64
                buffer = io.BytesIO()
                fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
                buffer.seek(0)

                # Convert to base64
                image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")
                plot_data = f"data:image/png;base64,{image_data}"

                # Add to captured plots if not already there
                if plot_data not in _captured_plots:
                    _captured_plots.append(plot_data)

                # Close the figure to free memory
                plt.close(fig)

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not capture matplotlib plots: {e}")


def _apply_matplotlib_patches():
    """
    Apply simple monkey patches to matplotlib functions to automatically capture plots.
    """
    try:
        import matplotlib.pyplot as plt

        # Only patch if matplotlib is available and not already patched
        if hasattr(plt, "_base_agent_patched"):
            return

        # Store original functions
        original_show = plt.show
        original_savefig = plt.savefig

        def show_with_capture(*args, **kwargs):
            """Enhanced show function that captures plots before displaying them."""
            # Capture any plots before showing
            _capture_matplotlib_plots()
            # Print a message to indicate plot was generated
            print("Plot generated and displayed")
            # Call the original show function
            return original_show(*args, **kwargs)

        def savefig_with_capture(*args, **kwargs):
            """Enhanced savefig function that captures plots after saving them."""
            # Get the filename from args if provided
            filename = args[0] if args else kwargs.get("fname", "unknown")
            # Call the original savefig function
            result = original_savefig(*args, **kwargs)
            # Capture the plot after saving
            _capture_matplotlib_plots()
            # Print a message to indicate plot was saved
            print(f"Plot saved to: {filename}")
            return result

        # Replace functions with enhanced versions
        plt.show = show_with_capture
        plt.savefig = savefig_with_capture

        # Mark as patched to avoid double-patching
        plt._BaseAgent_patched = True

    except ImportError:
        # matplotlib not available
        pass
    except Exception as e:
        print(f"Warning: Could not apply matplotlib patches: {e}")


def get_captured_plots():
    """
    Get all captured matplotlib plots.

    Returns:
        list: A list of captured matplotlib plots
    """
    global _captured_plots
    return _captured_plots.copy()


def clear_captured_plots():
    """
    Clear all captured matplotlib plots.
    """
    global _captured_plots
    _captured_plots = []


def read_function_source_code(function_name: str) -> str:
    """Read the source code of a function from any module path.

    Args:
        function_name (str): Fully qualified function name (e.g., 'bioagentos.tool.support_tools.write_python_code')

    Returns:
        str: The source code of the function

    """
    import importlib
    import inspect

    # Split the function name into module path and function name
    parts = function_name.split(".")
    module_path = ".".join(parts[:-1])
    func_name = parts[-1]

    try:
        # Import the module
        module = importlib.import_module(module_path)

        # Get the function object from the module
        function = getattr(module, func_name)

        # Get the source code of the function
        source_code = inspect.getsource(function)

        return source_code
    except (ImportError, AttributeError) as e:
        return f"Error: Could not find function '{function_name}'. Details: {str(e)}"