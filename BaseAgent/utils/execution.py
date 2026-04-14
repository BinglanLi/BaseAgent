"""Execution utilities: run R code, Bash scripts, and Python with timeout."""

import ctypes
import os
import queue
import re
import subprocess
import tempfile
import threading


# Error string prefixes returned by run_with_timeout.
# Used by nodes.py to detect infrastructure failures for consecutive-error tracking.
TIMEOUT_ERROR_PREFIX = "ERROR: Code execution timed out"
EXECUTION_ERROR_PREFIX = "Error in execution:"

# Language marker prefixes → (language, tool_name).
# Single source of truth used by detect_code_language and strip_code_markers.
_LANGUAGE_MARKERS: dict[str, tuple[str, str]] = {
    "#!R": ("r", "R REPL"),
    "# R code": ("r", "R REPL"),
    "# R script": ("r", "R REPL"),
    "#!BASH": ("bash", "Bash Script"),
    "# Bash script": ("bash", "Bash Script"),
    "#!CLI": ("bash", "CLI Command"),
}

# Strip patterns keyed by language, derived from the markers above.
_STRIP_PATTERNS: dict[str, str] = {
    "r": r"^#!R|^# R code|^# R script",
    "bash": r"^#!BASH|^# Bash script|^#!CLI",
}


def _run_script_in_tempfile(code: str, suffix: str, build_command) -> str:
    """Write *code* to a temp file, run it via subprocess, and return stdout.

    Args:
        code: Source code to write to the temp file.
        suffix: File suffix, e.g. ``".R"`` or ``".sh"``.
        build_command: Callable ``(temp_file_path: str) -> list[str]`` that
            returns the subprocess argv. May perform side-effects like chmod.

    Returns:
        Stdout string on success, or an error string on failure.
    """
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name

        command = build_command(temp_file)
        result = subprocess.run(command, capture_output=True, text=True, check=False)

        if result.returncode != 0:
            return f"Error (exit code {result.returncode}):\n{result.stderr}"
        return result.stdout

    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.unlink(temp_file)
            except OSError:
                pass


def run_r_code(code: str) -> str:
    """Run R code using subprocess.

    Args:
        code: R code to run

    Returns:
        Output of the R code

    """
    return _run_script_in_tempfile(
        code,
        suffix=".R",
        build_command=lambda path: ["Rscript", path],
    )


def run_bash_script(script: str) -> str:
    """Run a Bash script using subprocess.

    Args:
        script: Bash script to run

    Returns:
        Output of the Bash script

    Example:
        This is how to use the function

        .. code-block:: python

            # Example of a complex Bash script
            script = '''
            #!/bin/bash

            # Define variables
            DATA_DIR="/path/to/data"
            OUTPUT_FILE="results.txt"

            # Create output directory if it doesn't exist
            mkdir -p $(dirname $OUTPUT_FILE)

            # Loop through files
            for file in $DATA_DIR/*.txt; do
                echo "Processing $file..."
                # Count lines in each file
                line_count=$(wc -l < $file)
                echo "$file: $line_count lines" >> $OUTPUT_FILE
            done

            echo "Processing complete. Results saved to $OUTPUT_FILE"
            '''
            result = run_bash_script(script)
            print(result)

    """
    script = script.strip()
    if not script:
        return "Error: Empty script"

    lines = []
    if not script.startswith("#!/"):
        lines.append("#!/bin/bash")
    if "set -e" not in script:
        lines.append("set -e")
    lines.append(script)
    full_script = "\n".join(lines)

    def _build_cmd(path: str) -> list[str]:
        os.chmod(path, 0o755)
        return [path]

    return _run_script_in_tempfile(full_script, ".sh", _build_cmd)


def run_with_timeout(func, args=None, kwargs=None, timeout=600):
    """Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    result_queue = queue.Queue()

    def thread_func(func, args, kwargs, result_queue):
        """Function to run in a separate thread."""
        try:
            result = func(*args, **kwargs)
            result_queue.put(("success", result))
        except Exception as e:
            result_queue.put(("error", str(e)))

    # Start a separate thread
    thread = threading.Thread(target=thread_func, args=(func, args, kwargs, result_queue))
    thread.daemon = True  # Set as daemon so it will be killed when main thread exits
    thread.start()

    # Wait for the specified timeout
    thread.join(timeout)

    # Check if the thread is still running after timeout
    if thread.is_alive():
        print(f"TIMEOUT: Code execution timed out after {timeout} seconds")

        # Unfortunately, there's no clean way to force terminate a thread in Python
        # The recommended approach is to use daemon threads and let them be killed when main thread exits
        # Here, we'll try to raise an exception in the thread to make it stop
        try:
            thread_id = thread.ident
            if thread_id:
                # This is a bit dangerous and not 100% reliable
                # It attempts to raise a SystemExit exception in the thread
                res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), ctypes.py_object(SystemExit))
                if res > 1:
                    # Oops, we raised too many exceptions
                    ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(thread_id), None)
        except Exception as e:
            print(f"Error trying to terminate thread: {e}")

        return f"ERROR: Code execution timed out after {timeout} seconds. Please try with simpler inputs or break your task into smaller steps."

    # Get the result from the queue if available
    try:
        status, result = result_queue.get(block=False)
        return result if status == "success" else f"Error in execution: {result}"
    except queue.Empty:
        return "Error: Execution completed but no result was returned"


def detect_code_language(code: str) -> tuple[str, str]:
    """Detect the programming language from code markers.

    Args:
        code: Code content to analyze

    Returns:
        Tuple of (language, tool_name) where language is one of "python", "r", "bash"
        and tool_name is a human-readable label.
    """
    for marker, result in _LANGUAGE_MARKERS.items():
        if code.startswith(marker):
            return result
    return "python", "Python REPL"


def strip_code_markers(code: str, language: str) -> str:
    """Remove language-specific markers from the start of code.

    Args:
        code: Raw code content that may contain language markers
        language: The detected language ("python", "r", "bash")

    Returns:
        Code with markers stripped.
    """
    pattern = _STRIP_PATTERNS.get(language)
    if pattern:
        return re.sub(pattern, "", code, count=1).strip()
    return code
