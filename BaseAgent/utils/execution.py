"""Execution utilities: run R code, Bash scripts, and Python with timeout."""

import os
import re
import subprocess
import tempfile
import traceback


def run_r_code(code: str) -> str:
    """Run R code using subprocess.

    Args:
        code: R code to run

    Returns:
        Output of the R code

    """
    try:
        # Create a temporary file to store the R code
        with tempfile.NamedTemporaryFile(suffix=".R", mode="w", delete=False) as f:
            f.write(code)
            temp_file = f.name

        # Run the R code using Rscript
        result = subprocess.run(["Rscript", temp_file], capture_output=True, text=True, check=False)

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            return f"Error running R code:\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        return f"Error running R code: {str(e)}"


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
    try:
        # Trim any leading/trailing whitespace
        script = script.strip()

        # If the script is empty, return an error
        if not script:
            return "Error: Empty script"

        # Create a temporary file to store the Bash script
        with tempfile.NamedTemporaryFile(suffix=".sh", mode="w", delete=False) as f:
            # Add shebang if not present
            if not script.startswith("#!/"):
                f.write("#!/bin/bash\n")
            # Add set -e to exit on error
            if "set -e" not in script:
                f.write("set -e\n")
            f.write(script)
            temp_file = f.name

        # Make the script executable
        os.chmod(temp_file, 0o755)

        # Get current environment variables and working directory
        env = os.environ.copy()
        cwd = os.getcwd()

        # Run the Bash script with the current environment and working directory
        result = subprocess.run(
            [temp_file],
            shell=True,
            capture_output=True,
            text=True,
            check=False,
            env=env,
            cwd=cwd,
        )

        # Clean up the temporary file
        os.unlink(temp_file)

        # Return the output
        if result.returncode != 0:
            traceback.print_stack()
            print(result)
            return f"Error running Bash script (exit code {result.returncode}):\n{result.stderr}"
        else:
            return result.stdout
    except Exception as e:
        traceback.print_exc()
        return f"Error running Bash script: {str(e)}"


def run_with_timeout(func, args=None, kwargs=None, timeout=600):
    """Run a function with a timeout using threading instead of multiprocessing.
    This allows variables to persist in the global namespace between function calls.
    Returns the function result or a timeout error message.
    """
    if args is None:
        args = []
    if kwargs is None:
        kwargs = {}

    import ctypes
    import queue
    import threading

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
            # Get thread ID and try to terminate it
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
    if code.startswith("#!R") or code.startswith("# R code") or code.startswith("# R script"):
        return "r", "R REPL"
    elif code.startswith("#!BASH") or code.startswith("# Bash script"):
        return "bash", "Bash Script"
    elif code.startswith("#!CLI"):
        return "bash", "CLI Command"
    else:
        return "python", "Python REPL"


def strip_code_markers(code: str, language: str) -> str:
    """Remove language-specific markers from the start of code.

    Args:
        code: Raw code content that may contain language markers
        language: The detected language ("python", "r", "bash")

    Returns:
        Code with markers stripped.
    """
    if language == "r":
        return re.sub(r"^#!R|^# R code|^# R script", "", code, count=1).strip()
    elif language == "bash":
        return re.sub(r"^#!BASH|^# Bash script|^#!CLI", "", code, count=1).strip()
    return code
