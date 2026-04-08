"""BaseAgent utility package.

Import from specific submodules for clarity, or use this package init for
convenience — core public names are re-exported here.
"""

from BaseAgent.utils.execution import run_with_timeout, run_bash_script, run_r_code, detect_code_language, strip_code_markers
from BaseAgent.utils.schema import (
    extract_schema_from_function,
    function_to_api_schema,
    enhance_description_with_llm,
    enhance_parameters_with_llm,
)
from BaseAgent.utils.formatting import (
    pretty_print,
    color_print,
    wrap_text,
    clean_message_content,
    langchain_to_gradio_message,
)
from BaseAgent.utils.download import (
    check_and_download_s3_files,
    download_and_unzip,
    check_or_create_path,
)
from BaseAgent.utils.tool_bridge import inject_custom_functions_to_repl

__all__ = [
    # execution
    "run_with_timeout",
    "run_bash_script",
    "run_r_code",
    "detect_code_language",
    "strip_code_markers",
    # schema
    "extract_schema_from_function",
    "function_to_api_schema",
    "enhance_description_with_llm",
    "enhance_parameters_with_llm",
    # formatting
    "pretty_print",
    "color_print",
    "wrap_text",
    "clean_message_content",
    "langchain_to_gradio_message",
    # download
    "check_and_download_s3_files",
    "download_and_unzip",
    "check_or_create_path",
    # tool_bridge
    "inject_custom_functions_to_repl",
]
