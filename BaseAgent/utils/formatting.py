"""Formatting utilities: pretty-printing, color output, text wrapping."""

import re
import textwrap

from langchain_core.messages.base import get_msg_title_repr
from langchain_core.utils.interactive_env import is_interactive_env


_TEXT_COLOR_MAPPING = {
    "blue": "36;1",
    "yellow": "33;1",
    "pink": "38;5;200",
    "green": "32;1",
    "red": "31;1",
}


def wrap_text(text, max_line_length=80):
    """Wrap a text to make it more readable."""
    return textwrap.fill(text, width=max_line_length)


def pretty_print(message, printout=True):
    if isinstance(message, tuple):
        title = message
    elif isinstance(message.content, list):
        title = get_msg_title_repr(message.type.title().upper() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"

        for i in message.content:
            if i["type"] == "text":
                title += f"\n{i['text']}\n"
            elif i["type"] == "tool_use":
                title += f"\nTool: {i['name']}"
                title += f"\nInput: {i['input']}"
        if printout:
            print(f"{title}")
    else:
        title = get_msg_title_repr(message.type.title() + " Message", bold=is_interactive_env())
        if message.name is not None:
            title += f"\nName: {message.name}"
        title += f"\n\n{message.content}"
        if printout:
            print(f"{title}")
    return title


def color_print(text, color="blue"):
    color_str = _TEXT_COLOR_MAPPING[color]
    print(f"\u001b[{color_str}m\033[1;3m{text}\u001b[0m")


def extract_agent_result(text: str) -> str:
    """Return the <solution> content from an agent response, or the cleaned text if absent.

    BaseAgent responses wrap answers in ``<solution>...</solution>`` and reasoning in
    ``<think>...</think>``.  This function extracts the user-facing answer and discards
    internal reasoning so orchestrators receive clean text.

    Falls back to stripping ``<think>``/``<execute>``/``<observation>`` blocks when no
    ``<solution>`` tag is present.  Returns the input unchanged when no known tags are
    found (e.g. ``"ERROR: ..."`` strings pass through as-is).

    Output is truncated to 2 000 characters to keep supervisor prompts manageable.
    """
    m = re.search(r"<solution>(.*?)</solution>", text, re.DOTALL)
    if m:
        return m.group(1).strip()[:2000]
    cleaned = re.sub(
        r"<(?:think|execute|observation)>.*?</(?:think|execute|observation)>",
        "",
        text,
        flags=re.DOTALL,
    )
    return cleaned.strip()[:2000] or text.strip()[:2000]


def clean_message_content(content: str) -> str:
    """Clean message content by removing ANSI escape codes.

    This function removes ANSI escape sequences (like color codes) from text content
    that might be present in terminal output or console messages. This ensures clean
    text for markdown generation and PDF conversion.

    Args:
        content: The raw message content that may contain ANSI escape codes

    Returns:
        Cleaned content with ANSI escape codes removed

    Example:
        >>> clean_message_content("Hello \x1b[31mworld\x1b[0m!")
        "Hello world!"
    """
    return re.sub(r"\x1b\[[0-9;]*m", "", content)


def langchain_to_gradio_message(message):
    # Build the title and content based on the message type
    if isinstance(message.content, list):
        # For a message with multiple content items (like text and tool use)
        gradio_messages = []
        for item in message.content:
            gradio_message = {
                "role": "user" if message.type == "human" else "assistant",
                "content": "",
                "metadata": {},
            }

            if item["type"] == "text":
                item["text"] = item["text"].replace("<think>", "\n")
                item["text"] = item["text"].replace("</think>", "\n")
                gradio_message["content"] += f"{item['text']}\n"
                gradio_messages.append(gradio_message)
            elif item["type"] == "tool_use":
                if item["name"] == "run_python_repl":
                    gradio_message["metadata"]["title"] = "🛠️ Writing code..."
                    gradio_message["metadata"]["log"] = "Executing Code block..."
                    gradio_message["content"] = f"##### Code: \n ```python \n {item['input']['command']} \n``` \n"
                else:
                    gradio_message["metadata"]["title"] = f"🛠️ Used tool ```{item['name']}```"
                    to_print = ";".join([i + ": " + str(j) for i, j in item["input"].items()])
                    gradio_message["metadata"]["log"] = f"🔍 Input -- {to_print}\n"
                gradio_message["metadata"]["status"] = "pending"
                gradio_messages.append(gradio_message)

    else:
        gradio_message = {
            "role": "user" if message.type == "human" else "assistant",
            "content": "",
            "metadata": {},
        }
        content = message.content
        content = content.replace("<think>", "\n")
        content = content.replace("</think>", "\n")
        content = content.replace("<solution>", "\n")
        content = content.replace("</solution>", "\n")

        gradio_message["content"] = content
        gradio_messages = [gradio_message]
    return gradio_messages
