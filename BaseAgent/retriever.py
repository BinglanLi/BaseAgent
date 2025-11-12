import contextlib
import re

from langchain_core.messages import HumanMessage    
from langchain_core.language_models.chat_models import BaseChatModel

from BaseAgent.prompts import _RESOURCE_SELECTION_PROMPT

class ToolRetriever:
    """
    Use a prompt-based approach to retrieve the most relevant resources for a query.
    """

    def __init__(self):
        pass

    def prompt_based_retrieval(
        self, 
        query: str, 
        resources: dict[str, list[dict[str, str]]], 
        llm: BaseChatModel | None = None
        ) -> dict[str, list[dict[str, str]]]:
        """
        Use a prompt-based approach to retrieve the most relevant resources for a query.

        Args:
            query: The user's query
            resources: A dictionary with keys 'all_tools', 'all_data', and 'all_libraries',
                      each containing a list of resource dictionaries with 'name' and 'description' keys
            llm: LLM instance to use for retrieval (required)

        Returns:
            A dictionary with keys 'selected_tools', 'selected_data', and 'selected_libraries',
            containing the most relevant resources

        """
        # Extract resources from input (keys: all_tools, all_data, all_libraries)
        all_tools = resources.get("all_tools", [])
        all_data = resources.get("all_data", [])
        all_libraries = resources.get("all_libraries", [])
        
        # Create a prompt for the LLM to select relevant resources
        prompt = _RESOURCE_SELECTION_PROMPT.format(
            query=query,
            tools=self._format_resources_for_prompt(all_tools),
            data=self._format_resources_for_prompt(all_data),
            libraries=self._format_resources_for_prompt(all_libraries),
        )

        # Invoke the LLM
        response = llm.invoke([HumanMessage(content=prompt)])
        response_content = response.content

        # Parse the response to extract the selected indices
        selected_indices = self._parse_llm_response(response_content)

        # Get the selected resources (return keys: selected_tools, selected_data, selected_libraries)
        selected_resources = {
            "selected_tools": [
                all_tools[i] 
                for i in selected_indices.get("selected_tools", []) 
                if i < len(all_tools)
            ],
            "selected_data": [
                all_data[i]
                for i in selected_indices.get("selected_data", [])
                if i < len(all_data)
            ],
            "selected_libraries": [
                all_libraries[i]
                for i in selected_indices.get("selected_libraries", [])
                if i < len(all_libraries)
            ],
        }

        return selected_resources

    def _format_resources_for_prompt(self, resources: list[dict[str, str]]) -> str:
        """
        Format resources for inclusion in the prompt.

        Args:
            resources: List of resource dictionaries

        Returns:
            Formatted string with numbered resource list
        """
        formatted = []
        for i, resource in enumerate(resources):
            formatted.append(f"{i}. {resource['name']}: {resource['description']}")

        return "\n".join(formatted) if formatted else "None available"

    def _parse_llm_response(self, response: str) -> dict[str, list[int]]:
        """
        Parse the LLM response to extract the selected indices.
        """
        selected_indices = {"selected_tools": [], "selected_data": [], "selected_libraries": []}

        # Extract indices for each category
        tools_match = re.search(r"TOOLS:\s*\[(.*?)\]", response, re.IGNORECASE)
        if tools_match and tools_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["selected_tools"] = [int(idx.strip()) for idx in tools_match.group(1).split(",") if idx.strip()]

        data_match = re.search(r"DATA:\s*\[(.*?)\]", response, re.IGNORECASE)
        if data_match and data_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["selected_data"] = [
                    int(idx.strip()) for idx in data_match.group(1).split(",") if idx.strip()
                ]

        libraries_match = re.search(r"LIBRARIES:\s*\[(.*?)\]", response, re.IGNORECASE)
        if libraries_match and libraries_match.group(1).strip():
            with contextlib.suppress(ValueError):
                selected_indices["selected_libraries"] = [
                    int(idx.strip()) for idx in libraries_match.group(1).split(",") if idx.strip()
                ]

        return selected_indices
