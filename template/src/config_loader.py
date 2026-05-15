"""
config_loader.py — Shared utility for loading project configuration.

Provides a single function, load_project_config(), that reads
config/project.yaml and returns the parsed project section.
All parsers and pipeline components should use this to access
project-level settings (e.g., disease_scope) rather than
hard-coding any domain-specific values.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Locate config/ relative to this file (src/config_loader.py → root/config/)
_CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


@lru_cache(maxsize=1)
def load_project_config() -> Dict[str, Any]:
    """
    Load and return the 'project' section of config/project.yaml.

    The result is cached after the first call so the file is only
    read once per process.

    Returns:
        Dictionary containing the full project configuration, including
        'disease_scope', 'ontology', 'node_types', 'edge_types', etc.

    Raises:
        FileNotFoundError: If config/project.yaml does not exist.
        yaml.YAMLError: If the file cannot be parsed.
    """
    config_path = _CONFIG_DIR / "project.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Project configuration file not found: {config_path}. "
            "Ensure config/project.yaml exists at the repository root."
        )
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = raw.get("project", raw)  # tolerate missing top-level key
    logger.debug("Loaded project config from %s", config_path)
    return project


def get_disease_scope() -> Dict[str, Any]:
    """
    Convenience wrapper that returns only the disease_scope block.

    Returns:
        The 'disease_scope' dict from project.yaml, or an empty dict
        if the key is absent.
    """
    return load_project_config().get("disease_scope", {})
