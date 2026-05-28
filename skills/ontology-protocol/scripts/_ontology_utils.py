"""Shared helpers for inspect_ontology.py and edit_ontology.py."""

import re
from pathlib import Path


_ABOUT_RE = re.compile(r'rdf:about="[^"]*#([^"]+)"')


def _read_rdf(ontology_path: str | Path) -> str:
    return Path(ontology_path).read_text()


def _get_base_iri(rdf_text: str) -> str:
    """Parse the base IRI from xml:base or owl:Ontology rdf:about."""
    m = re.search(r'xml:base="([^"]+)"', rdf_text)
    if m:
        return m.group(1)
    m = re.search(r'owl:Ontology rdf:about="([^"]+)"', rdf_text)
    if m:
        return m.group(1)
    raise ValueError("Could not determine base IRI from ontology file.")


def _extract_names(rdf_text: str, owl_type: str) -> list[str]:
    """Return sorted local names for all declarations of owl_type."""
    names: set[str] = set()
    inside = False
    for line in rdf_text.splitlines():
        stripped = line.strip()
        if f"<{owl_type} " in stripped or f"<{owl_type}>" in stripped:
            inside = True
        if inside:
            m = _ABOUT_RE.search(stripped)
            if m:
                names.add(m.group(1))
                inside = False
        if inside and "/>" in stripped:
            inside = False
    return sorted(names)
