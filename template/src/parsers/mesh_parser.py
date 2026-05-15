"""
MeSH Parser for the knowledge graph.

Parses MeSH (Medical Subject Headings) to extract Symptom nodes
from the Signs and Symptoms subtree (MeSH tree C23.888).

Data Source:
  - MeSH XML: https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/

Output:
  - symptom_nodes.tsv: MeSH Signs and Symptoms descriptors (C23.888 subtree)
    Columns: mesh_id, mesh_name, sourceDatabase
"""

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from lxml import etree

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# Default NLM base URL for MeSH XML files
DEFAULT_BASE_URL = "https://nlmpubs.nlm.nih.gov/projects/mesh/MESH_FILES/xmlmesh/"

# Years to try in descending order when searching for the latest descriptor file
_CANDIDATE_YEARS = [2026, 2025, 2024, 2023]

# Signs and Symptoms subtree root: D012816, tree number C23.888.
# Trailing dot ensures we match descendants only (not the root itself).
SYMPTOM_TREE_PREFIX = "C23.888."

OUTPUT_NAME = "symptom_nodes"


class MeSHParser(BaseParser):
    """Parser for MeSH Signs and Symptoms descriptors (C23.888 subtree)."""

    def __init__(self, data_dir: str, base_url: str = None):
        super().__init__(data_dir)
        self.base_url = (base_url or DEFAULT_BASE_URL).rstrip("/") + "/"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_filenames(self):
        """Return (filename, url) pairs for each candidate year."""
        for year in _CANDIDATE_YEARS:
            fname = f"desc{year}.xml"
            yield fname, self.base_url + fname

    def _cached_xml(self) -> Optional[Path]:
        """Return path to the first already-cached descriptor XML, or None."""
        for fname, _ in self._candidate_filenames():
            p = self.source_dir / fname
            if p.exists() and p.stat().st_size > 1_000_000:   # >1 MB → real file
                logger.info(f"Found cached MeSH XML: {p}")
                return p
        return None

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Download the latest available MeSH descriptor XML from NLM."""
        # If a valid cached file already exists and force is not set, skip download
        if not self.force and self._cached_xml():
            return True

        for fname, url in self._candidate_filenames():
            logger.info(f"Attempting to download {url}")
            result = self.download_file(url, fname)
            if result:
                p = Path(result)
                if p.stat().st_size > 1_000_000:
                    logger.info(f"Successfully downloaded {fname} ({p.stat().st_size // 1024 // 1024} MB)")
                    return True
                else:
                    logger.warning(f"{fname} downloaded but appears empty ({p.stat().st_size} bytes); trying next year")
                    p.unlink(missing_ok=True)

        logger.error("Failed to download any MeSH descriptor XML")
        return False

    def _parse_xml(self, xml_path: Path) -> list:
        """Stream-parse MeSH descriptor XML into a list of dicts.

        Each dict has keys: mesh_id, mesh_name, tree_numbers (list[str]).
        tree_numbers is used for filtering and is not written to the output TSV.
        """
        descriptors = []
        context = etree.iterparse(str(xml_path), events=("end",), tag="DescriptorRecord")
        for _, elem in context:
            ui = elem.findtext(".//DescriptorUI")
            name = elem.findtext(".//DescriptorName/String")
            if ui and name:
                tree_numbers = [tn.text for tn in elem.findall(".//TreeNumber") if tn.text]
                descriptors.append({"mesh_id": ui, "mesh_name": name, "tree_numbers": tree_numbers})
            elem.clear()
        logger.info(f"Parsed {len(descriptors):,} MeSH descriptors from {xml_path.name}")
        return descriptors

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """Parse MeSH XML and return symptom nodes under C23.888.

        Returns:
            {'symptom_nodes': DataFrame with columns mesh_id, mesh_name, sourceDatabase}
        """
        xml_path = self._cached_xml()
        if xml_path is None:
            logger.error("No valid MeSH XML found in source directory; run download first")
            return {}

        all_terms = self._parse_xml(xml_path)

        # Filter to C23.888 subtree descendants (trailing dot excludes root D012816)
        symptom_terms = [
            t for t in all_terms
            if any(tn.startswith(SYMPTOM_TREE_PREFIX) for tn in t["tree_numbers"])
        ]

        logger.info(f"Extracted {len(symptom_terms):,} symptom terms from C23.888 subtree")
        if len(symptom_terms) < 400:
            logger.warning(
                f"Unexpectedly few symptom terms: {len(symptom_terms)} (expected ≥400)"
            )

        symptom_df = pd.DataFrame(
            [
                {"mesh_id": t["mesh_id"], "mesh_name": t["mesh_name"], "sourceDatabase": "mesh"}
                for t in symptom_terms
            ]
        )

        return {OUTPUT_NAME: symptom_df}

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        return {
            OUTPUT_NAME: {
                "mesh_id": "MeSH Descriptor UI (e.g., D005221)",
                "mesh_name": "Symptom preferred name from MeSH",
                "sourceDatabase": "Source database identifier (mesh)",
            }
        }
