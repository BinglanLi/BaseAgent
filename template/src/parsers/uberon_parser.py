"""
Uberon Anatomy Parser for the knowledge graph.

Downloads and parses the Uberon anatomy ontology (basic.obo + human-view.obo)
to extract BodyPart (Anatomy) nodes with hierarchy encoded as in-row columns.

Data Sources:
  - http://purl.obolibrary.org/obo/uberon/basic.obo        (core anatomy ontology)
  - http://purl.obolibrary.org/obo/uberon/subsets/human-view.obo  (human-specific subset)

Output (written to data/processed/uberon/):
  - uberon_nodes.tsv  : Filtered BodyPart nodes (uberon_slim, human-view)
                        with xrefs, human flag, and in-row is_a / part_of columns.

Node filtering logic:
  - Keep only terms present in human-view.obo  (is_human == 1)
  - Keep only terms tagged with the uberon_slim subset
  - Exclude terms whose subsets contain non_informative, upper_level, or grouping_class

No edges are produced by this parser.
No credentials required.
No disease-specific values are hardcoded.
"""

import logging
import re
from typing import Dict, Set

import pandas as pd

try:
    import obonet
except ImportError:
    obonet = None

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URLs and filenames
# ---------------------------------------------------------------------------

UBERON_BASIC_URL  = "http://purl.obolibrary.org/obo/uberon/basic.obo"
HUMAN_VIEW_URL    = "http://purl.obolibrary.org/obo/uberon/subsets/human-view.obo"

UBERON_BASIC_FILE = "basic.obo"
HUMAN_VIEW_FILE   = "human-view.obo"

# ---------------------------------------------------------------------------
# Output TSV stem (must match source_filename in ontology_mappings.yaml)
# ---------------------------------------------------------------------------

NODES_OUTPUT = "uberon_nodes"

# ---------------------------------------------------------------------------
# Filtering constants — no disease-specific values hardcoded
# ---------------------------------------------------------------------------

EXCLUDE_SUBSETS = frozenset({"non_informative", "upper_level", "grouping_class"})
INCLUDE_SUBSET  = "uberon_slim"


class UberonParser(BaseParser):
    """
    Parser for the Uberon anatomy ontology.

    Downloads basic.obo and human-view.obo, extracts all UBERON:* terms,
    applies the human-slim filter, and returns a single node DataFrame:
      - uberon_nodes.tsv : filtered BodyPart node table

    Columns:
      uberon_id, uberon_name, synonyms, definition, mesh_id, fma_id,
      bto_id, subsets, is_human, is_a, part_of

    No edges are produced.  No credentials required.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        # Override source_name to guarantee the processed subdir is "uberon"
        self.source_name = "uberon"
        self.source_dir  = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        logger.info("Downloading Uberon ontology files ...")
        ok_basic = self.download_file(UBERON_BASIC_URL, UBERON_BASIC_FILE)
        if not ok_basic:
            logger.error("Failed to download basic.obo")
            return False
        ok_human = self.download_file(HUMAN_VIEW_URL, HUMAN_VIEW_FILE)
        if not ok_human:
            logger.error("Failed to download human-view.obo")
            return False
        logger.info("Uberon OBO files are ready.")
        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        basic_path = self.source_dir / UBERON_BASIC_FILE
        human_path = self.source_dir / HUMAN_VIEW_FILE

        if not basic_path.exists():
            logger.error("basic.obo not found: %s", basic_path)
            return {}
        if not human_path.exists():
            logger.error("human-view.obo not found: %s", human_path)
            return {}

        if obonet is None:
            logger.error("obonet is not installed; cannot parse OBO files")
            return {}

        # 1. Load basic.obo
        logger.info("Loading %s ...", basic_path)
        basic_graph = obonet.read_obo(str(basic_path))
        logger.info("basic.obo: %d nodes loaded", basic_graph.number_of_nodes())

        # 2. Load human-view.obo; collect the set of UBERON IDs present in it
        logger.info("Loading %s ...", human_path)
        human_graph = obonet.read_obo(str(human_path))
        human_ids: Set[str] = {
            nid for nid in human_graph.nodes()
            if str(nid).startswith("UBERON:")
        }
        logger.info("human-view.obo: %d UBERON IDs", len(human_ids))

        # 3. Extract all UBERON:* terms from basic.obo
        rows = []
        for node_id, node_data in basic_graph.nodes(data=True):
            node_id = str(node_id)
            if not node_id.startswith("UBERON:"):
                continue
            if node_data.get("is_obsolete", False):
                continue

            # Collect is_a parent IDs (pipe-delimited)
            is_a_ids = []
            for entry in node_data.get("is_a", []):
                pid = self._extract_id(str(entry))
                if pid and pid.startswith("UBERON:"):
                    is_a_ids.append(pid)

            # Collect part_of parent IDs (pipe-delimited)
            part_of_ids = []
            for entry in node_data.get("relationship", []):
                entry_str = str(entry)
                if entry_str.startswith("part_of "):
                    pid = self._extract_id(entry_str[len("part_of "):])
                    if pid and pid.startswith("UBERON:"):
                        part_of_ids.append(pid)

            mesh_id, bto_id, fma_id = self._parse_xrefs(node_data.get("xref", []))
            subsets = "|".join(node_data.get("subset", []))

            rows.append({
                "uberon_id":   node_id,
                "uberon_name": node_data.get("name", ""),
                "synonyms":    self._parse_synonyms(node_data.get("synonym", [])),
                "definition":  self._parse_definition(node_data.get("def", "")),
                "mesh_id":     mesh_id,
                "fma_id":      fma_id,
                "bto_id":      bto_id,
                "subsets":     subsets,
                "is_human":    1 if node_id in human_ids else 0,
                "is_a":        "|".join(is_a_ids),
                "part_of":     "|".join(part_of_ids),
            })

        logger.info("Extracted %d UBERON terms before filtering", len(rows))

        nodes_df = pd.DataFrame(rows)

        # 4. Apply filter: human-view + uberon_slim, excluding noisy subsets
        nodes_filtered = self._apply_filter(nodes_df)
        logger.info(
            "After filtering (uberon_slim + human-view, excl. non_informative/upper_level/grouping_class): %d nodes",
            len(nodes_filtered),
        )

        # 5. Enforce exact column order required by ontology_mappings.yaml
        col_order = [
            "uberon_id", "uberon_name", "synonyms", "definition",
            "mesh_id", "fma_id", "bto_id", "subsets",
            "is_human", "is_a", "part_of",
        ]
        nodes_out = nodes_filtered[[c for c in col_order if c in nodes_filtered.columns]].copy()
        nodes_out["source_database"] = "Uberon"

        return {NODES_OUTPUT: nodes_out}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_synonyms(synonym_list) -> str:
        """Extract synonym text from OBO synonym strings; return pipe-delimited."""
        texts = []
        for syn in synonym_list:
            # OBO synonym format: "text" TYPE [refs]
            m = re.match(r'^"(.*?)"\s+\w', str(syn))
            if m:
                texts.append(m.group(1))
            else:
                cleaned = str(syn).strip('"').split('"')[0]
                texts.append(cleaned)
        return "|".join(t for t in texts if t)

    @staticmethod
    def _parse_xrefs(xref_list):
        """
        Parse xref list and return (mesh_id, bto_id, fma_id) as pipe-delimited strings.
        Handles MESH:, MSH:, MeSH:, BTO:, FMA: prefixes.
        """
        mesh_vals = []
        bto_vals  = []
        fma_vals  = []

        for xref in xref_list:
            xref_str = str(xref).strip()
            upper    = xref_str.upper()
            if upper.startswith("MESH:") or upper.startswith("MSH:"):
                mesh_vals.append(xref_str)
            elif upper.startswith("BTO:"):
                bto_vals.append(xref_str)
            elif upper.startswith("FMA:"):
                fma_vals.append(xref_str)

        return (
            "|".join(mesh_vals) if mesh_vals else "",
            "|".join(bto_vals)  if bto_vals  else "",
            "|".join(fma_vals)  if fma_vals  else "",
        )

    @staticmethod
    def _extract_id(text: str) -> str:
        """Extract the first whitespace-delimited token (the CURIE ID)."""
        m = re.match(r"^(\S+)", text.strip())
        return m.group(1) if m else ""

    @staticmethod
    def _parse_definition(raw: str) -> str:
        """Extract plain text from an OBO def: field.

        OBO format: '"text" [citation1, citation2]'
        Returns the text between the first pair of double-quotes.
        """
        m = re.match(r'^"(.*?)"', raw.strip())
        return m.group(1) if m else ""

    @staticmethod
    def _apply_filter(df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep rows where:
          - is_human == 1
          - subsets contains INCLUDE_SUBSET (uberon_slim)
          - subsets does NOT contain any tag in EXCLUDE_SUBSETS
        """
        if df.empty:
            return df

        def passes(row) -> bool:
            if not row["is_human"]:
                return False
            subset_tags = set(row["subsets"].split("|")) if row["subsets"] else set()
            if INCLUDE_SUBSET not in subset_tags:
                return False
            if subset_tags & EXCLUDE_SUBSETS:
                return False
            return True

        mask = df.apply(passes, axis=1)
        return df[mask].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        return {
            NODES_OUTPUT: {
                "uberon_id":   "Uberon anatomy ID (e.g., UBERON:0000955)",
                "uberon_name": "Anatomical structure name",
                "synonyms":    "Pipe-delimited list of synonym strings",
                "definition":  "Anatomical structure definition text (from OBO def: field)",
                "mesh_id":     "MeSH cross-reference ID(s), pipe-delimited (MESH: or MSH: prefix)",
                "fma_id":      "FMA cross-reference ID(s), pipe-delimited",
                "bto_id":      "BTO cross-reference ID(s), pipe-delimited",
                "subsets":     "Pipe-delimited list of subset tags",
                "is_human":    "1 if term is present in human-view.obo, 0 otherwise",
                "is_a":        "Pipe-delimited parent UBERON IDs via is_a relationships",
                "part_of":     "Pipe-delimited parent UBERON IDs via part_of relationships",
            },
        }
