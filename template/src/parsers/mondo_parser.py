"""
MONDO (Monarch Disease Ontology) Parser for the knowledge graph.

Downloads mondo.obo and extracts Disease nodes with cross-references.
MONDO IDs serve as the canonical disease identifier in the KG; UMLS CUIs
are extracted from xrefs (UMLS:CXXXXXXX entries) embedded in the OBO file,
so no separate UMLS download is needed.

Data Source:
  - http://purl.obolibrary.org/obo/mondo.obo  (public, no credentials required)

Output (written to data/mondo/):
  - disease_nodes.tsv  — MONDO disease nodes with cross-references

Filtering:
  - Excludes obsolete terms
  - Excludes grouping/subset terms tagged as non-disease categories
  - If disease_scope is configured in project.yaml, keeps only terms that
    match by MONDO ID, OMIM ID, UMLS CUI, or disease name substring.
    If disease_scope is empty, all non-obsolete MONDO:* terms are returned.

Cross-references extracted from xrefs field:
  UMLS:CXXXXXXX → umls_cui (first value; this is the primary mapping target)
  OMIM:XXXXXX   → omim_id
  Orphanet:XXXX → orpha_id
  DOID:XXXXXXX  → doid
  ICD10CM:XXX   → icd10cm
  ICD10:XXX     → icd10 (WHO ICD-10, when ICD10CM absent)
  MESH:DXXXXXX  → mesh_id
  SNOMEDCT:XXXX → snomed_id
"""

import logging
import re
from typing import Dict, List, Optional

import pandas as pd

try:
    import obonet
    HAS_OBONET = True
except ImportError:
    HAS_OBONET = False

from .base_parser import BaseParser
from config_loader import get_disease_scope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Source URL and local filename
# ---------------------------------------------------------------------------

MONDO_OBO_URL  = "http://purl.obolibrary.org/obo/mondo.obo"
MONDO_OBO_FILE = "mondo.obo"

# ---------------------------------------------------------------------------
# Output table name — must match source_filename in ontology_mappings.yaml
# ---------------------------------------------------------------------------

DISEASE_NODES = "disease_nodes"

# MONDO subsets that mark non-disease grouping terms to exclude
_EXCLUDE_SUBSETS = frozenset({
    "non_grouping",
    "grouping_class",
    "upper_level",
    "non_informative",
})


class MONDOParser(BaseParser):
    """
    Parser for the Monarch Disease Ontology (MONDO).

    Extracts Disease nodes from mondo.obo. UMLS CUI mappings are embedded
    as xref entries (UMLS:CXXXXXXX) in the OBO file — no separate UMLS
    download is required.

    Outputs
    -------
    disease_nodes
        One row per non-obsolete MONDO:* term that passes scope filtering.
        Columns: mondo_id, disease_name, definition, synonyms, is_a,
                 umls_cui, omim_id, orpha_id, doid, icd10cm, mesh_id,
                 snomed_id
    """

    def __init__(self, data_dir: str, disease_scope: Optional[Dict] = None):
        """
        Parameters
        ----------
        data_dir:
            Directory for storing raw data files.
        disease_scope:
            Disease scope dict from project config (auto-injected by pipeline).
            Falls back to config/project.yaml via get_disease_scope().
        """
        super().__init__(data_dir)
        self.source_name = "mondo"
        self.source_dir  = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

        _cfg_scope = disease_scope if disease_scope is not None else get_disease_scope()
        self._scope_mondo_ids:   set = set(_cfg_scope.get("mondo_ids", []))
        self._scope_omim_ids:    set = set(_cfg_scope.get("omim_ids", []))
        self._scope_umls_cuis:   set = set(_cfg_scope.get("umls_cuis", []))
        self._scope_terms:      List[str] = [
            t.lower() for t in _cfg_scope.get("primary_terms", [])
        ]

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Download the MONDO OBO file."""
        if not HAS_OBONET:
            logger.error("obonet is not installed — install with: pip install obonet")
            return False

        logger.info("Downloading MONDO OBO file …")
        if not self.download_file(MONDO_OBO_URL, MONDO_OBO_FILE):
            logger.error("Failed to download mondo.obo")
            return False

        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse mondo.obo and return a disease node DataFrame.

        Returns
        -------
        dict with key:
          disease_nodes  — MONDO disease node table
        """
        if not HAS_OBONET:
            logger.error("obonet is not installed — cannot parse mondo.obo")
            return {}

        obo_path = self.source_dir / MONDO_OBO_FILE
        if not obo_path.exists():
            logger.error("mondo.obo not found: %s", obo_path)
            return {}

        logger.info("Parsing MONDO from %s …", obo_path)
        try:
            graph = obonet.read_obo(str(obo_path))
        except Exception as exc:
            logger.error("Failed to read mondo.obo: %s", exc)
            return {}

        logger.info("mondo.obo: %d total nodes loaded", graph.number_of_nodes())

        rows = []
        for node_id, data in graph.nodes(data=True):
            node_id = str(node_id)
            if not node_id.startswith("MONDO:"):
                continue
            if data.get("is_obsolete", False):
                continue

            # Exclude pure grouping / non-disease subset terms
            subsets = set(data.get("subset", []))
            if subsets & _EXCLUDE_SUBSETS:
                continue

            xrefs = data.get("xref", [])
            umls_list  = self._xrefs_with_prefix(xrefs, "UMLS:")
            omim_list  = self._xrefs_with_prefix(xrefs, "OMIM:")
            orpha_list = self._xrefs_with_prefix(xrefs, "Orphanet:")
            doid_list  = self._xrefs_with_prefix(xrefs, "DOID:")
            # Prefer ICD10CM; fall back to ICD10 (WHO variant)
            icd10_list = (
                self._xrefs_with_prefix(xrefs, "ICD10CM:")
                or self._xrefs_with_prefix(xrefs, "ICD10:")
            )
            mesh_list   = self._xrefs_with_prefix(xrefs, "MESH:")
            snomed_list = self._xrefs_with_prefix(xrefs, "SNOMEDCT:")

            name = data.get("name", "")

            if not self._in_scope(node_id, name, umls_list, omim_list):
                continue

            is_a_ids = []
            for entry in data.get("is_a", []):
                pid = self._extract_id(str(entry))
                if pid.startswith("MONDO:"):
                    is_a_ids.append(pid)

            rows.append({
                "mondo_id":     node_id,
                "disease_name": name,
                "definition":   self._clean_definition(data.get("def", "")),
                "synonyms":     self._parse_synonyms(data.get("synonym", [])),
                "is_a":         "|".join(is_a_ids),
                "umls_cui":     umls_list[0] if umls_list else "",
                "omim_id":      omim_list[0] if omim_list else "",
                "orpha_id":     orpha_list[0] if orpha_list else "",
                "doid":         doid_list[0] if doid_list else "",
                "icd10cm":      icd10_list[0] if icd10_list else "",
                "mesh_id":      mesh_list[0] if mesh_list else "",
                "snomed_id":    snomed_list[0] if snomed_list else "",
            })

        logger.info("Extracted %d MONDO disease nodes", len(rows))

        if not rows:
            logger.warning(
                "No MONDO terms passed filters. "
                "Check disease_scope settings in config/project.yaml."
            )
            return {}

        df = pd.DataFrame(rows)
        df["source_database"] = "MONDO"
        return {DISEASE_NODES: df}

    # ------------------------------------------------------------------
    # Scope filter
    # ------------------------------------------------------------------

    def _in_scope(
        self,
        mondo_id: str,
        name: str,
        umls_list: List[str],
        omim_list: List[str],
    ) -> bool:
        """Return True if this term matches the configured disease scope.

        When no scope is configured (all scope sets are empty), every term passes.
        """
        no_scope = not any([
            self._scope_mondo_ids,
            self._scope_omim_ids,
            self._scope_umls_cuis,
            self._scope_terms,
        ])
        if no_scope:
            return True

        if mondo_id in self._scope_mondo_ids:
            return True
        if any(omim in self._scope_omim_ids for omim in omim_list):
            return True
        if any(cui in self._scope_umls_cuis for cui in umls_list):
            return True
        name_lower = name.lower()
        if any(term in name_lower for term in self._scope_terms):
            return True
        return False

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _xrefs_with_prefix(xrefs: list, prefix: str) -> List[str]:
        """Return xref values that start with prefix, with the prefix stripped."""
        return [x[len(prefix):] for x in xrefs if str(x).startswith(prefix)]

    @staticmethod
    def _extract_id(text: str) -> str:
        """Extract the first whitespace-delimited token (the CURIE)."""
        m = re.match(r"^(\S+)", text.strip())
        return m.group(1) if m else ""

    @staticmethod
    def _clean_definition(raw: str) -> str:
        """Strip OBO-format quotes and citation brackets from a def: field."""
        if not raw:
            return ""
        if raw.startswith('"'):
            raw = raw[1:]
        bracket_idx = raw.rfind(" [")
        if bracket_idx != -1:
            raw = raw[:bracket_idx]
        return raw.rstrip('"').strip()

    @staticmethod
    def _parse_synonyms(synonym_list) -> str:
        """Extract synonym text from OBO synonym strings; return pipe-delimited."""
        texts = []
        for syn in synonym_list:
            m = re.match(r'^"(.*?)"\s+\w', str(syn))
            if m:
                texts.append(m.group(1))
            else:
                cleaned = str(syn).strip('"').split('"')[0]
                texts.append(cleaned)
        return "|".join(t for t in texts if t)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return column schema for the disease_nodes output DataFrame."""
        return {
            DISEASE_NODES: {
                "mondo_id":      "MONDO disease ID (e.g. MONDO:0004975)",
                "disease_name":  "Disease name",
                "definition":    "Text definition (from OBO def: field)",
                "synonyms":      "Pipe-delimited synonym strings",
                "is_a":          "Pipe-delimited parent MONDO IDs via is_a relationships",
                "umls_cui":      "UMLS CUI cross-reference (first UMLS: xref; primary mapping target)",
                "omim_id":       "OMIM ID cross-reference (first OMIM: xref)",
                "orpha_id":      "Orphanet ID cross-reference (first Orphanet: xref)",
                "doid":          "Disease Ontology ID cross-reference (first DOID: xref)",
                "icd10cm":       "ICD-10-CM code cross-reference (ICD10CM: preferred; ICD10: fallback)",
                "mesh_id":       "MeSH descriptor ID cross-reference (first MESH: xref)",
                "snomed_id":     "SNOMED CT ID cross-reference (first SNOMEDCT: xref)",
                "source_database": "Source name string (MONDO)",
            },
        }
