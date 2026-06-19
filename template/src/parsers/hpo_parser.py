"""
Human Phenotype Ontology (HPO) Parser for the knowledge graph.

Downloads and parses HPO to extract:
  - effect/phenotype nodes (hp.obo)
  - disease_phenotype_positive edges (HPOA, qualifier != NOT)
  - disease_phenotype_negative edges (HPOA, qualifier == NOT)

Data Sources:
  - http://purl.obolibrary.org/obo/hp.obo
  - http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa
  - https://ftp.ncbi.nlm.nih.gov/pub/medgen/MedGenIDMappings.txt.gz

Output (written to data/hpo/):
  - phenotype_nodes.tsv            — HPO term nodes (hp_id, name, definition, synonyms, is_a)
  - disease_phenotype_positive.tsv — disease→phenotype edges (qualifier != NOT)
                                     columns: hp_id, cui, evidence, onset, frequency, sex
  - disease_phenotype_negative.tsv — disease→phenotype absent edges (qualifier == NOT)
                                     columns: hp_id, cui, evidence, onset, frequency, sex

No credentials required.  No disease-specific values are hardcoded.
"""

import gzip
import logging
import re
from typing import Dict

import pandas as pd

try:
    import obonet
    HAS_OBONET = True
except ImportError:
    HAS_OBONET = False

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# URLs and local filenames
# ---------------------------------------------------------------------------

HPO_OBO_URL  = "http://purl.obolibrary.org/obo/hp.obo"
HPOA_URL     = "http://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa"
MEDGEN_URL   = "https://ftp.ncbi.nlm.nih.gov/pub/medgen/MedGenIDMappings.txt.gz"

HPO_OBO_FILE = "hp.obo"
HPOA_FILE    = "phenotype.hpoa"
MEDGEN_FILE  = "MedGenIDMappings.txt.gz"

# ---------------------------------------------------------------------------
# Output table name constants — must match source_filename in ontology_mappings.yaml
# ---------------------------------------------------------------------------

PHENOTYPE_NODES = "phenotype_nodes"
POSITIVE_EDGES  = "disease_phenotype_positive"
NEGATIVE_EDGES  = "disease_phenotype_negative"

# HPOA column names (from the header line that starts with #)
_HPOA_COLS = [
    "database_id", "disease_name", "qualifier", "hpo_id",
    "reference", "evidence", "onset", "frequency", "sex",
    "modifier", "aspect", "biocuration",
]


class HPOParser(BaseParser):
    """
    Parser for the Human Phenotype Ontology (HPO).

    Extracts phenotype nodes from hp.obo and disease-phenotype associations
    from the HPOA annotation file.  OMIM/ORPHA disease IDs are mapped to
    UMLS CUIs via NCBI MedGen.

    Outputs
    -------
    phenotype_nodes
        One row per non-obsolete HP:* term.
        Columns: hp_id, name, definition, synonyms, is_a
    disease_phenotype_positive
        Phenotype-present edges (HPOA qualifier != NOT).
        Columns: hp_id, cui, evidence, onset, frequency, sex
    disease_phenotype_negative
        Phenotype-absent edges (HPOA qualifier == NOT).
        Columns: hp_id, cui, evidence, onset, frequency, sex
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.source_name = "hpo"
        self.source_dir  = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Download hp.obo, phenotype.hpoa, and MedGen ID mappings."""
        logger.info("Downloading HPO OBO file …")
        if not self.download_file(HPO_OBO_URL, HPO_OBO_FILE):
            logger.error("Failed to download hp.obo")
            return False

        logger.info("Downloading HPOA annotation file …")
        if not self.download_file(HPOA_URL, HPOA_FILE):
            logger.error("Failed to download phenotype.hpoa")
            return False

        logger.info("Downloading MedGen ID mappings …")
        if not self.download_file(MEDGEN_URL, MEDGEN_FILE):
            logger.warning("Failed to download MedGenIDMappings — CUI column will be empty")

        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse HPO OBO and HPOA files.

        Returns
        -------
        dict with keys:
          phenotype_nodes            — HP term nodes
          disease_phenotype_positive — phenotype-present edges
          disease_phenotype_negative — phenotype-absent edges
        """
        result: Dict[str, pd.DataFrame] = {}

        obo_path = self.source_dir / HPO_OBO_FILE
        if obo_path.exists():
            result.update(self._parse_obo(obo_path))
        else:
            logger.error("hp.obo not found: %s", obo_path)

        hpoa_path = self.source_dir / HPOA_FILE
        if hpoa_path.exists():
            medgen_path = self.source_dir / MEDGEN_FILE
            cui_map = self._load_cui_map(medgen_path) if medgen_path.exists() else {}
            result.update(self._parse_hpoa(hpoa_path, cui_map))
        else:
            logger.error("phenotype.hpoa not found: %s", hpoa_path)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_obo(self, obo_path) -> Dict[str, pd.DataFrame]:
        """Parse hp.obo and return phenotype node DataFrame."""
        if not HAS_OBONET:
            logger.error("obonet is not installed — cannot parse hp.obo")
            return {}

        logger.info("Parsing hp.obo from %s …", obo_path)
        try:
            graph = obonet.read_obo(str(obo_path))
        except Exception as exc:
            logger.error("Failed to read hp.obo: %s", exc)
            return {}

        rows = []
        for node_id, data in graph.nodes(data=True):
            node_id = str(node_id)
            if not node_id.startswith("HP:"):
                continue
            if data.get("is_obsolete", False):
                continue

            is_a_ids = []
            for entry in data.get("is_a", []):
                pid = self._extract_id(str(entry))
                if pid.startswith("HP:"):
                    is_a_ids.append(pid)

            rows.append({
                "hp_id":      node_id,
                "name":       data.get("name", ""),
                "definition": self._clean_definition(data.get("def", "")),
                "synonyms":   self._parse_synonyms(data.get("synonym", [])),
                "is_a":       "|".join(is_a_ids),
            })

        logger.info("Extracted %d HPO phenotype terms", len(rows))

        nodes_df = pd.DataFrame(rows)
        nodes_df["source_database"] = "HPO"
        return {PHENOTYPE_NODES: nodes_df}

    def _load_cui_map(self, medgen_path) -> Dict[str, str]:
        """
        Build {database_id → CUI} from MedGenIDMappings.txt.gz.

        Handles OMIM (source="OMIM", source_id="200100" → key "OMIM:200100")
        and Orphanet (source="Orphanet", source_id="Orphanet_14" → key "ORPHA:14").
        """
        mapping: Dict[str, str] = {}
        try:
            with gzip.open(medgen_path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("|")
                    if len(parts) < 4:
                        continue
                    cui, source_id, source = parts[0], parts[2], parts[3]
                    if source == "OMIM":
                        mapping[f"OMIM:{source_id}"] = cui
                    elif source == "Orphanet":
                        numeric = source_id.replace("Orphanet_", "")
                        mapping[f"ORPHA:{numeric}"] = cui
        except Exception as exc:
            logger.error("Failed to read MedGenIDMappings: %s", exc)
        logger.info("Loaded %d disease → CUI mappings", len(mapping))
        return mapping

    def _parse_hpoa(self, hpoa_path, cui_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
        """Parse phenotype.hpoa and return positive/negative edge DataFrames."""
        logger.info("Parsing HPOA from %s …", hpoa_path)

        rows = []
        try:
            with open(hpoa_path, encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 4:
                        continue
                    rows.append(parts[:12] if len(parts) >= 12 else parts + [""] * (12 - len(parts)))
        except Exception as exc:
            logger.error("Failed to read phenotype.hpoa: %s", exc)
            return {}

        df = pd.DataFrame(rows, columns=_HPOA_COLS)
        logger.info("Loaded %d raw HPOA records", len(df))

        # Must have both disease and HPO ID
        df = df.dropna(subset=["database_id", "hpo_id"])
        df = df[df["database_id"].str.strip() != ""]
        df = df[df["hpo_id"].str.strip().str.startswith("HP:")]

        df["database_id"] = df["database_id"].str.strip()
        df["hpo_id"]      = df["hpo_id"].str.strip()
        df["qualifier"]   = df["qualifier"].str.strip().fillna("")
        df["evidence"]    = df["evidence"].str.strip().fillna("")
        df["onset"]       = df["onset"].str.strip().fillna("")
        df["frequency"]   = df["frequency"].str.strip().fillna("")
        df["sex"]         = df["sex"].str.strip().fillna("")

        df["cui"] = df["database_id"].map(cui_map).fillna("")
        if cui_map:
            n_mapped = (df["cui"] != "").sum()
            logger.info("CUI mapped: %d / %d records", n_mapped, len(df))

        # Split into positive and negative (PrimeKG filter logic)
        neg_mask = df["qualifier"] == "NOT"

        edge_cols = ["hp_id", "cui", "evidence", "onset", "frequency", "sex"]

        pos_edges = (
            df[~neg_mask].rename(columns={"hpo_id": "hp_id"})[edge_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        pos_edges["source_database"] = "HPO"

        neg_edges = (
            df[neg_mask].rename(columns={"hpo_id": "hp_id"})[edge_cols]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        neg_edges["source_database"] = "HPO"

        logger.info(
            "HPOA edges — positive: %d, negative: %d",
            len(pos_edges), len(neg_edges),
        )

        return {POSITIVE_EDGES: pos_edges, NEGATIVE_EDGES: neg_edges}

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_id(text: str) -> str:
        """Extract the first whitespace-delimited token (the CURIE ID)."""
        m = re.match(r"^(\S+)", text.strip())
        return m.group(1) if m else ""

    @staticmethod
    def _clean_definition(raw: str) -> str:
        """Strip OBO-format quotes and citation brackets from a def: field."""
        if not raw:
            return ""
        if raw.startswith('"'):
            raw = raw[1:]
        if " [" in raw:
            raw = raw.split(" [")[0]
        if raw.endswith('"'):
            raw = raw[:-1]
        return raw.replace("\t", " ").strip()

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
        """Return column schemas for all three output DataFrames."""
        return {
            PHENOTYPE_NODES: {
                "hp_id":           "HPO term ID (e.g. HP:0001250)",
                "name":            "Human-readable phenotype name",
                "definition":      "Text definition (from OBO def: field)",
                "synonyms":        "Pipe-delimited synonym strings",
                "is_a":            "Pipe-delimited parent HP IDs via is_a relationships",
                "source_database": "Source name string (HPO)",
            },
            POSITIVE_EDGES: {
                "hp_id":           "HPO term ID (e.g. HP:0001250)",
                "cui":             "UMLS CUI mapped from OMIM/ORPHA via MedGen (e.g. C0002395)",
                "evidence":        "Evidence code (IEA, PCS, TAS, etc.)",
                "onset":           "Age of onset HPO term (e.g. HP:0003577)",
                "frequency":       "Phenotype frequency annotation (e.g. HP:0040281, 3/10)",
                "sex":             "Sex modifier (MALE, FEMALE, or empty)",
                "source_database": "Source name string (HPO)",
            },
            NEGATIVE_EDGES: {
                "hp_id":           "HPO term ID (e.g. HP:0001250)",
                "cui":             "UMLS CUI mapped from OMIM/ORPHA via MedGen (e.g. C0002395)",
                "evidence":        "Evidence code (IEA, PCS, TAS, etc.)",
                "onset":           "Age of onset HPO term (e.g. HP:0003577)",
                "frequency":       "Phenotype frequency annotation (e.g. HP:0040281, 3/10)",
                "sex":             "Sex modifier (MALE, FEMALE, or empty)",
                "source_database": "Source name string (HPO)",
            },
        }
