"""
ClinPGx / CPIC Parser for the knowledge graph.

Queries the CPIC REST API (https://api.cpicpgx.org/v1) to retrieve
pharmacogenomic guidelines and gene-drug-phenotype associations.

Produces:
  - drug_label_nodes.tsv       : DrugLabel nodes (CPIC guidelines)
  - gene_drug_interactions.tsv : AFFECTS_RESPONSE_TO edges (Gene → Drug)

API: https://api.cpicpgx.org/v1
"""

import logging
import time
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

_DEFAULT_BASE_URL = "https://api.cpicpgx.org/v1"

DRUG_LABEL_NODES  = "drug_label_nodes"
GENE_DRUG_INTERV  = "gene_drug_interactions"

_CALL_DELAY = 0.2


class ClinPGxParser(BaseParser):
    """
    Parser for CPIC pharmacogenomic guidelines.

    Queries the CPIC REST API for gene-drug pairs and guideline recommendations,
    producing DrugLabel nodes and AFFECTS_RESPONSE_TO relationship edges.

    Constructor args (injected from databases.yaml):
        data_dir – base directory for raw/cached files
        base_url – CPIC API base URL
    """

    def __init__(
        self,
        data_dir: str,
        base_url: Optional[str] = None,
        source_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        super().__init__(data_dir)
        self.source_name = "clinpgx"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)
        # Accept source_url or base_url (databases.yaml uses source_url)
        self.base_url = (source_url or base_url or _DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Data fetched via API in parse_data(); no pre-download needed."""
        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch CPIC gene-drug pairs, guidelines, and recommendations.
        """
        # 1. Fetch all gene-drug pairs
        gene_drug_pairs = self._fetch_gene_drug_pairs()
        if not gene_drug_pairs:
            logger.warning("No CPIC gene-drug pairs found.")
            return {}

        # 2. Fetch guidelines
        guidelines = self._fetch_guidelines()

        # 3. Fetch recommendations (gene-drug-phenotype mappings)
        recommendations = self._fetch_recommendations()

        # ---- DrugLabel nodes ----
        label_rows = []
        seen_labels = set()
        for g in guidelines:
            name = g.get("name", "")
            if name and name not in seen_labels:
                seen_labels.add(name)
                label_rows.append({
                    "guideline_id":    g.get("id", ""),
                    "guideline_name":  name,
                    "url":             g.get("url", ""),
                    "version":         str(g.get("version", "")),
                    "source_database": "CPIC",
                })

        drug_label_df = pd.DataFrame(label_rows).drop_duplicates(
            subset=["guideline_id"]
        ).reset_index(drop=True)
        logger.info("CPIC: %d drug label nodes", len(drug_label_df))

        # ---- AFFECTS_RESPONSE_TO edges ----
        interv_rows = []
        for rec in recommendations:
            gene_symbol = rec.get("gene", "")
            drug_name   = rec.get("drug", "")
            phenotype   = rec.get("phenotype", "")
            implication = rec.get("implication", "")
            recommendation_text = rec.get("recommendation", "")
            classification = rec.get("classification", "")
            if gene_symbol and drug_name:
                interv_rows.append({
                    "gene_symbol":       gene_symbol,
                    "drug_name":         drug_name,
                    "phenotype":         phenotype,
                    "implication":       implication,
                    "recommendation":    recommendation_text,
                    "classification":    classification,
                    "source_database":   "CPIC",
                })

        # Also add from gene-drug pairs if recommendations empty
        if not interv_rows:
            for pair in gene_drug_pairs:
                gene_symbol = pair.get("genesymbol", "")
                drug_name   = pair.get("drugname", "")
                if gene_symbol and drug_name:
                    interv_rows.append({
                        "gene_symbol":     gene_symbol,
                        "drug_name":       drug_name,
                        "phenotype":       "",
                        "implication":     "",
                        "recommendation":  "",
                        "classification":  pair.get("cpicstatus", ""),
                        "source_database": "CPIC",
                    })

        gene_drug_df = pd.DataFrame(interv_rows).drop_duplicates(
            subset=["gene_symbol", "drug_name"]
        ).reset_index(drop=True)
        logger.info("CPIC: %d gene-drug interaction edges", len(gene_drug_df))

        return {
            DRUG_LABEL_NODES: drug_label_df,
            GENE_DRUG_INTERV: gene_drug_df,
        }

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _get_json(self, endpoint: str, params: Optional[dict] = None) -> Optional[list]:
        """GET an API endpoint and return parsed JSON list."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("CPIC API error at %s: %s", url, exc)
            return None

    def _fetch_gene_drug_pairs(self) -> List[dict]:
        """Fetch all CPIC gene-drug pairs."""
        data = self._get_json("pair")
        if data is None:
            return []
        logger.info("CPIC: %d gene-drug pairs", len(data))
        return data if isinstance(data, list) else []

    def _fetch_guidelines(self) -> List[dict]:
        """Fetch all CPIC guidelines."""
        data = self._get_json("guideline")
        if data is None:
            return []
        logger.info("CPIC: %d guidelines", len(data))
        return data if isinstance(data, list) else []

    def _fetch_recommendations(self) -> List[dict]:
        """
        Fetch gene-drug-phenotype recommendations from CPIC.
        Queries the /recommendation endpoint.
        """
        data = self._get_json("recommendation")
        if not data:
            return []
        logger.info("CPIC: %d raw recommendations", len(data))

        # Build drug ID -> drug name map from pair data
        drug_id_map = {}
        pair_data = self._get_json("pair") or []
        for pair in pair_data:
            drug_id = pair.get("drugid", "")
            gene_sym = pair.get("genesymbol", "")
            drug_name = pair.get("drugname", "")
            if drug_id:
                drug_id_map[drug_id] = {"gene": gene_sym, "name": drug_name}

        rows = []
        for rec in data:
            # Extract gene symbol from implications dict keys
            implications = rec.get("implications", {})
            if isinstance(implications, dict):
                gene_symbol = "|".join(implications.keys())
            else:
                gene_symbol = ""

            # Get drug name from drug ID via pair lookup
            drug_id = rec.get("drugid", "")
            drug_info = drug_id_map.get(drug_id, {})
            gene_from_pair = drug_info.get("gene", "")
            if not gene_symbol and gene_from_pair:
                gene_symbol = gene_from_pair

            # Get drug name from drug ID via pair lookup
            drug_name = drug_info.get("name", "")
            if not drug_name and drug_id:
                drug_name = drug_id.replace("RxNorm:", "").replace("DrugBank:", "")

            # Extract allele status as phenotype
            allele_status = rec.get("allelestatus", {})
            phenotype = "|".join(
                f"{k}:{v}" for k, v in allele_status.items()
            ) if isinstance(allele_status, dict) else ""

            # Extract implication text
            impl_text = "|".join(implications.values()) if isinstance(implications, dict) else ""

            if gene_symbol and drug_name:
                rows.append({
                    "gene":           gene_symbol,
                    "drug":           drug_name,
                    "phenotype":      phenotype,
                    "implication":    impl_text,
                    "recommendation": rec.get("drugrecommendation", ""),
                    "classification": rec.get("classification", ""),
                })

        return rows

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        return {
            DRUG_LABEL_NODES: {
                "guideline_id":   "CPIC guideline ID",
                "guideline_name": "CPIC guideline name",
                "url":            "Guideline URL",
                "version":        "Guideline version",
                "source_database": "Source database (CPIC)",
            },
            GENE_DRUG_INTERV: {
                "gene_symbol":    "Gene symbol",
                "drug_name":      "Drug name",
                "phenotype":      "Pharmacogenomic phenotype",
                "implication":    "Clinical implication",
                "recommendation": "CPIC drug recommendation",
                "classification": "CPIC classification level",
                "source_database": "Source database (CPIC)",
            },
        }
