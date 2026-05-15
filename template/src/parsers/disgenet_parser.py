"""
DisGeNETParser: Parser for DisGeNET gene-disease association data.

DisGeNET is a comprehensive database of gene-disease associations
from various sources including literature and databases.

Source: https://www.disgenet.org/
API Documentation: https://www.disgenet.org/api/

Disease scope is configurable via the disease_scope parameter, which
is read from config/project.yaml by the pipeline.
"""

import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .base_parser import BaseParser
from config_loader import get_disease_scope

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output DataFrame names (stems for TSV files under data/processed/disgenet/)
# These MUST match the source_filename values in config/ontology_mappings.yaml
# where applicable.
# ---------------------------------------------------------------------------
GENES_OUTPUT = "genes"                          # → genes.tsv
DISEASES_OUTPUT = "disease_classifications"     # → disease_classifications.tsv  (ontology_mappings.yaml)
DISEASE_MAPPINGS_OUTPUT = "disease_mappings"    # → disease_mappings.tsv         (ontology_mappings.yaml)
GDA_OUTPUT = "gene_disease_associations"        # → gene_disease_associations.tsv

# Raw cache file names written to data/raw/disgenet/
RAW_GDA_FILE = "api_gene_disease_associations.tsv"
RAW_DISEASE_CLASSIFICATIONS_FILE = "api_disease_classifications.tsv"
RAW_DISEASE_MAPPINGS_FILE = "api_disease_mappings.tsv"

# Vocabulary prefixes used in the DisGeNET diseaseMapping field
VOCAB_MAP = {
    "MESH": "MSH", "MSH": "MSH",
    "ICD10": "ICD10", "NCI": "NCI",
    "OMIM": "OMIM", "ICD9CM": "ICD9CM",
    "HPO": "HPO", "DO": "DO",
    "MONDO": "MONDO", "UMLS": "UMLS",
    "EFO": "EFO", "ORDO": "ORDO",
}
VOCAB_COLS = ["MSH", "ICD10", "NCI", "OMIM", "ICD9CM", "HPO", "DO",
              "MONDO", "UMLS", "EFO", "ORDO"]


class DisGeNETParser(BaseParser):
    """
    Parser for DisGeNET gene-disease association data via REST API.

    Queries the DisGeNET REST API to retrieve gene-disease associations (GDAs)
    for diseases configured in config/project.yaml.  Disease scope is never
    hard-coded; it is read from the disease_scope parameter or from
    config/project.yaml via get_disease_scope().

    Outputs
    -------
    genes
        Gene nodes: geneId, geneSymbol, geneName, ensemblId, proteinId,
        pLI, DSI, DPI.
    disease_classifications
        Disease nodes: diseaseId, diseaseName, diseaseType, diseaseClass,
        diseaseSemanticType.
    disease_mappings
        Disease cross-reference codes: diseaseId, MSH, ICD10, DO, MONDO, …
    gene_disease_associations
        GDA edges: geneId, diseaseId, gdaScore, evidenceIndex,
        numberOfPublications, numberOfSnps.
    """

    def __init__(
        self,
        data_dir: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        disease_scope: Optional[Dict] = None,
    ):
        """
        Parameters
        ----------
        data_dir:
            Directory for storing raw data files.
        api_key:
            DisGeNET API key.  Falls back to the DISGENET_API_KEY env var.
        base_url:
            Base URL for the DisGeNET REST API.
            Defaults to https://www.disgenet.org/api.
        disease_scope:
            Disease scope dict from project config (auto-injected by pipeline).
            Falls back to config/project.yaml via get_disease_scope().
        """
        super().__init__(data_dir)
        self.api_key = api_key or os.getenv("DISGENET_API_KEY")
        self.API_BASE_URL = (base_url or "https://www.disgenet.org/api").rstrip("/")

        self.session = requests.Session()

        # Disease scope — prefer caller-supplied dict, then project.yaml
        _cfg_scope = disease_scope if disease_scope else get_disease_scope()
        self.disease_terms: List[str] = _cfg_scope.get("primary_terms", [])
        self.umls_cuis: List[str] = _cfg_scope.get("umls_cuis", [])

        if self.api_key:
            self.session.headers.update({
                "Authorization": self.api_key,
                "accept": "application/json",
            })
            logger.info("DisGeNET API key configured; base URL: %s", self.API_BASE_URL)
        else:
            logger.warning(
                "No DisGeNET API key provided — set DISGENET_API_KEY or pass api_key."
            )

        if not self.disease_terms and not self.umls_cuis:
            logger.warning(
                "No disease terms or UMLS CUIs in disease_scope; "
                "API queries will return nothing."
            )

    # ------------------------------------------------------------------
    # download_data
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """
        Download GDA data from the DisGeNET REST API and derive disease files.

        Strategy
        --------
        1. Start with explicit umls_cuis from project.yaml.
        2. Search by primary_terms to discover additional disease CUIs.
        3. Fetch GDAs for every unique CUI (with pagination).
        4. Derive disease classification and mapping files from GDA data.

        Returns True when data is ready (freshly downloaded or cached).
        """
        if not self.api_key:
            logger.error("No API key — cannot download DisGeNET data.")
            return False

        raw_gda = self.get_file_path(RAW_GDA_FILE)
        raw_cls = self.get_file_path(RAW_DISEASE_CLASSIFICATIONS_FILE)
        raw_map = self.get_file_path(RAW_DISEASE_MAPPINGS_FILE)

        if (Path(raw_gda).exists() and Path(raw_cls).exists()
                and Path(raw_map).exists() and not self.force):
            logger.info("DisGeNET raw files already present; skipping download.")
            return True

        # ---- Step 1: Collect disease CUIs --------------------------------
        all_cuis: List[str] = list(self.umls_cuis)

        if self.disease_terms:
            logger.info(
                "Searching for additional disease CUIs by term(s): %s",
                self.disease_terms,
            )
            term_cuis = self._search_disease_cuis(self.disease_terms)
            for cui in term_cuis:
                if cui not in all_cuis:
                    all_cuis.append(cui)

        if not all_cuis:
            logger.error(
                "No disease CUIs available. "
                "Check disease_scope.umls_cuis / primary_terms in project.yaml."
            )
            return False

        logger.info(
            "Will query GDAs for %d disease CUI(s): %s", len(all_cuis), all_cuis
        )

        # ---- Step 2: Fetch GDAs for each CUI -----------------------------
        all_gda_records: List[Dict] = []
        for cui in all_cuis:
            records = self._fetch_gdas_for_disease(cui)
            logger.info("  %s → %d GDA record(s)", cui, len(records))
            all_gda_records.extend(records)
            time.sleep(0.5)

        if not all_gda_records:
            logger.error("No GDA records retrieved from DisGeNET API.")
            return False

        gda_df = pd.DataFrame(all_gda_records).drop_duplicates()
        gda_df.to_csv(raw_gda, sep="\t", index=False)
        logger.info("✓ Saved %d GDA records → %s", len(gda_df), raw_gda)

        # ---- Step 3: Derive disease classification file from GDA data ----
        cls_cols = [c for c in [
            "diseaseId", "diseaseName", "diseaseType",
            "diseaseClasses_MSH", "diseaseClasses_UMLS_ST",
            "diseaseClasses_DO", "diseaseClasses_HPO",
        ] if c in gda_df.columns]

        if cls_cols and "diseaseId" in cls_cols:
            cls_df = gda_df[cls_cols].drop_duplicates(subset=["diseaseId"]).copy()
            cls_df["sourceDatabase"] = "DisGeNET"
            cls_df.to_csv(raw_cls, sep="\t", index=False)
            logger.info("✓ Saved %d disease classification records → %s",
                        len(cls_df), raw_cls)
        else:
            pd.DataFrame(columns=["diseaseId", "diseaseName"]).to_csv(
                raw_cls, sep="\t", index=False
            )

        # ---- Step 4: Derive disease mappings from diseaseMapping field ---
        if "diseaseId" in gda_df.columns and "diseaseMapping" in gda_df.columns:
            map_df = self._parse_disease_mappings(gda_df)
            map_df.to_csv(raw_map, sep="\t", index=False)
            logger.info("✓ Saved %d disease mapping records → %s",
                        len(map_df), raw_map)
        else:
            pd.DataFrame(columns=["diseaseId"] + VOCAB_COLS).to_csv(
                raw_map, sep="\t", index=False
            )

        return True

    def _parse_disease_mappings(self, gda_df: pd.DataFrame) -> pd.DataFrame:
        """Parse the diseaseMapping comma-separated field into per-vocab columns."""
        # Get one row per disease (first occurrence)
        base = gda_df[["diseaseId", "diseaseName"]].drop_duplicates(
            subset=["diseaseId"]
        ).copy() if "diseaseName" in gda_df.columns else (
            gda_df[["diseaseId"]].drop_duplicates().copy()
        )

        # Parse mapping codes per disease
        first_mapping = (
            gda_df.groupby("diseaseId")["diseaseMapping"].first()
        )

        def _parse(mapping_str):
            codes: Dict[str, Optional[str]] = {v: None for v in VOCAB_COLS}
            if pd.isna(mapping_str):
                return codes
            for token in str(mapping_str).split(","):
                token = token.strip()
                if "_" in token:
                    vocab, code = token.split("_", 1)
                    std = VOCAB_MAP.get(vocab.upper())
                    if std and codes.get(std) is None:
                        codes[std] = code
            return codes

        parsed = first_mapping.apply(_parse).apply(pd.Series)
        result = base.set_index("diseaseId").join(parsed, how="left").reset_index()
        result["sourceDatabase"] = "DisGeNET"
        return result

    # ------------------------------------------------------------------
    # API helpers
    # ------------------------------------------------------------------

    def _search_disease_cuis(self, terms: List[str]) -> List[str]:
        """Search for disease CUIs using GET /disease/search."""
        cuis: List[str] = []
        for term in terms:
            endpoint = f"{self.API_BASE_URL}/disease/search"
            params = {"q": term}
            try:
                resp = self.session.get(endpoint, params=params, timeout=30)
                resp.raise_for_status()
                data = resp.json()
                if isinstance(data, list):
                    for item in data:
                        cui = item.get("diseaseId") or item.get("disease_id", "")
                        if cui and cui not in cuis:
                            cuis.append(cui)
                    logger.info("  '%s' → %d disease(s) found", term, len(data))
                time.sleep(0.3)
            except requests.RequestException as exc:
                logger.warning("Disease search failed for '%s': %s", term, exc)
        return cuis

    def _fetch_gdas_for_disease(self, disease_cui: str) -> List[Dict]:
        """
        Fetch all GDA records for a disease CUI via GET /gda/disease/{diseaseid}.
        Handles pagination (page_number parameter, 100 records/page default).
        """
        endpoint = f"{self.API_BASE_URL}/gda/disease/{disease_cui}"
        all_records: List[Dict] = []
        page = 0

        while True:
            params = {"page_number": page}
            try:
                resp = self.session.get(endpoint, params=params, timeout=60)

                if resp.status_code == 404:
                    logger.debug("No GDAs for %s (404)", disease_cui)
                    break
                if resp.status_code == 429:
                    logger.warning("Rate limited by DisGeNET; sleeping 10 s…")
                    time.sleep(10)
                    continue

                resp.raise_for_status()
                data = resp.json()

                if not isinstance(data, list) or not data:
                    break

                all_records.extend(data)
                logger.debug(
                    "  %s page %d: %d record(s)", disease_cui, page, len(data)
                )

                if len(data) < 100:
                    break

                page += 1
                time.sleep(0.3)

            except requests.RequestException as exc:
                logger.error("Failed to fetch GDAs for %s: %s", disease_cui, exc)
                break

        return all_records

    # ------------------------------------------------------------------
    # parse_data
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse DisGeNET raw files into four clean DataFrames.

        Returns
        -------
        dict with keys:
          genes                       Gene nodes
          disease_classifications     Disease nodes
          disease_mappings            Disease cross-references
          gene_disease_associations   GDA edges
        """
        logger.info("Parsing DisGeNET data…")

        # Load GDA data
        raw_gda = self.get_file_path(RAW_GDA_FILE)
        if not Path(raw_gda).exists():
            logger.warning("No raw GDA file found at %s", raw_gda)
            return {}

        gda_df = self.read_tsv(raw_gda)
        if gda_df is None or gda_df.empty:
            logger.warning("Raw GDA file is empty or unreadable.")
            return {}

        logger.info(
            "Loaded %d GDA record(s); columns: %s", len(gda_df), list(gda_df.columns)
        )

        # Load disease classification file
        raw_cls = self.get_file_path(RAW_DISEASE_CLASSIFICATIONS_FILE)
        cls_df: Optional[pd.DataFrame] = None
        if Path(raw_cls).exists():
            cls_df = self.read_tsv(raw_cls)
            if cls_df is not None and cls_df.empty:
                cls_df = None

        # Load disease mappings file
        raw_map = self.get_file_path(RAW_DISEASE_MAPPINGS_FILE)
        map_df: Optional[pd.DataFrame] = None
        if Path(raw_map).exists():
            map_df = self.read_tsv(raw_map)
            if map_df is not None and map_df.empty:
                map_df = None

        result: Dict[str, pd.DataFrame] = {
            GENES_OUTPUT: self._build_gene_nodes(gda_df),
            DISEASES_OUTPUT: self._build_disease_nodes(gda_df, cls_df),
            DISEASE_MAPPINGS_OUTPUT: self._build_disease_mappings(gda_df, map_df),
            GDA_OUTPUT: self._build_gda_edges(gda_df),
        }

        for df in result.values():
            df["source_database"] = "DisGeNET"

        for key, df in result.items():
            logger.info("  %s: %d rows × %d cols", key, len(df), len(df.columns))

        return result

    # ------------------------------------------------------------------
    # DataFrame builders
    # ------------------------------------------------------------------

    def _build_gene_nodes(self, gda_df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract unique Gene nodes from GDA data.

        Output columns: geneId, geneSymbol, geneName, ensemblId, proteinId,
                        pLI, DSI, DPI
        """
        # Normalise API field names
        rename_map = {
            "geneid": "geneId",
            "gene_symbol": "geneSymbol",
            "gene_dsi": "DSI",
            "gene_dpi": "DPI",
            "gene_pli": "pLI",
        }
        df = gda_df.rename(
            columns={k: v for k, v in rename_map.items() if k in gda_df.columns}
        )

        present = [c for c in ["geneId", "geneSymbol", "DSI", "DPI", "pLI"]
                   if c in df.columns]
        if not present:
            logger.warning("No gene-level columns found in GDA data.")
            return pd.DataFrame(
                columns=["geneId", "geneSymbol", "geneName", "ensemblId",
                         "proteinId", "pLI", "DSI", "DPI"]
            )

        genes = df[present].drop_duplicates(
            subset=["geneId"] if "geneId" in present else present
        ).copy()

        # Add columns required by schema but not returned by this API endpoint
        for col in ["geneName", "ensemblId", "proteinId", "DSI", "DPI", "pLI"]:
            if col not in genes.columns:
                genes[col] = None

        # Return in canonical column order
        col_order = ["geneId", "geneSymbol", "geneName", "ensemblId",
                     "proteinId", "pLI", "DSI", "DPI"]
        genes = genes[[c for c in col_order if c in genes.columns]]
        return genes.reset_index(drop=True)

    def _build_disease_nodes(
        self,
        gda_df: pd.DataFrame,
        cls_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build Disease nodes DataFrame.

        Output columns: diseaseId, diseaseName, diseaseType, diseaseClass,
                        diseaseSemanticType
        """
        want = ["diseaseId", "diseaseName", "diseaseType",
                "diseaseClass", "diseaseSemanticType"]

        if cls_df is not None and not cls_df.empty:
            # Use the dedicated disease classifications file
            rename = {
                "diseaseClasses_MSH": "diseaseClass",
                "diseaseClasses_UMLS_ST": "diseaseSemanticType",
                "sourceDatabase": "source_database",
            }
            d = cls_df.rename(columns={k: v for k, v in rename.items()
                                        if k in cls_df.columns})
            keep = [c for c in want if c in d.columns]
            diseases = d[keep].drop_duplicates(subset=["diseaseId"]).copy()

            # Supplement diseaseType from GDA data if not already present
            if "diseaseType" not in diseases.columns or diseases["diseaseType"].isna().all():
                if "diseaseId" in gda_df.columns and "diseaseType" in gda_df.columns:
                    dtype_map = (
                        gda_df[["diseaseId", "diseaseType"]]
                        .drop_duplicates(subset=["diseaseId"])
                        .set_index("diseaseId")["diseaseType"]
                    )
                    diseases = diseases.set_index("diseaseId")
                    diseases["diseaseType"] = dtype_map
                    diseases = diseases.reset_index()
        else:
            # Fall back to GDA data
            rename = {
                "disease_name": "diseaseName",
                "disease_type": "diseaseType",
                "diseaseClasses_MSH": "diseaseClass",
                "diseaseClasses_UMLS_ST": "diseaseSemanticType",
            }
            df = gda_df.rename(
                columns={k: v for k, v in rename.items() if k in gda_df.columns}
            )
            keep = [c for c in want if c in df.columns]
            diseases = df[keep].drop_duplicates(
                subset=["diseaseId"] if "diseaseId" in keep else keep
            ).copy() if keep else pd.DataFrame()

        # Ensure all required columns exist
        for col in want:
            if col not in diseases.columns:
                diseases[col] = None

        diseases = diseases[want].copy()
        return diseases.reset_index(drop=True)

    def _build_disease_mappings(
        self,
        gda_df: pd.DataFrame,
        map_df: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Build disease cross-reference mappings DataFrame.

        Output columns: diseaseId, diseaseName, MSH, ICD10, NCI, OMIM,
                        ICD9CM, HPO, DO, MONDO, UMLS, EFO, ORDO
        """
        want = ["diseaseId", "diseaseName"] + VOCAB_COLS

        if map_df is not None and not map_df.empty:
            # Use the dedicated disease mappings file
            keep = [c for c in want if c in map_df.columns]
            mappings = map_df[keep].drop_duplicates(
                subset=["diseaseId"] if "diseaseId" in keep else keep
            ).copy()
        else:
            # Derive from GDA diseaseMapping field
            if "diseaseId" in gda_df.columns and "diseaseMapping" in gda_df.columns:
                mappings = self._parse_disease_mappings(gda_df)
                keep = [c for c in want if c in mappings.columns]
                mappings = mappings[keep].copy()
            else:
                mappings = pd.DataFrame(columns=want)

        # Ensure all required columns exist
        for col in want:
            if col not in mappings.columns:
                mappings[col] = None

        # DO column is required by ontology_mappings.yaml filter
        if "DO" not in mappings.columns:
            mappings["DO"] = None

        mappings = mappings[[c for c in want if c in mappings.columns]].copy()
        return mappings.reset_index(drop=True)

    def _build_gda_edges(self, gda_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Gene-Disease Association edge DataFrame.

        Output columns: geneId, diseaseId, gdaScore, evidenceIndex,
                        numberOfPublications, numberOfSnps
        """
        # Normalise field names from various API versions
        rename_map = {
            # Older API snake_case
            "geneid": "geneId",
            "gene_symbol": "geneSymbol",
            "disease_name": "diseaseName",
            "Npubmeds": "numberOfPublications",
            "NofSnps": "numberOfSnps",
            # Current API
            "score": "gdaScore",
            "EI": "evidenceIndex",
            "nPubs": "numberOfPublications",
            "nSnps": "numberOfSnps",
        }
        df = gda_df.rename(
            columns={k: v for k, v in rename_map.items() if k in gda_df.columns}
        ).copy()

        # Required output columns
        required = ["geneId", "diseaseId", "gdaScore",
                    "evidenceIndex", "numberOfPublications", "numberOfSnps"]
        for col in required:
            if col not in df.columns:
                df[col] = None

        edges = df[required].copy()
        return edges.reset_index(drop=True)

    # ------------------------------------------------------------------
    # get_schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return the column schema for all DisGeNET output DataFrames."""
        return {
            GENES_OUTPUT: {
                "geneId": "NCBI Gene ID",
                "geneSymbol": "HGNC gene symbol",
                "geneName": "Full gene name",
                "ensemblId": "Ensembl gene identifier",
                "proteinId": "UniProt protein identifier",
                "pLI": "Probability of loss-of-function intolerance (gnomAD)",
                "DSI": "Disease Specificity Index (0–1; higher = more disease-specific)",
                "DPI": "Disease Pleiotropy Index (0–1; higher = more disease classes)",
            },
            DISEASES_OUTPUT: {
                "diseaseId": "Disease identifier (UMLS CUI)",
                "diseaseName": "Disease name",
                "diseaseType": "Disease type (disease, group, phenotype)",
                "diseaseClass": "Disease class (MeSH hierarchy code)",
                "diseaseSemanticType": "UMLS semantic type",
            },
            DISEASE_MAPPINGS_OUTPUT: {
                "diseaseId": "Disease identifier (UMLS CUI)",
                "diseaseName": "Disease name",
                "MSH": "MeSH code",
                "ICD10": "ICD-10 code",
                "NCI": "NCI Thesaurus code",
                "OMIM": "OMIM identifier",
                "ICD9CM": "ICD-9-CM code",
                "HPO": "Human Phenotype Ontology code",
                "DO": "Disease Ontology identifier",
                "MONDO": "MONDO identifier",
                "UMLS": "UMLS CUI cross-reference",
                "EFO": "Experimental Factor Ontology code",
                "ORDO": "Orphanet code",
            },
            GDA_OUTPUT: {
                "geneId": "NCBI Gene ID",
                "diseaseId": "Disease identifier (UMLS CUI)",
                "gdaScore": "GDA score (0–1)",
                "evidenceIndex": "Evidence Index (0–1; proportion of supporting evidence)",
                "numberOfPublications": "Number of supporting PubMed publications",
                "numberOfSnps": "Number of associated SNPs",
            },
        }
