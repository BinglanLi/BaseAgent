"""
Reactome Pathway Parser for the knowledge graph.

This module parses Reactome pathway data to extract Pathway nodes and
gene-pathway relationships for the knowledge graph.

Data Sources:
  - https://reactome.org/download/current/ReactomePathways.txt
      Columns (no header): stable_id, pathway_name, species
  - https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt
      Columns (no header): ncbi_gene_id, reactome_id, url, event_name,
                           evidence_code, species

Filtered to Homo sapiens only.

Output:
  - pathways.tsv                    : Pathway nodes
  - ncbi_gene_pathway_relationships.tsv : Gene-pathway edges (NCBI Gene IDs)
"""

import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHWAYS_URL = "https://reactome.org/download/current/ReactomePathways.txt"
NCBI_GENE_PATHWAY_URL = "https://reactome.org/download/current/NCBI2Reactome_All_Levels.txt"

HOMO_SAPIENS = "Homo sapiens"
SOURCE_DB = "Reactome"


class ReactomeParser(BaseParser):
    """
    Parser for the Reactome curated pathway database.

    Downloads pathway definitions and NCBI Gene → Pathway mappings directly
    from Reactome's public download area, then filters to Homo sapiens.

    No credentials are required (public data source).
    """

    def __init__(self, data_dir: str):
        """
        Initialise the Reactome parser.

        Args:
            data_dir: Root directory for raw downloaded files.
        """
        super().__init__(data_dir)
        self.source_name = "reactome"
        # Re-derive source_dir after setting source_name
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """
        Download ReactomePathways.txt and NCBI2Reactome_All_Levels.txt.

        Returns:
            True when both files are available (downloaded or cached).
        """
        logger.info("Downloading Reactome pathway data …")

        pathways_ok = self.download_file(PATHWAYS_URL, "ReactomePathways.txt")
        ncbi_ok = self.download_file(NCBI_GENE_PATHWAY_URL, "NCBI2Reactome_All_Levels.txt")

        success = bool(pathways_ok) and bool(ncbi_ok)
        if success:
            logger.info("✓ Reactome files downloaded / cached successfully.")
        else:
            logger.error("✗ One or more Reactome downloads failed.")
        return success

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse downloaded Reactome files.

        Returns:
            Dictionary with keys:
              - 'pathways'                        → Pathway node DataFrame
              - 'ncbi_gene_pathway_relationships' → Gene-pathway edge DataFrame
        """
        pathways_df = self._parse_pathways()
        relationships_df = self._parse_ncbi_gene_pathway()

        if pathways_df is None or relationships_df is None:
            return {}

        return {
            "pathways": pathways_df,
            "ncbi_gene_pathway_relationships": relationships_df,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_pathways(self) -> pd.DataFrame:
        """
        Parse ReactomePathways.txt → pathways DataFrame.

        File format (tab-separated, no header):
            stable_id   pathway_name   species

        Returns:
            DataFrame with columns: reactome_id, pathway_name, species,
                                    source_database
        """
        filepath = self.source_dir / "ReactomePathways.txt"
        if not filepath.exists():
            logger.error(f"ReactomePathways.txt not found: {filepath}")
            return None

        logger.info(f"Parsing ReactomePathways.txt from {filepath}")

        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=["reactome_id", "pathway_name", "species"],
            dtype=str,
        )

        logger.info(f"  Total pathways (all species): {len(df)}")

        # Filter to Homo sapiens
        df = df[df["species"] == HOMO_SAPIENS].copy()
        logger.info(f"  Homo sapiens pathways: {len(df)}")

        df["source_database"] = SOURCE_DB
        df = df.reset_index(drop=True)

        logger.info(f"✓ Parsed {len(df)} Homo sapiens pathways.")
        return df

    def _parse_ncbi_gene_pathway(self) -> pd.DataFrame:
        """
        Parse NCBI2Reactome_All_Levels.txt → gene-pathway relationships.

        File format (tab-separated, no header):
            ncbi_gene_id   reactome_id   url   event_name   evidence_code   species

        Returns:
            DataFrame with columns: ncbi_gene_id, reactome_id, evidence_code,
                                    source_database
        """
        filepath = self.source_dir / "NCBI2Reactome_All_Levels.txt"
        if not filepath.exists():
            logger.error(f"NCBI2Reactome_All_Levels.txt not found: {filepath}")
            return None

        logger.info(f"Parsing NCBI2Reactome_All_Levels.txt from {filepath}")

        df = pd.read_csv(
            filepath,
            sep="\t",
            header=None,
            names=[
                "ncbi_gene_id",
                "reactome_id",
                "url",
                "event_name",
                "evidence_code",
                "species",
            ],
            dtype=str,
        )

        logger.info(f"  Total gene-pathway mappings (all species): {len(df)}")

        # Filter to Homo sapiens
        df = df[df["species"] == HOMO_SAPIENS].copy()
        logger.info(f"  Homo sapiens gene-pathway mappings: {len(df)}")

        # Keep only required columns
        df = df[["ncbi_gene_id", "reactome_id", "evidence_code"]].copy()

        # Drop duplicates
        before = len(df)
        df = df.drop_duplicates()
        logger.info(f"  Deduplicated: {before} → {len(df)} rows")

        df["source_database"] = SOURCE_DB
        df = df.reset_index(drop=True)

        logger.info(f"✓ Parsed {len(df)} Homo sapiens gene-pathway relationships.")
        return df

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """
        Return the column schema for each output DataFrame.

        Returns:
            Nested dict: {output_name: {column_name: description}}
        """
        return {
            "pathways": {
                "reactome_id":    "Reactome stable pathway identifier (e.g. R-HSA-XXXXXXX)",
                "pathway_name":   "Human-readable pathway name",
                "species":        "Species name (Homo sapiens)",
                "source_database": "Data source label",
            },
            "ncbi_gene_pathway_relationships": {
                "ncbi_gene_id":   "NCBI Entrez Gene identifier",
                "reactome_id":    "Reactome stable pathway identifier",
                "evidence_code":  "Evidence code for the gene-pathway association",
                "source_database": "Data source label",
            },
        }
