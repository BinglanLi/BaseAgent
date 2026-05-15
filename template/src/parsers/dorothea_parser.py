"""
DoRothEA Parser for the knowledge graph.

This module parses DoRothEA data to extract transcription factor nodes
and TF-gene regulatory relationships for the knowledge graph.

DoRothEA is a gene regulatory network containing signed TF-target interactions.
Data is retrieved from OmniPath API which provides DoRothEA interactions.

Data Source:
  - OmniPath API: https://omnipathdb.org/

Output:
  - transcription_factor_nodes.tsv: TranscriptionFactor nodes
  - tf_gene_interactions.tsv: transcriptionFactorInteractsWithGene relationships

References:
  - https://saezlab.github.io/dorothea/
  - https://bioconductor.org/packages/release/data/experiment/vignettes/dorothea/inst/doc/dorothea.R
  - DoRothEA resource: https://doi.org/10.1101/gr.240663.118
"""

import logging
import pandas as pd
import requests

from pathlib import Path
from typing import Dict, List, Optional
from .base_parser import BaseParser

DOROTHEA_TRANSCRIPTION_FACTORS = 'transcription_factors'
DOROTHEA_TF_GENE_INTERACTIONS = 'tf_gene_interactions'

logger = logging.getLogger(__name__)


class DoRothEAParser(BaseParser):
    """
    Parser for DoRothEA transcription factor regulatory network.

    Extracts TF-gene regulatory relationships for use in the knowledge graph's
    transcriptionFactorInteractsWithGene relationships.

    Uses OmniPath API to retrieve DoRothEA data, which provides:
    - Confidence levels A, B, C, D (A = highest confidence)
    - Mode of regulation (stimulation/inhibition)
    - Curation effort scores
    """

    # OmniPath API endpoint for DoRothEA interactions
    OMNIPATH_DOROTHEA_URL = (
        "https://omnipathdb.org/interactions"
        "?datasets=dorothea"
        "&fields=curation_effort,dorothea_level"
        "&genesymbols=1"
        "&license=academic"
    )

    # Default confidence levels to include (A, B, C, D available)
    DEFAULT_CONFIDENCE_LEVELS = ['A', 'B', 'C', 'D']

    def __init__(self, data_dir: str, confidence_levels: Optional[List[str]] = None):
        """
        Initialize the DoRothEA parser.

        Args:
            data_dir: Directory to store downloaded and processed data
            confidence_levels: List of confidence levels to include (A, B, C, D)
                             Default is ['A', 'B', 'C', 'D']
        """
        super().__init__(data_dir)
        self.confidence_levels = confidence_levels or self.DEFAULT_CONFIDENCE_LEVELS

    def download_data(self) -> bool:
        """
        Download DoRothEA data from OmniPath API.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Downloading DoRothEA TF-gene regulatory network from OmniPath...")

        dorothea_path = self.get_file_path(f"{DOROTHEA_TF_GENE_INTERACTIONS}.tsv")
        if Path(dorothea_path).exists() and not self.force:
            logger.info(f"File already exists: {dorothea_path}")
            return True

        try:
            response = requests.get(self.OMNIPATH_DOROTHEA_URL, timeout=120)
            response.raise_for_status()

            with open(dorothea_path, 'w') as f:
                f.write(response.text)

            # Count lines to verify download
            line_count = len(response.text.strip().split('\n'))
            logger.info(f"Successfully downloaded DoRothEA to {dorothea_path}")
            logger.info(f"Downloaded {line_count - 1} interactions (excluding header)")

            return True

        except Exception as e:
            logger.error(f"Failed to download DoRothEA from OmniPath: {e}")
            return False

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse DoRothEA TF-gene regulatory data.

        Returns:
            Dictionary with:
              - 'transcription_factor_nodes': TF nodes
              - 'tf_gene_interactions': TF-gene regulatory relationships
        """
        dorothea_path = self.get_file_path(f"{DOROTHEA_TF_GENE_INTERACTIONS}.tsv")

        if not Path(dorothea_path).exists():
            logger.error(f"DoRothEA file not found: {dorothea_path}")
            return {}

        logger.info(f"Parsing DoRothEA from {dorothea_path}")

        try:
            # Read DoRothEA data from OmniPath
            # Columns: source, target, source_genesymbol, target_genesymbol,
            #          is_directed, is_stimulation, is_inhibition,
            #          consensus_direction, consensus_stimulation, consensus_inhibition,
            #          curation_effort, dorothea_level
            df = pd.read_csv(dorothea_path, sep='\t')

            logger.info(f"Loaded {len(df)} DoRothEA TF-target interactions")

            # Filter by confidence levels
            # Handle cases where dorothea_level may contain multiple levels (e.g., "A;D")
            if 'dorothea_level' in df.columns:
                def matches_confidence(level_str):
                    if pd.isna(level_str):
                        return False
                    levels = str(level_str).split(';')
                    return any(lvl in self.confidence_levels for lvl in levels)

                df = df[df['dorothea_level'].apply(matches_confidence)]
                logger.info(f"After filtering by confidence levels {self.confidence_levels}: {len(df)} interactions")

            # Extract unique TFs as nodes
            tfs = df['source_genesymbol'].dropna().unique()
            tf_nodes = [{
                "tf_symbol": tf,
                "node_type": "TranscriptionFactor",
                "source_database": "DoRothEA"
            } for tf in tfs]
            logger.info(f"Found {len(tf_nodes)} unique transcription factors")

            # Format TF-gene interactions
            interactions = []
            for _, row in df.iterrows():
                tf = row.get('source_genesymbol', '')
                target = row.get('target_genesymbol', '')

                if not tf or not target or pd.isna(tf) or pd.isna(target):
                    continue

                # Determine mode of regulation
                is_stimulation = row.get('is_stimulation', 0)
                is_inhibition = row.get('is_inhibition', 0)

                if is_stimulation and not is_inhibition:
                    mor = "activation"
                    mor_score = 1
                elif is_inhibition and not is_stimulation:
                    mor = "repression"
                    mor_score = -1
                elif is_stimulation and is_inhibition:
                    mor = "dual"  # Can both activate and repress
                    mor_score = 0
                else:
                    mor = "unknown"
                    mor_score = 0

                interaction = {
                    "tf_symbol": tf,
                    "target_gene": target,
                    "tf_uniprot": row.get('source', ''),
                    "target_uniprot": row.get('target', ''),
                    "confidence": row.get('dorothea_level', ''),
                    "curation_effort": row.get('curation_effort', 0),
                    "mode_of_regulation": mor,
                    "mor_score": mor_score,
                    "is_directed": row.get('is_directed', 1),
                    "relationship": "transcriptionFactorInteractsWithGene",
                    "source_database": "DoRothEA"
                }
                interactions.append(interaction)

            logger.info(f"Total TF-gene interactions: {len(interactions)}")

            # Log confidence level breakdown
            confidence_counts = {}
            for i in interactions:
                conf = i.get('confidence', 'unknown')
                confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
            logger.info(f"Interactions by confidence: {confidence_counts}")

            # Log mode of regulation breakdown
            mor_counts = {}
            for i in interactions:
                mor = i.get('mode_of_regulation', 'unknown')
                mor_counts[mor] = mor_counts.get(mor, 0) + 1
            logger.info(f"Interactions by mode: {mor_counts}")

            return {
                f'{DOROTHEA_TRANSCRIPTION_FACTORS}': pd.DataFrame(tf_nodes),
                f'{DOROTHEA_TF_GENE_INTERACTIONS}': pd.DataFrame(interactions)
            }

        except Exception as e:
            logger.error(f"Error parsing DoRothEA: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """
        Get the schema for DoRothEA data.

        Returns:
            Dictionary defining the schema for TF nodes and interactions
        """
        return {
            f'{DOROTHEA_TRANSCRIPTION_FACTORS}': {
                "tf_symbol": "Transcription factor gene symbol",
                "node_type": "Node type (TranscriptionFactor)",
                "source_database": "Data source (DoRothEA)"
            },
            f'{DOROTHEA_TF_GENE_INTERACTIONS}': {
                "tf_symbol": "Transcription factor gene symbol",
                "target_gene": "Target gene symbol",
                "tf_uniprot": "TF UniProt accession",
                "target_uniprot": "Target gene UniProt accession",
                "confidence": "DoRothEA confidence level (A-D)",
                "curation_effort": "Number of supporting publications/databases",
                "mode_of_regulation": "Mode of regulation (activation/repression/dual/unknown)",
                "mor_score": "Mode of regulation score (1=activation, -1=repression, 0=unknown/dual)",
                "is_directed": "Whether the interaction is directed (1=yes)",
                "relationship": "Relationship type (transcriptionFactorInteractsWithGene)",
                "source_database": "Data source (DoRothEA)"
            }
        }
