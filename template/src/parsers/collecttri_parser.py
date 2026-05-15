"""
CollectTRI Parser for the knowledge graph.

This module parses CollectTRI data to extract transcription factor nodes
and TF-gene regulatory relationships for the knowledge graph.

CollectTRI is a comprehensive collection of signed TF-target interactions
compiled from 12 different resources. Data is retrieved from the OmniPath
API which provides CollectTRI interactions.

Data Source:
  - OmniPath API: https://omnipathdb.org/

Output:
  - transcription_factors: unique TranscriptionFactor nodes
  - tf_gene_interactions: transcriptionFactorInteractsWithGene relationships

References:
  - https://saezlab.github.io/CollectTRI/
  - CollectTRI resource: https://doi.org/10.1093/nar/gkad841
"""

import logging
import pandas as pd
import requests

from pathlib import Path
from typing import Dict
from .base_parser import BaseParser

COLLECTTRI_TRANSCRIPTION_FACTORS = 'transcription_factors'
COLLECTTRI_TF_GENE_INTERACTIONS = 'tf_gene_interactions'

logger = logging.getLogger(__name__)


class CollectTRIParser(BaseParser):
    """
    Parser for the CollectTRI transcription factor regulatory network.

    Extracts TF-gene regulatory relationships for use in the knowledge graph's
    transcriptionFactorInteractsWithGene relationships.

    Uses the OmniPath API to retrieve CollectTRI data, which provides:
    - Directed TF-target interactions signed with stimulation / inhibition flags
    - Consensus direction, stimulation, and inhibition scores across resources
    """

    # OmniPath API endpoint for CollectTRI interactions
    OMNIPATH_COLLECTTRI_URL = (
        "https://omnipathdb.org/interactions"
        "?datasets=collectri"
        "&genesymbols=1"
    )

    def __init__(self, data_dir: str):
        """
        Initialize the CollectTRI parser.

        Args:
            data_dir: Directory to store downloaded and processed data
        """
        super().__init__(data_dir)

    def download_data(self) -> bool:
        """
        Download CollectTRI data from the OmniPath API.

        Returns:
            True if successful, False otherwise
        """
        logger.info("Downloading CollectTRI TF-gene regulatory network from OmniPath...")

        collecttri_path = self.get_file_path(f"{COLLECTTRI_TF_GENE_INTERACTIONS}.tsv")
        if Path(collecttri_path).exists() and not self.force:
            logger.info(f"File already exists: {collecttri_path}")
            return True

        try:
            response = requests.get(self.OMNIPATH_COLLECTTRI_URL, timeout=120)
            response.raise_for_status()

            with open(collecttri_path, 'w') as f:
                f.write(response.text)

            # Count lines to verify download
            line_count = len(response.text.strip().split('\n'))
            logger.info(f"Successfully downloaded CollectTRI to {collecttri_path}")
            logger.info(f"Downloaded {line_count - 1} interactions (excluding header)")

            return True

        except Exception as e:
            logger.error(f"Failed to download CollectTRI from OmniPath: {e}")
            return False

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse CollectTRI TF-gene regulatory data.

        Returns:
            Dictionary with:
              - 'transcription_factors': unique TF nodes with tf_symbol and source_database
              - 'tf_gene_interactions': TF-gene regulatory relationships
        """
        collecttri_path = self.get_file_path(f"{COLLECTTRI_TF_GENE_INTERACTIONS}.tsv")

        if not Path(collecttri_path).exists():
            logger.error(f"CollectTRI file not found: {collecttri_path}")
            return {}

        logger.info(f"Parsing CollectTRI from {collecttri_path}")

        try:
            # Read CollectTRI data from OmniPath.
            # Expected columns:
            #   source, target, source_genesymbol, target_genesymbol,
            #   is_directed, is_stimulation, is_inhibition,
            #   consensus_direction, consensus_stimulation, consensus_inhibition
            df = pd.read_csv(collecttri_path, sep='\t')

            logger.info(f"Loaded {len(df)} CollectTRI TF-target interactions")

            # Validate required columns are present
            required_cols = [
                'source_genesymbol', 'target_genesymbol',
                'is_directed', 'is_stimulation', 'is_inhibition',
                'consensus_direction', 'consensus_stimulation', 'consensus_inhibition'
            ]
            if not self.validate_data(df, required_cols):
                logger.error("CollectTRI data is missing required columns")
                return {}

            # Drop rows where TF or target gene symbol is missing
            df = df.dropna(subset=['source_genesymbol', 'target_genesymbol'])
            logger.info(f"After dropping rows with missing gene symbols: {len(df)} interactions")

            # ----------------------------------------------------------------
            # Build transcription_factors DataFrame
            # One row per unique TF (source_genesymbol)
            # ----------------------------------------------------------------
            unique_tfs = df['source_genesymbol'].unique()
            tf_df = pd.DataFrame({
                'tf_symbol': unique_tfs,
                'source_database': 'CollectTRI'
            })
            logger.info(f"Found {len(tf_df)} unique transcription factors")

            # ----------------------------------------------------------------
            # Build tf_gene_interactions DataFrame
            # One row per TF-target pair with all regulatory flags
            # ----------------------------------------------------------------
            interactions_df = pd.DataFrame({
                'tf_symbol':              df['source_genesymbol'].values,
                'target_gene':            df['target_genesymbol'].values,
                'is_directed':            df['is_directed'].values,
                'is_stimulation':         df['is_stimulation'].values,
                'is_inhibition':          df['is_inhibition'].values,
                'consensus_direction':    df['consensus_direction'].values,
                'consensus_stimulation':  df['consensus_stimulation'].values,
                'consensus_inhibition':   df['consensus_inhibition'].values,
                'source_database':        'CollectTRI'
            })

            logger.info(f"Total TF-gene interactions: {len(interactions_df)}")

            # Log stimulation / inhibition breakdown
            stim_count = int(interactions_df['is_stimulation'].sum())
            inhib_count = int(interactions_df['is_inhibition'].sum())
            logger.info(
                f"Stimulatory interactions: {stim_count}, "
                f"Inhibitory interactions: {inhib_count}"
            )

            return {
                COLLECTTRI_TRANSCRIPTION_FACTORS: tf_df,
                COLLECTTRI_TF_GENE_INTERACTIONS: interactions_df
            }

        except Exception as e:
            logger.error(f"Error parsing CollectTRI: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {}

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """
        Get the schema for CollectTRI data.

        Returns:
            Dictionary defining the schema for TF nodes and interactions
        """
        return {
            COLLECTTRI_TRANSCRIPTION_FACTORS: {
                'tf_symbol':        'Transcription factor gene symbol (source_genesymbol)',
                'source_database':  'Data source (CollectTRI)'
            },
            COLLECTTRI_TF_GENE_INTERACTIONS: {
                'tf_symbol':             'Transcription factor gene symbol (source_genesymbol)',
                'target_gene':           'Target gene symbol (target_genesymbol)',
                'is_directed':           'Whether the interaction is directed',
                'is_stimulation':        'Whether the TF activates / stimulates the target',
                'is_inhibition':         'Whether the TF represses / inhibits the target',
                'consensus_direction':   'Consensus across resources that the interaction is directed',
                'consensus_stimulation': 'Consensus across resources for stimulation',
                'consensus_inhibition':  'Consensus across resources for inhibition',
                'source_database':       'Data source (CollectTRI)'
            }
        }
