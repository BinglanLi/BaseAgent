"""
Normalize clinical trial condition names for Disease Ontology matching.

ClinicalTrials.gov conditions are free-text (title case, parenthetical
abbreviations). Disease Ontology names are lowercase without qualifiers.
This normalizer bridges the gap so the ista populate step can match them.
"""

import logging
import re
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def normalize_condition(name: str) -> str:
    if not isinstance(name, str):
        return name
    name = re.sub(r'\s*\([^)]*\)\s*', ' ', name)
    name = name.strip().lower()
    name = re.sub(r'\s+', ' ', name)
    return name


def normalize_conditions(processed_dir: Path) -> int:
    """Normalize condition names in trial_disease_associations.tsv.

    Returns the number of rows updated.
    """
    tsv_path = processed_dir / "clinicaltrials" / "trial_disease_associations.tsv"
    if not tsv_path.exists():
        logger.warning("trial_disease_associations.tsv not found; skipping normalization")
        return 0

    df = pd.read_csv(tsv_path, sep="\t", dtype=str)
    if "condition" not in df.columns:
        logger.warning("No 'condition' column found; skipping")
        return 0

    original = df["condition"].copy()
    df["condition"] = df["condition"].apply(normalize_condition)
    changed = (original != df["condition"]).sum()

    df.to_csv(tsv_path, sep="\t", index=False)
    logger.info("Normalized %d/%d condition names in trial_disease_associations.tsv",
                changed, len(df))
    return changed
