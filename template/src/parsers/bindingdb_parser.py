"""
BindingDB Parser for the knowledge graph.

Downloads the BindingDB bulk TSV, extracts drug-gene binding relationships
(chemicalBindsGene) keyed by DrugBank ID and target gene name.

Data Source: https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp

Output:
  - drug_binds_gene.tsv: chemicalBindsGene edges with columns
      drugbank_id | target_name | source
"""

import logging
import re
import zipfile
from pathlib import Path

import pandas as pd
import requests

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

OUTPUT_NAME = "drug_binds_gene"

# BindingDB column names as they appear in the TSV
_COL_DRUGBANK = "DrugBank ID of Ligand"
_COL_TARGET   = "Target Name"
_COL_ORGANISM = "Target Source Organism According to Curator or DataSource"

DOWNLOAD_PAGE = (
    "https://www.bindingdb.org/rwd/bind/chemsearch/marvin/Download.jsp"
)
BASE_URL = "https://www.bindingdb.org"


class BindingDBParser(BaseParser):
    """
    Parser for BindingDB.

    Produces a single output — drug_binds_gene.tsv — containing
    (drugbank_id, target_name, source) triples for all human-target
    binding entries that carry a DrugBank identifier.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        # source_name is auto-derived as "bindingdb" by BaseParser

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _discover_tsv_url(self) -> str:
        """
        Fetch the BindingDB download page and return the URL of the
        latest BindingDB_All_*_tsv.zip file.

        Falls back to the most recent known URL if discovery fails.
        """
        fallback = (
            BASE_URL + "/rwd/bind/downloads/BindingDB_All_202605_tsv.zip"
        )
        try:
            resp = requests.get(DOWNLOAD_PAGE, timeout=30)
            resp.raise_for_status()
            # Look for the all-data TSV zip link
            matches = re.findall(
                r'/rwd/bind/downloads/(BindingDB_All_\d+_tsv\.zip)',
                resp.text,
            )
            if not matches:
                logger.warning(
                    "Could not find TSV zip link on BindingDB download page; "
                    "using fallback URL."
                )
                return fallback
            # Use the first (most prominent) match
            filename = matches[0]
            url = f"{BASE_URL}/rwd/bind/downloads/{filename}"
            logger.info(f"Discovered BindingDB TSV URL: {url}")
            return url
        except Exception as exc:
            logger.warning(
                f"URL discovery failed ({exc}); using fallback URL."
            )
            return fallback

    def _find_extracted_tsv(self) -> Path | None:
        """Return the path to an already-extracted BindingDB TSV, or None."""
        candidates = sorted(self.source_dir.glob("BindingDB_All*.tsv"))
        return candidates[0] if candidates else None

    def _is_valid_zip(self, path: Path) -> bool:
        """Return True if *path* is a readable, complete ZIP archive."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                zf.namelist()          # forces reading the central directory
            return True
        except Exception:
            return False

    def _download_large_file(self, url: str, dest: Path) -> bool:
        """
        Stream-download *url* to *dest* with no read-timeout and 1 MB chunks.

        Uses timeout=(30, None): 30 s to establish the connection, then no
        timeout on the read so very large files (500 MB+) are not cut off.
        Returns True on success, False on any error.
        """
        logger.info(f"Downloading {url}")
        try:
            with requests.get(url, stream=True, timeout=(30, None)) as resp:
                resp.raise_for_status()

                content_length = resp.headers.get("content-length")
                expected_bytes = int(content_length) if content_length else None
                if expected_bytes:
                    logger.info(
                        f"Expected download size: "
                        f"{expected_bytes / 1024 / 1024:.1f} MB"
                    )

                written = 0
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                            written += len(chunk)

            logger.info(
                f"Download complete: {written / 1024 / 1024:.1f} MB written "
                f"to {dest}"
            )

            # Sanity-check: did we receive all advertised bytes?
            if expected_bytes and written < expected_bytes * 0.99:
                logger.error(
                    f"Download appears truncated: received {written} bytes "
                    f"but expected {expected_bytes} bytes."
                )
                return False

            return True

        except Exception as exc:
            logger.error(f"Download failed: {exc}")
            return False

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Download and extract the BindingDB TSV zip."""
        # Skip if already extracted
        if self._find_extracted_tsv() and not self.force:
            logger.info("BindingDB TSV already extracted; skipping download.")
            return True

        url = self._discover_tsv_url()
        zip_filename = url.rsplit("/", 1)[-1]
        zip_path = Path(self.get_file_path(zip_filename))

        # If a cached ZIP exists, validate it before trusting it
        if zip_path.exists() and not self.force:
            if self._is_valid_zip(zip_path):
                logger.info(
                    f"Valid cached ZIP found: {zip_path}; skipping download."
                )
            else:
                logger.warning(
                    f"Cached ZIP {zip_path} is corrupt or incomplete; "
                    "deleting and re-downloading."
                )
                zip_path.unlink()

        # Download if we still need the file
        if not zip_path.exists() or self.force:
            ok = self._download_large_file(url, zip_path)
            if not ok:
                logger.error("Failed to download BindingDB zip.")
                return False

        # Validate the freshly downloaded (or previously cached) ZIP
        if not self._is_valid_zip(zip_path):
            logger.error(
                f"Downloaded ZIP {zip_path} failed integrity check "
                "(BadZipFile). The file may be truncated or the server "
                "returned an error page."
            )
            return False

        # Extract
        try:
            logger.info(f"Extracting {zip_path} …")
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.source_dir)
            logger.info("BindingDB zip extracted successfully.")
            return True
        except Exception as exc:
            logger.error(f"Failed to extract BindingDB zip: {exc}")
            return False

    def parse_data(self) -> dict[str, pd.DataFrame]:
        """
        Parse the BindingDB TSV and return drug-gene binding edges.

        Filters to:
          - Human (Homo sapiens) targets
          - Rows with a non-empty DrugBank ID
          - Rows with a non-empty target name

        Returns:
            {"drug_binds_gene": DataFrame[drugbank_id, target_name, source]}
        """
        tsv_path = self._find_extracted_tsv()
        if tsv_path is None:
            logger.error("BindingDB TSV not found; run download_data() first.")
            return {}

        logger.info(f"Parsing BindingDB from {tsv_path} …")

        # Only load the three columns we need to keep memory usage manageable
        usecols = [_COL_DRUGBANK, _COL_TARGET, _COL_ORGANISM]

        try:
            df = pd.read_csv(
                tsv_path,
                sep="\t",
                usecols=usecols,
                low_memory=False,
                on_bad_lines="skip",
                dtype=str,
            )
        except Exception as exc:
            logger.error(f"Failed to read BindingDB TSV: {exc}")
            return {}

        logger.info(f"Loaded {len(df):,} raw BindingDB rows.")

        # --- Filter to human targets ---
        if _COL_ORGANISM in df.columns:
            mask_human = df[_COL_ORGANISM].str.contains(
                "Homo sapiens", case=False, na=False
            )
            df = df[mask_human]
            logger.info(f"After human-target filter: {len(df):,} rows.")

        # --- Require non-empty DrugBank ID ---
        df[_COL_DRUGBANK] = df[_COL_DRUGBANK].str.strip()
        mask_db = df[_COL_DRUGBANK].notna() & (df[_COL_DRUGBANK] != "")
        df = df[mask_db]
        logger.info(f"After DrugBank ID filter: {len(df):,} rows.")

        # --- Require non-empty target name ---
        df[_COL_TARGET] = df[_COL_TARGET].str.strip()
        mask_tgt = df[_COL_TARGET].notna() & (df[_COL_TARGET] != "")
        df = df[mask_tgt]
        logger.info(f"After target-name filter: {len(df):,} rows.")

        # --- Rename and add source column ---
        out = df[[_COL_DRUGBANK, _COL_TARGET]].rename(
            columns={
                _COL_DRUGBANK: "drugbank_id",
                _COL_TARGET:   "target_name",
            }
        )
        out["source_database"] = "BindingDB"

        # --- Deduplicate ---
        out = out.drop_duplicates(subset=["drugbank_id", "target_name"])
        logger.info(f"Final drug_binds_gene edges: {len(out):,} rows.")

        return {OUTPUT_NAME: out}

    def get_schema(self) -> dict[str, dict[str, str]]:
        return {
            OUTPUT_NAME: {
                "drugbank_id": (
                    "DrugBank identifier of the ligand/drug "
                    "(e.g. DB00001)"
                ),
                "target_name": (
                    "Name of the protein target / gene symbol as recorded "
                    "in BindingDB"
                ),
                "source_database": "Source database name (BindingDB)",
            }
        }
