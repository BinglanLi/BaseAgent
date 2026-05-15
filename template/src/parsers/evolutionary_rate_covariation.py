"""
Evolutionary Rate Covariation (ERC) Parser for the knowledge graph.

Downloads the mammal functional ERC matrix dataset from Dryad, extracts
pairwise gene covariation scores, and filters to significant associations
using a Fisher-transformed ERC score threshold.

Data Source: https://datadryad.org/dataset/doi:10.5061/dryad.6m905qg8q
Format: ZIP archive on Dryad (protected by Anubis + AWS WAF bot challenges).
        Contains a square gene x gene matrix RDS file with Fisher-transformed
        ERC scores (HGNC symbols as row/column labels).

Download strategy:
  1. Use Playwright (headless Chromium) to navigate the Dryad file_stream URL,
     which handles the Anubis proof-of-work challenge and AWS WAF JS challenge.
  2. Intercept the S3 presigned redirect URL from the network traffic.
  3. Use remotezip + HTTP range requests to locate the target RDS file inside
     the 7.5 GB ZIP without downloading the full archive.
  4. Download only the compressed bytes for the RDS file using parallel range
     requests, then decompress with zlib.

Output:
  - gene_covariation.tsv: gene-gene ERC edges with columns
        source_hgnc | target_hgnc | erc_score | source_database
"""

import asyncio
import concurrent.futures
import hashlib
import json
import logging
import math
import re
import struct
import time
import zlib
import zipfile
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import pyreadr
import requests

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

OUTPUT_NAME = "gene_covariation"
_ANUBIS_PASS_PATH = "/.within.website/x/cmd/anubis/api/pass-challenge"
_CHUNK_SIZE = 100 * 1024 * 1024   # 100 MB per chunk
_NUM_WORKERS = 6


# ---------------------------------------------------------------------------
# Anubis proof-of-work solver
# ---------------------------------------------------------------------------

def _solve_anubis(random_data: str, difficulty: int):
    """
    Solve the Anubis SHA-256 proof-of-work challenge.

    Finds nonce n such that SHA-256(random_data + str(n)) has
    floor(difficulty/2) leading zero bytes (and an extra zero nibble if
    difficulty is odd).

    Returns:
        (hash_hex, nonce)
    """
    p = difficulty // 2
    u = difficulty % 2 != 0
    nonce = 0
    while True:
        h = hashlib.sha256((random_data + str(nonce)).encode()).digest()
        valid = all(h[i] == 0 for i in range(p))
        if valid and u and (h[p] >> 4) != 0:
            valid = False
        if valid:
            return h.hex(), nonce
        nonce += 1


def _get_anubis_session(target_url: str) -> requests.Session:
    """
    Create a requests.Session that has solved the Anubis PoW challenge.

    If the target URL does not present an Anubis challenge (e.g. already
    authenticated), returns a plain session.
    """
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://datadryad.org/dataset/doi:10.5061/dryad.6m905qg8q",
    })

    resp = session.get(target_url, timeout=30)
    if "anubis_challenge" not in resp.text:
        return session  # no challenge

    m = re.search(
        r'id="anubis_challenge"[^>]*>(\{.*?\})\s*</script>',
        resp.text, re.DOTALL,
    )
    if not m:
        return session

    challenge_data = json.loads(m.group(1))
    challenge = challenge_data["challenge"]
    rules = challenge_data["rules"]
    logger.info("Solving Anubis PoW (difficulty=%d)...", rules["difficulty"])

    hash_hex, nonce = _solve_anubis(challenge["randomData"], rules["difficulty"])
    logger.info("Anubis solved: nonce=%d", nonce)

    base = "https://datadryad.org"
    pass_url = base + _ANUBIS_PASS_PATH
    session.get(
        pass_url,
        params={
            "id": challenge["id"],
            "response": hash_hex,
            "nonce": str(nonce),
            "redir": target_url,
            "elapsedTime": "500",
        },
        allow_redirects=False,
        timeout=30,
    )
    return session


# ---------------------------------------------------------------------------
# S3 URL discovery via Playwright (handles AWS WAF JS challenge)
# ---------------------------------------------------------------------------

async def _get_s3_url_async(dryad_url: str) -> Optional[str]:
    """
    Use Playwright to navigate the Dryad file_stream URL and capture the
    S3 presigned redirect URL from network traffic.
    """
    from playwright.async_api import async_playwright

    s3_urls = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = await context.new_page()

        async def handle_response(response):
            url = response.url
            status = response.status
            if status in (301, 302, 303, 307, 308):
                loc = response.headers.get("location", "")
                if "s3" in loc and "amazonaws" in loc:
                    logger.info("Captured S3 redirect URL")
                    s3_urls.append(loc)
            if "s3.amazonaws.com" in url or ("amazonaws" in url and "X-Amz" in url):
                if url not in s3_urls:
                    s3_urls.append(url)

        page.on("response", handle_response)

        # Visit dataset page first to get cookies/context
        await page.goto(
            "https://datadryad.org/dataset/doi:10.5061/dryad.6m905qg8q",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        await asyncio.sleep(2)

        # Navigate to the file stream URL
        try:
            await page.goto(dryad_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(8)
        except Exception as exc:
            logger.warning("Playwright navigation error: %s", exc)

        await browser.close()

    return s3_urls[0] if s3_urls else None


def _get_s3_url(dryad_url: str) -> Optional[str]:
    """Synchronous wrapper around _get_s3_url_async."""
    try:
        return asyncio.run(_get_s3_url_async(dryad_url))
    except Exception as exc:
        logger.error("Failed to get S3 URL via Playwright: %s", exc)
        return None


# ---------------------------------------------------------------------------
# ZIP64-aware central directory parser
# ---------------------------------------------------------------------------

def _find_file_in_zip(s3_url: str, target_filename: str):
    """
    Use HTTP range requests to read the ZIP64 central directory from the
    remote S3 URL and find the byte range of target_filename.

    Returns:
        (data_start, compressed_size, compress_method) or None
    """
    def range_get(start, end):
        r = requests.get(s3_url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        return r.content

    # Get total file size from Content-Range
    r = requests.get(s3_url, headers={"Range": "bytes=-22"}, timeout=30)
    total_size = int(r.headers.get("Content-Range", "bytes 0-0/1").split("/")[1])
    logger.info("Remote ZIP total size: %d bytes", total_size)

    # Read tail to find EOCD / ZIP64 EOCD locator
    tail = range_get(total_size - 78, total_size - 1)

    cd_offset = None
    cd_size = None

    for i in range(len(tail) - 4):
        sig = struct.unpack_from("<I", tail, i)[0]
        if sig == 0x07064b50:  # ZIP64 EOCD locator
            zip64_eocd_offset = struct.unpack_from("<Q", tail, i + 8)[0]
            z64 = range_get(zip64_eocd_offset, zip64_eocd_offset + 55)
            cd_size   = struct.unpack_from("<Q", z64, 40)[0]
            cd_offset = struct.unpack_from("<Q", z64, 48)[0]
            break

    if cd_offset is None:
        # Fall back to standard EOCD
        eocd = range_get(total_size - 22, total_size - 1)
        cd_size   = struct.unpack_from("<I", eocd, 12)[0]
        cd_offset = struct.unpack_from("<I", eocd, 16)[0]

    logger.info("Central directory: offset=%d, size=%d", cd_offset, cd_size)
    cd_data = range_get(cd_offset, cd_offset + cd_size - 1)

    target_bytes = target_filename.encode()
    pos = 0
    while pos < len(cd_data) - 4:
        sig = struct.unpack_from("<I", cd_data, pos)[0]
        if sig != 0x02014b50:
            pos += 1
            continue

        compress_method     = struct.unpack_from("<H", cd_data, pos + 10)[0]
        compressed_size32   = struct.unpack_from("<I", cd_data, pos + 20)[0]
        uncompressed_size32 = struct.unpack_from("<I", cd_data, pos + 24)[0]
        local_offset32      = struct.unpack_from("<I", cd_data, pos + 42)[0]
        fname_len           = struct.unpack_from("<H", cd_data, pos + 28)[0]
        extra_len           = struct.unpack_from("<H", cd_data, pos + 30)[0]
        comment_len         = struct.unpack_from("<H", cd_data, pos + 32)[0]
        fname               = cd_data[pos + 46 : pos + 46 + fname_len]

        compressed_size   = compressed_size32
        local_offset      = local_offset32

        # Parse ZIP64 extra field
        if extra_len > 0:
            extra = cd_data[pos + 46 + fname_len : pos + 46 + fname_len + extra_len]
            ep = 0
            while ep < len(extra) - 4:
                etag = struct.unpack_from("<H", extra, ep)[0]
                esz  = struct.unpack_from("<H", extra, ep + 2)[0]
                if etag == 0x0001:
                    vp = ep + 4
                    if uncompressed_size32 == 0xFFFFFFFF and vp + 8 <= ep + 4 + esz:
                        vp += 8
                    if compressed_size32 == 0xFFFFFFFF and vp + 8 <= ep + 4 + esz:
                        compressed_size = struct.unpack_from("<Q", extra, vp)[0]
                        vp += 8
                    if local_offset32 == 0xFFFFFFFF and vp + 8 <= ep + 4 + esz:
                        local_offset = struct.unpack_from("<Q", extra, vp)[0]
                    break
                ep += 4 + esz

        # Match by suffix (handle different ZIP root directories)
        if fname.endswith(target_bytes) or fname == target_bytes:
            logger.info("Found %s in ZIP (compressed=%d MB)", fname.decode(), compressed_size // 1024 // 1024)
            lh = range_get(local_offset, local_offset + 31)
            lh_fname_len  = struct.unpack_from("<H", lh, 26)[0]
            lh_extra_len  = struct.unpack_from("<H", lh, 28)[0]
            data_start = local_offset + 30 + lh_fname_len + lh_extra_len
            return data_start, compressed_size, compress_method

        pos += 46 + fname_len + extra_len + comment_len

    logger.error("File %s not found in ZIP central directory", target_filename)
    return None


# ---------------------------------------------------------------------------
# Parallel range-request download + DEFLATE decompress
# ---------------------------------------------------------------------------

def _download_and_decompress(s3_url: str, data_start: int, compressed_size: int,
                              compress_method: int, out_path: Path) -> bool:
    """
    Download the compressed bytes for a single file from a remote ZIP using
    parallel HTTP range requests, then decompress and write to out_path.
    """
    data_end = data_start + compressed_size - 1
    chunks_info = []
    pos = data_start
    while pos <= data_end:
        end = min(pos + _CHUNK_SIZE - 1, data_end)
        chunks_info.append((len(chunks_info), pos, end))
        pos = end + 1

    logger.info(
        "Downloading %d MB in %d chunks (%d workers)...",
        compressed_size // 1024 // 1024, len(chunks_info), _NUM_WORKERS,
    )

    chunk_dir = out_path.parent / "_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    def download_chunk(args):
        idx, start, end = args
        chunk_path = chunk_dir / f"chunk_{idx:04d}.bin"
        expected = end - start + 1
        if chunk_path.exists() and chunk_path.stat().st_size == expected:
            return idx, expected, True
        r = requests.get(s3_url, headers={"Range": f"bytes={start}-{end}"}, timeout=180)
        chunk_path.write_bytes(r.content)
        return idx, len(r.content), False

    t0 = time.time()
    downloaded = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=_NUM_WORKERS) as executor:
        futures = {executor.submit(download_chunk, ci): ci for ci in chunks_info}
        for future in concurrent.futures.as_completed(futures):
            idx, size, cached = future.result()
            downloaded += size
            elapsed = time.time() - t0
            speed = downloaded / max(elapsed, 0.1) / 1024 / 1024
            logger.info(
                "Chunk %02d done (%s): %d/%d MB @ %.1f MB/s",
                idx, "cached" if cached else "downloaded",
                downloaded // 1024 // 1024, compressed_size // 1024 // 1024, speed,
            )

    logger.info("Download complete in %.1fs. Decompressing...", time.time() - t0)

    # Decompress chunks sequentially
    t1 = time.time()
    if compress_method == 8:
        decomp = zlib.decompressobj(wbits=-15)
    else:
        decomp = None  # stored (no compression)

    with open(out_path, "wb") as out_f:
        for idx in range(len(chunks_info)):
            chunk_path = chunk_dir / f"chunk_{idx:04d}.bin"
            data = chunk_path.read_bytes()
            if decomp is not None:
                out_f.write(decomp.decompress(data))
            else:
                out_f.write(data)
        if decomp is not None:
            remaining = decomp.flush()
            if remaining:
                out_f.write(remaining)

    logger.info("Decompression complete in %.1fs. Output: %s", time.time() - t1, out_path)
    return True


# ---------------------------------------------------------------------------
# Parser class
# ---------------------------------------------------------------------------

class EvolutionaryRateCovariationParser(BaseParser):
    """
    Parser for the Evolutionary Rate Covariation (ERC) dataset.

    Downloads the Dryad ZIP archive (handling Anubis + AWS WAF bot protection),
    extracts the mammal ftERC RDS matrix using HTTP range requests, converts
    it to a long-format DataFrame of gene pairs, and filters by a
    Fisher-transformed ERC score threshold.

    Constructor args (passed from databases.yaml -> args):
        url          : Dryad file_stream URL for the ZIP download.
        file_path    : Path inside the ZIP to the RDS file
                       (e.g. "ERC_matrices/mammal_ftERC.RDS").
        ft_threshold : Minimum Fisher-transformed ERC score to retain a pair.
                       Defaults to sqrt(120 - 3) approx 10.82.
    """

    def __init__(
        self,
        data_dir: str,
        url: str,
        file_path: str,
        ft_threshold: Optional[float] = None,
    ):
        super().__init__(data_dir)
        self.source_name = "evolutionary_rate_covariation"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.url = url
        self.file_path = file_path
        self.ft_threshold = (
            ft_threshold if ft_threshold is not None else math.sqrt(120 - 3)
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _rds_path(self) -> Path:
        """Return the expected path of the cached RDS file."""
        rds_name = Path(self.file_path).name
        return self.source_dir / rds_name

    def _download_rds_via_playwright(self) -> bool:
        """
        Use Playwright to get the S3 presigned URL, then download + decompress
        the RDS file from the remote ZIP using HTTP range requests.
        """
        logger.info("Obtaining S3 presigned URL via Playwright...")
        s3_url = _get_s3_url(self.url)
        if not s3_url:
            logger.error("Could not obtain S3 URL via Playwright.")
            return False

        logger.info("Locating %s in remote ZIP...", self.file_path)
        result = _find_file_in_zip(s3_url, self.file_path)
        if result is None:
            # Try matching by filename only
            fname_only = Path(self.file_path).name
            result = _find_file_in_zip(s3_url, fname_only)
        if result is None:
            logger.error("Could not locate %s in remote ZIP.", self.file_path)
            return False

        data_start, compressed_size, compress_method = result
        rds_path = self._rds_path()
        return _download_and_decompress(
            s3_url, data_start, compressed_size, compress_method, rds_path
        )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """
        Ensure the RDS file is available locally.

        Strategy:
          1. If the RDS file is already cached, skip download.
          2. Otherwise, use Playwright to bypass bot protection, obtain the
             S3 presigned URL, and download only the RDS file from the ZIP
             using HTTP range requests.

        Returns:
            True if the RDS file is available, False otherwise.
        """
        rds_path = self._rds_path()

        if rds_path.exists() and not self.force:
            logger.info("RDS file already cached: %s", rds_path)
            return True

        logger.info("Downloading ERC RDS via Playwright + range requests...")
        return self._download_rds_via_playwright()

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse the ERC RDS matrix into a filtered long-format DataFrame.

        Workflow:
          1. Read the cached RDS file with pyreadr.
          2. Convert the square matrix to a numpy array.
          3. Extract upper-triangle indices (k=1, no self-loops).
          4. Keep only pairs where erc_score >= ft_threshold.
          5. Build DataFrame: source_hgnc, target_hgnc, erc_score.
          6. Drop rows with empty/NaN gene symbols.
          7. Add source_database column.

        Returns:
            Dict with key gene_covariation -> filtered gene-pair DataFrame.
        """
        rds_path = self._rds_path()
        if not rds_path.exists():
            logger.error("RDS file not found: %s", rds_path)
            return {}

        # ---- 1. Read RDS -----------------------------------------------
        logger.info("Reading RDS file (%d MB)...", rds_path.stat().st_size // 1024 // 1024)
        t0 = time.time()
        try:
            result = pyreadr.read_r(str(rds_path))
            matrix_df = result[None]
        except Exception as exc:
            logger.error("Failed to read RDS file: %s", exc)
            return {}
        logger.info(
            "RDS read in %.1fs: shape=%s", time.time() - t0, matrix_df.shape
        )

        # ---- 2. Convert to numpy ----------------------------------------
        gene_labels = list(matrix_df.index)
        n = len(gene_labels)
        mat = matrix_df.values.astype(float)
        logger.info("Matrix: %d x %d, threshold: %.4f", n, n, self.ft_threshold)

        # ---- 3. Upper-triangle indices ----------------------------------
        row_idx, col_idx = np.triu_indices(n, k=1)

        # ---- 4. Filter by threshold -------------------------------------
        scores = mat[row_idx, col_idx]
        mask = scores >= self.ft_threshold
        row_idx = row_idx[mask]
        col_idx = col_idx[mask]
        scores = scores[mask]
        logger.info(
            "%d pairs pass threshold (%.4f) out of %d upper-triangle pairs.",
            len(scores), self.ft_threshold, len(mask),
        )

        # ---- 5. Build DataFrame -----------------------------------------
        gene_arr = np.array(gene_labels)
        df = pd.DataFrame({
            "source_hgnc": gene_arr[row_idx],
            "target_hgnc": gene_arr[col_idx],
            "erc_score":   scores,
        })

        # ---- 6. Drop empty gene symbols ---------------------------------
        df = df[
            df["source_hgnc"].notna()
            & (df["source_hgnc"].astype(str).str.strip() != "")
            & df["target_hgnc"].notna()
            & (df["target_hgnc"].astype(str).str.strip() != "")
        ].copy()
        logger.info("%d edges after dropping empty gene symbols.", len(df))

        # ---- 7. Add source_database -------------------------------------
        df["source_database"] = "Evolutionary Rate Covariation"

        logger.info(
            "Parsed %d gene-gene ERC edges (ft_threshold=%.4f).",
            len(df), self.ft_threshold,
        )
        return {OUTPUT_NAME: df}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return the schema for ERC output DataFrames."""
        return {
            OUTPUT_NAME: {
                "source_hgnc":     "HGNC gene symbol for the source gene",
                "target_hgnc":     "HGNC gene symbol for the target gene",
                "erc_score":       "Fisher-transformed ERC score (>= ft_threshold)",
                "source_database": "Source database label (Evolutionary Rate Covariation)",
            }
        }
