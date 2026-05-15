"""
MEDLINE Cooccurrence Parser for the knowledge graph.

Two-phase approach:

Phase 1 — PMID fetch (~1,224 EDirect calls, one per unique MeSH entity):
  esearch -db pubmed | efetch -format uid
  EDirect handles pagination internally, bypassing PubMed's 9,999-record
  REST API limit.  Results are cached per entity under data/raw/medline/pmids/.

Phase 2 — in-memory statistics (no additional API calls):
  For each pair (S, T) with corpus C = union(S-class) ∩ union(T-class):
    a = |S ∩ T|          (co-occurrence)
    b = |S| - a
    c = |T| - a
    d = |C| - a - b - c
    enrichment = a / (|S| × |T| / |C|)
    → one-tailed Fisher's exact test → odds_ratio, p_fisher

Compared to the previous pairwise-esearch approach (200,736 API calls),
this reduces calls by ~165×.

Data Sources:
  - data/processed/disease_ontology/slim_terms.tsv -- disease list (DOID)
  - data/raw/diseaseontology/doid.obo              -- disease->MeSH xref mapping
  - data/raw/mesh/desc{year}.xml                   -- MeSH ID->term name lookup
  - data/processed/mesh/symptom_nodes.tsv           -- symptom MeSH terms
  - data/processed/uberon/uberon_nodes.tsv          -- anatomy Uberon->MeSH mapping

PMID cache (data/raw/medline/pmids/):
  - {mesh_id}.txt.gz   (one PMID per line; reused across all three relation types)

Outputs (data/processed/medline/):
  - disease_symptom_cooccurrence.tsv  -> symptomManifestationOfDisease edges
  - disease_anatomy_cooccurrence.tsv  -> diseaseLocalizesToAnatomy edges
  - disease_disease_cooccurrence.tsv  -> diseaseAssociatesWithDisease edges

Requires EDirect CLI (esearch + efetch) installed under ./edirect/ in the
project root.  Install via:
  HOME=$(pwd) sh -c "$(curl -fsSL https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)"
Set NCBI_EUTILS_API_KEY in .env (or databases.yaml api_key_env) for
10 req/s throughput instead of the default 3 req/s.
"""

import gzip
import logging
import os
import shlex
import shutil
import subprocess
from pathlib import Path
from typing import Dict, FrozenSet, Optional

import pandas as pd
from lxml import etree
from scipy.stats import fisher_exact

from .base_parser import BaseParser
from config_loader import get_disease_scope

logger = logging.getLogger(__name__)

# EDirect install directory (project-local; falls back to system PATH)
_EDIRECT_DIR = str(Path(__file__).parent.parent.parent / "edirect")

MESH_YEAR = 2026

# Problematic DO<->MeSH cross-references (flagged in Disease Ontology issue tracker)
EXCLUDED_MESH_IDS = {"D003327", "D017202"}

# Output TSV stem names - must match source_filename in ontology_mappings.yaml
DS_OUTPUT = "disease_symptom_cooccurrence"
DA_OUTPUT = "disease_anatomy_cooccurrence"
DD_OUTPUT = "disease_disease_cooccurrence"


class MEDLINEParser(BaseParser):
    """
    Parser for MEDLINE literature co-occurrence data.

    Fetches per-entity PMID sets from PubMed via EDirect CLI, then computes
    pairwise co-occurrence statistics in memory using set intersection and
    Fisher's exact test.  See module docstring for full description.
    """

    def __init__(self, data_dir: str, api_key: str = None):
        super().__init__(data_dir)
        self.source_name = "medline"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key
        self._processed_dir = self.data_dir.parent / "processed"
        self._mesh_dir = self.data_dir / "mesh"
        self._pmid_cache_dir = self.source_dir / "pmids"
        self._pmid_cache_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # BaseParser interface
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """No direct downloads; depends on disease_ontology, mesh, and uberon parsers."""
        return True

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Two-phase computation:
          Phase 1 — fetch PMID sets for each unique MeSH entity via EDirect.
          Phase 2 — compute co-occurrence counts and Fisher stats in memory.
        """
        if not self._check_edirect():
            return {}

        mesh_names = self._load_mesh_names()
        if not mesh_names:
            logger.error("Could not load MeSH names; aborting")
            return {}

        disease_df = self._load_disease_list(mesh_names)
        if disease_df is None or disease_df.empty:
            logger.error("Empty disease list; aborting medline parse")
            return {}

        symptom_df = self._load_symptom_list()
        if symptom_df is None or symptom_df.empty:
            logger.error("Empty symptom list; aborting medline parse")
            return {}

        anatomy_df = self._load_anatomy_list(mesh_names)
        if anatomy_df is None or anatomy_df.empty:
            logger.error("Empty anatomy list; aborting medline parse")
            return {}

        # ---- Phase 1: fetch PMID sets ------------------------------------
        n_d, n_s, n_a = len(disease_df), len(symptom_df), len(anatomy_df)
        logger.info(
            f"Phase 1: fetching PMID sets via EDirect "
            f"({n_d} diseases + {n_s} symptoms + {n_a} anatomy = "
            f"~{n_d + n_s + n_a} EDirect calls)..."
        )
        disease_pmids = self._fetch_all_pmids(
            disease_df, "source_mesh", "source_query_name"
        )
        symptom_pmids = self._fetch_all_pmids(
            symptom_df, "target_mesh", "target_query_name"
        )
        anatomy_pmids = self._fetch_all_pmids(
            anatomy_df, "target_mesh", "target_query_name"
        )

        # ---- Phase 2: compute statistics in memory -----------------------
        logger.info("Phase 2: computing co-occurrence statistics in memory...")
        result = {}

        # Disease-Symptom
        ds_corpus = len(_pmid_union(disease_pmids) & _pmid_union(symptom_pmids))
        logger.info(f"  D-S corpus: {ds_corpus:,} PMIDs")
        ds_df = self._compute_stats(
            disease_df, symptom_df, disease_pmids, symptom_pmids, ds_corpus
        )
        if not ds_df.empty:
            result[DS_OUTPUT] = ds_df.rename(columns={
                "source_id": "doid_code",
                "target_id": "mesh_id",
            })[["doid_code", "mesh_id", "cooccurrence", "enrichment",
                "p_fisher", "odds_ratio", "source_mesh"]].copy()

        # Disease-Anatomy
        da_corpus = len(_pmid_union(disease_pmids) & _pmid_union(anatomy_pmids))
        logger.info(f"  D-A corpus: {da_corpus:,} PMIDs")
        da_df = self._compute_stats(
            disease_df, anatomy_df, disease_pmids, anatomy_pmids, da_corpus
        )
        if not da_df.empty:
            result[DA_OUTPUT] = da_df.rename(columns={
                "source_id": "doid_code",
                "target_id": "uberon_id",
            })[["doid_code", "uberon_id", "cooccurrence", "enrichment",
                "p_fisher", "odds_ratio", "source_mesh", "target_mesh"]].copy()

        # Disease-Disease (upper triangle only; co-occurrence is symmetric)
        dd_corpus = len(_pmid_union(disease_pmids))
        logger.info(f"  D-D corpus: {dd_corpus:,} PMIDs")
        disease_as_target = disease_df.rename(columns={
            "source_id":         "target_id",
            "source_mesh":       "target_mesh",
            "source_query_name": "target_query_name",
        })
        dd_df = self._compute_stats(
            disease_df, disease_as_target,
            disease_pmids, disease_pmids, dd_corpus,
            upper_triangle=True,
        )
        if not dd_df.empty:
            result[DD_OUTPUT] = dd_df.rename(columns={
                "source_id":   "doid_code_0",
                "target_id":   "doid_code_1",
                "source_mesh": "mesh_id_0",
                "target_mesh": "mesh_id_1",
            })[["doid_code_0", "doid_code_1", "cooccurrence", "enrichment",
                "p_fisher", "odds_ratio", "mesh_id_0", "mesh_id_1"]].copy()

        # Guarantee all three output keys are present (may be empty DataFrames)
        for key, cols in [
            (DS_OUTPUT, ["doid_code", "mesh_id", "cooccurrence", "enrichment",
                         "p_fisher", "odds_ratio", "source_mesh"]),
            (DA_OUTPUT, ["doid_code", "uberon_id", "cooccurrence", "enrichment",
                         "p_fisher", "odds_ratio", "source_mesh", "target_mesh"]),
            (DD_OUTPUT, ["doid_code_0", "doid_code_1", "cooccurrence", "enrichment",
                         "p_fisher", "odds_ratio", "mesh_id_0", "mesh_id_1"]),
        ]:
            if key not in result:
                result[key] = pd.DataFrame(columns=cols)

        for df in result.values():
            df["source_database"] = "MEDLINE"

        return result

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        _stats = {
            "cooccurrence": "PMID set intersection size (|S ∩ T|)",
            "enrichment":   "Observed / expected co-occurrence ratio",
            "p_fisher":     "One-tailed Fisher's exact p-value (enrichment)",
            "odds_ratio":   "Fisher's exact odds ratio",
        }
        return {
            DS_OUTPUT: {
                "doid_code":  "Disease Ontology ID (e.g. DOID:10652)",
                "mesh_id":    "MeSH symptom descriptor ID (e.g. D000544)",
                "source_mesh": "MeSH descriptor ID of the disease used in query",
                **_stats,
            },
            DA_OUTPUT: {
                "doid_code":   "Disease Ontology ID (e.g. DOID:10652)",
                "uberon_id":   "Uberon anatomy ID (e.g. UBERON:0000955)",
                "source_mesh": "MeSH descriptor ID of the disease",
                "target_mesh": "MeSH descriptor ID of the anatomy term",
                **_stats,
            },
            DD_OUTPUT: {
                "doid_code_0": "Disease Ontology ID of the first disease",
                "doid_code_1": "Disease Ontology ID of the second disease",
                "mesh_id_0":   "MeSH descriptor ID of the first disease",
                "mesh_id_1":   "MeSH descriptor ID of the second disease",
                **_stats,
            },
        }

    # ------------------------------------------------------------------
    # EDirect helpers
    # ------------------------------------------------------------------

    def _check_edirect(self) -> bool:
        """Return True if esearch and efetch are reachable; log an error if not."""
        env_path = f"{_EDIRECT_DIR}{os.pathsep}{os.environ.get('PATH', '')}"
        missing = [t for t in ("esearch", "efetch") if shutil.which(t, path=env_path) is None]
        if missing:
            logger.error(
                f"EDirect tools not found: {missing}. "
                f"Expected at {_EDIRECT_DIR} or on system PATH. "
                "Install with: "
                "HOME=$(pwd) sh -c \"$(curl -fsSL "
                "https://ftp.ncbi.nlm.nih.gov/entrez/entrezdirect/install-edirect.sh)\""
            )
            return False
        return True

    def _fetch_pmids(self, mesh_id: str, mesh_name: str) -> FrozenSet[str]:
        """
        Fetch all PubMed IDs for a MeSH term via EDirect (esearch | efetch -format uid).

        EDirect handles internal pagination and the PubMed 9,999-record REST limit.
        Results are cached in pmids/{mesh_id}.txt.gz.
        Returns a frozenset of PMID strings; empty frozenset on failure.
        """
        cache_path = self._pmid_cache_dir / f"{mesh_id}.txt.gz"

        if cache_path.exists() and not self.force:
            with gzip.open(cache_path, "rt") as fh:
                pmids = frozenset(line.strip() for line in fh if line.strip())
            logger.info(f"Cache hit: {len(pmids):,} PMIDs for {mesh_name}")
            return pmids

        env = {**os.environ, "PATH": f"{_EDIRECT_DIR}{os.pathsep}{os.environ.get('PATH', '')}"}
        if self.api_key:
            env["NCBI_API_KEY"] = self.api_key

        query = f'"{mesh_name}"[MeSH Terms]'
        cmd = f"esearch -db pubmed -query {shlex.quote(query)} | efetch -format uid"
        try:
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, env=env, timeout=600
            )
            pmids = frozenset(line.strip() for line in proc.stdout.splitlines() if line.strip())
            if proc.returncode != 0 and not pmids:
                logger.warning(
                    f"EDirect non-zero exit for {mesh_name!r}: {proc.stderr.strip()[:200]}"
                )
                return frozenset()
            with gzip.open(cache_path, "wt") as fh:
                fh.write("\n".join(sorted(pmids)))
            logger.info(f"Fetched {len(pmids):,} PMIDs for {mesh_name} ({mesh_id})")
            return pmids
        except subprocess.TimeoutExpired:
            logger.warning(f"EDirect timed out for {mesh_name!r}")
            return frozenset()
        except Exception as exc:
            logger.warning(f"EDirect failed for {mesh_name!r}: {exc}")
            return frozenset()

    def _fetch_all_pmids(
        self,
        entity_df: pd.DataFrame,
        mesh_col: str,
        name_col: str,
    ) -> Dict[str, FrozenSet[str]]:
        """
        Fetch PMID sets for every row in entity_df.
        Deduplicates by mesh_id so each MeSH term is fetched at most once.
        Returns {mesh_id: frozenset(pmids)}.
        """
        pmid_sets: Dict[str, FrozenSet[str]] = {}
        total = len(entity_df)
        for i, (_, row) in enumerate(entity_df.iterrows(), 1):
            mesh_id   = row[mesh_col]
            mesh_name = row[name_col]
            if mesh_id not in pmid_sets:
                logger.info(f"  [{i}/{total}] {mesh_name} ({mesh_id})")
                pmid_sets[mesh_id] = self._fetch_pmids(mesh_id, mesh_name)
        return pmid_sets

    # ------------------------------------------------------------------
    # Co-occurrence statistics
    # ------------------------------------------------------------------

    def _compute_stats(
        self,
        source_df: pd.DataFrame,
        target_df: pd.DataFrame,
        source_pmids: Dict[str, FrozenSet[str]],
        target_pmids: Dict[str, FrozenSet[str]],
        corpus_size: int,
        upper_triangle: bool = False,
    ) -> pd.DataFrame:
        """
        Compute pairwise co-occurrence and Fisher's exact test for all pairs.

        Contingency table for (S, T):
          a = |S ∩ T|            b = |S| - a
          c = |T| - a            d = corpus - a - b - c

        enrichment = a / (|S| × |T| / corpus)

        upper_triangle: skip pairs where source_id >= target_id (D-D symmetry).
        Returns only pairs with cooccurrence > 0.

        Output columns: source_id, target_id, cooccurrence, enrichment,
                        p_fisher, odds_ratio, source_mesh, target_mesh.
        """
        rows = []

        for _, src in source_df.iterrows():
            src_id   = src["source_id"]
            src_mesh = src["source_mesh"]
            pmids_s  = source_pmids.get(src_mesh, frozenset())
            n_s      = len(pmids_s)
            if n_s == 0:
                continue

            for _, tgt in target_df.iterrows():
                tgt_id   = tgt["target_id"]
                tgt_mesh = tgt["target_mesh"]

                if upper_triangle and src_id >= tgt_id:
                    continue

                pmids_t = target_pmids.get(tgt_mesh, frozenset())
                n_t     = len(pmids_t)
                if n_t == 0:
                    continue

                a = len(pmids_s & pmids_t)
                if a == 0:
                    continue

                b = n_s - a
                c = n_t - a
                d = max(corpus_size - a - b - c, 0)

                expected = (n_s * n_t / corpus_size) if corpus_size > 0 else 0
                enrichment = (a / expected) if expected > 0 else float("inf")

                odds_ratio, p_fisher = fisher_exact(
                    [[a, b], [c, d]], alternative="greater"
                )
                rows.append({
                    "source_id":   src_id,
                    "target_id":   tgt_id,
                    "cooccurrence": a,
                    "enrichment":  round(float(enrichment), 4),
                    "p_fisher":    float(p_fisher),
                    "odds_ratio":  round(float(odds_ratio), 6),
                    "source_mesh": src_mesh,
                    "target_mesh": tgt_mesh,
                })

        if not rows:
            return pd.DataFrame(columns=[
                "source_id", "target_id", "cooccurrence", "enrichment",
                "p_fisher", "odds_ratio", "source_mesh", "target_mesh",
            ])
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # MeSH name lookup
    # ------------------------------------------------------------------

    def _load_mesh_names(self) -> Dict[str, str]:
        """
        Parse MeSH descriptor XML to build a {mesh_id: mesh_name} mapping.
        Tries MESH_YEAR and the two preceding years.
        """
        xml_path = None
        for year in [MESH_YEAR, MESH_YEAR - 1, MESH_YEAR - 2]:
            candidate = self._mesh_dir / f"desc{year}.xml"
            if candidate.exists() and candidate.stat().st_size > 1_000_000:
                xml_path = candidate
                break

        if xml_path is None:
            logger.error(
                f"No valid MeSH XML found in {self._mesh_dir}; "
                "run MeSHParser first"
            )
            return {}

        logger.info(f"Parsing MeSH names from {xml_path}...")
        names: Dict[str, str] = {}
        context = etree.iterparse(str(xml_path), events=("end",),
                                  tag="DescriptorRecord")
        for _, elem in context:
            ui   = elem.findtext(".//DescriptorUI")
            name = elem.findtext(".//DescriptorName/String")
            if ui and name:
                names[ui] = name
            elem.clear()

        logger.info(f"Loaded {len(names)} MeSH descriptor names")
        return names

    # ------------------------------------------------------------------
    # Entity list builders
    # ------------------------------------------------------------------

    def _load_disease_list(self, mesh_names: Dict[str, str]) -> Optional[pd.DataFrame]:
        """
        Build disease list from slim_terms.tsv (DOIDs) and doid.obo (MeSH xrefs).

        Uses the union of all slim_terms diseases and any explicit doid_ids from
        the project disease scope (config/project.yaml), so scope diseases are
        always present even if missing from the slim.

        Columns: source_id, source_mesh, source_query_name.
        One row per disease; first valid MeSH xref wins.
        """
        slim_path = self._processed_dir / "disease_ontology" / "slim_terms.tsv"
        obo_path  = self.data_dir / "diseaseontology" / "doid.obo"

        for path in (slim_path, obo_path):
            if not path.exists():
                logger.error(f"Required file not found: {path}")
                return None

        slim_df    = pd.read_csv(slim_path, sep="\t").rename(columns={"doid": "source_id"})
        slim_doids = set(slim_df["source_id"].dropna().tolist())

        # Union with explicit scope DOIDs so disease-scope entries are
        # always queried even if absent from the slim
        scope_doids = set(get_disease_scope().get("doid_ids", []))
        extra       = scope_doids - slim_doids
        all_doids   = slim_doids | scope_doids
        logger.info(
            f"Disease list: {len(slim_doids)} slim_terms"
            + (f" + {len(extra)} scope-only" if extra else "")
            + f" = {len(all_doids)} total"
        )

        logger.info(f"Parsing {obo_path} for MeSH xrefs ({len(all_doids)} diseases)...")
        rows = []
        current_id = None

        with open(obo_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if line.startswith("id: DOID:"):
                    current_id = line[4:].strip()
                elif line.startswith("xref: MESH:") and current_id in all_doids:
                    mesh_id = line[11:].strip()
                    if mesh_id not in EXCLUDED_MESH_IDS and mesh_id in mesh_names:
                        rows.append({
                            "source_id":         current_id,
                            "source_mesh":       mesh_id,
                            "source_query_name": mesh_names[mesh_id],
                        })
                elif not line or line.startswith("["):
                    current_id = None

        if not rows:
            logger.warning("No MeSH xrefs found for diseases in slim_terms.tsv")
            return pd.DataFrame(
                columns=["source_id", "source_mesh", "source_query_name"]
            )

        df = pd.DataFrame(rows)
        df = df.drop_duplicates(subset=["source_id"], keep="first")
        logger.info(f"Loaded {len(df)} disease->MeSH mappings")
        return df

    def _load_symptom_list(self) -> Optional[pd.DataFrame]:
        """
        Load symptom list from symptom_nodes.tsv.
        Columns: target_id, target_mesh, target_query_name.
        For symptoms, target_id == target_mesh (both MeSH IDs).
        """
        path = self._processed_dir / "mesh" / "symptom_nodes.tsv"
        if not path.exists():
            logger.error(f"symptom_nodes.tsv not found: {path}")
            return None
        df = pd.read_csv(path, sep="\t")
        df = df.rename(columns={
            "mesh_id":   "target_id",
            "mesh_name": "target_query_name",
        })
        df["target_mesh"] = df["target_id"]
        logger.info(f"Loaded {len(df)} symptom terms from {path}")
        return df[["target_id", "target_mesh", "target_query_name"]]

    def _load_anatomy_list(
        self, mesh_names: Dict[str, str]
    ) -> Optional[pd.DataFrame]:
        """
        Load anatomy list from uberon_nodes.tsv, keeping only rows with MeSH xrefs.
        Columns: target_id (Uberon), target_mesh, target_query_name.
        """
        path = self._processed_dir / "uberon" / "uberon_nodes.tsv"
        if not path.exists():
            logger.error(f"uberon_nodes.tsv not found: {path}")
            return None
        df = pd.read_csv(path, sep="\t")
        df = df[df["mesh_id"].notna()].copy()
        df["mesh_id"] = df["mesh_id"].str.replace("^MESH:", "", regex=True)
        df = df[df["mesh_id"].isin(mesh_names)].copy()
        df = df.drop_duplicates(subset=["uberon_id"], keep="first")
        df = df.rename(columns={
            "uberon_id": "target_id",
            "mesh_id":   "target_mesh",
        })
        df["target_query_name"] = df["target_mesh"].map(mesh_names)
        logger.info(f"Loaded {len(df)} anatomy terms with MeSH names from {path}")
        return df[["target_id", "target_mesh", "target_query_name"]]


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------

def _pmid_union(pmid_dict: Dict[str, FrozenSet[str]]) -> FrozenSet[str]:
    """Return the union of all PMID sets in the dict."""
    union: set = set()
    for s in pmid_dict.values():
        union |= s
    return frozenset(union)
