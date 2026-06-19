"""
CTD (Comparative Toxicogenomics Database) Exposure Parser.

Downloads two CTD bulk files and extracts exposure-centric nodes and edges
directly from curated epidemiological exposure-event associations:

  exposure_nodes.tsv                              — Exposure (stressor) nodes
  exposure_linked_to_disease.tsv                 — exposureLinkedToDisease edges
  exposure_interacts_with_gene.tsv               — exposureInteractsWithGene edges
  exposure_interacts_with_biological_process.tsv — exposureInteractsWithBiologicalProcess edges
  exposure_interacts_with_molecular_function.tsv — exposureInteractsWithMolecularFunction edges
  exposure_interacts_with_cellular_component.tsv — exposureInteractsWithCellularComponent edges

Data Sources:
  http://ctdbase.org/reports/CTD_exposure_studies.tsv.gz  (exposure-event associations)
  http://ctdbase.org/reports/CTD_chem_go_enriched.tsv.gz (GO term namespace lookup)

File format (CTD_exposure_studies.tsv.gz, 10 tab-separated columns):
  [0] PubmedID
  [1] StudyFactors        (pipe-separated covariates, e.g. "age|sex")
  [2] Stressors           (pipe-sep "Name^MeSH-ID^MESH" entries)
  [3] Receptors           (pipe-sep "Name^type^ID^Source^notes"; type="gene" entries carry NCBI Gene IDs)
  [4] Countries
  [5] ExposureMedium
  [6] ExposureMarkers     (pipe-sep "Name^MeSH-ID^MESH" entries for measured chemicals)
  [7] Diseases            (pipe-sep "Name^MeSH-ID^MESH" disease outcomes)
  [8] GOTerms             (pipe-sep "Name^GO:ID^GO" GO term associations)
  [9] StudyNotes
"""

import logging
from typing import Dict, List, Set

import pandas as pd

from .base_parser import BaseParser

logger = logging.getLogger(__name__)


class CTDExposureParser(BaseParser):
    """
    Parser for CTD exposure studies.

    All edges are derived from curated exposure-event associations in
    CTD_exposure_studies.tsv.gz.  GO term namespaces are resolved using
    CTD_chem_go_enriched.tsv.gz.  No credentials required.
    """

    EXPOSURE_URL = "http://ctdbase.org/reports/CTD_exposure_studies.tsv.gz"
    CHEM_GO_URL = "http://ctdbase.org/reports/CTD_chem_go_enriched.tsv.gz"

    _EXPOSURE_FILENAME = "CTD_exposure_studies.tsv.gz"
    _CHEM_GO_FILENAME = "CTD_chem_go_enriched.tsv.gz"

    _EXPOSURE_COLS = [
        "PubmedID", "StudyFactors", "Stressors", "Receptors", "Countries",
        "ExposureMedium", "ExposureMarkers", "Diseases", "GOTerms", "StudyNotes",
    ]
    _CHEM_GO_COLS = [
        "ChemicalName", "ChemicalID", "CasRN", "Ontology", "GOTermName",
        "GOTermID", "HighestGOLevel", "PValue", "CorrectedPValue",
        "TargetMatchQty", "TargetTotalQty", "BackgroundMatchQty", "BackgroundTotalQty",
    ]

    # Maps CTD Ontology labels to output table name suffixes
    _GO_NAMESPACES = {
        "Biological Process": "biological_process",
        "Molecular Function": "molecular_function",
        "Cellular Component": "cellular_component",
    }

    def __init__(self, data_dir: str):
        super().__init__(data_dir)
        self.source_name = "ctd"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """Download CTD exposure studies and chem-GO enriched files."""
        logger.info("Downloading CTD exposure studies …")
        if not self.download_file(self.EXPOSURE_URL, self._EXPOSURE_FILENAME):
            logger.error("Failed to download CTD exposure studies.")
            return False

        logger.info("Downloading CTD chem-GO enriched (namespace lookup) …")
        if not self.download_file(self.CHEM_GO_URL, self._CHEM_GO_FILENAME):
            logger.warning("CTD chem-GO file unavailable; GO edges will not be split by namespace.")

        return True

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse CTD exposure-event associations.

        Returns a dict with up to six DataFrames:
          - exposure_nodes
          - exposure_linked_to_disease
          - exposure_interacts_with_gene
          - exposure_interacts_with_biological_process
          - exposure_interacts_with_molecular_function
          - exposure_interacts_with_cellular_component
        """
        df = self._load_exposure_studies()
        if df is None:
            return {}

        go_namespace = self._load_go_namespace()

        result: Dict[str, pd.DataFrame] = {}

        exposure_nodes = self._build_exposure_nodes(df)
        if not exposure_nodes.empty:
            result["exposure_nodes"] = exposure_nodes

        disease_edges = self._build_disease_edges(df)
        if not disease_edges.empty:
            result["exposure_linked_to_disease"] = disease_edges

        gene_edges = self._build_gene_edges(df)
        if not gene_edges.empty:
            result["exposure_interacts_with_gene"] = gene_edges

        go_edges = self._build_go_edges(df, go_namespace)
        for namespace_slug, edges in go_edges.items():
            result[f"exposure_interacts_with_{namespace_slug}"] = edges

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_exposure_studies(self) -> pd.DataFrame:
        """Load CTD_exposure_studies.tsv.gz into a DataFrame."""
        path = self.source_dir / self._EXPOSURE_FILENAME
        if not path.exists():
            logger.error("CTD exposure studies file not found: %s", path)
            return None

        logger.info("Parsing CTD exposure studies from %s …", path)
        try:
            df = pd.read_csv(
                path,
                sep="\t",
                compression="gzip",
                comment="#",
                header=None,
                names=self._EXPOSURE_COLS,
                low_memory=False,
                dtype=str,
            )
        except Exception as exc:
            logger.exception("Failed to read CTD exposure studies: %s", exc)
            return None

        logger.info("Loaded %d exposure study rows.", len(df))
        return df

    def _load_go_namespace(self) -> Dict[str, str]:
        """
        Load CTD_chem_go_enriched.tsv.gz and return a {GOTermID: namespace_slug} mapping.
        Returns an empty dict if the file is missing.
        """
        path = self.source_dir / self._CHEM_GO_FILENAME
        if not path.exists():
            logger.warning("CTD chem-GO file not found at %s; GO edges will be skipped.", path)
            return {}

        try:
            df = pd.read_csv(
                path,
                sep="\t",
                compression="gzip",
                comment="#",
                header=None,
                names=self._CHEM_GO_COLS,
                usecols=["Ontology", "GOTermID"],
                low_memory=False,
                dtype=str,
            )
        except Exception as exc:
            logger.exception("Failed to read CTD chem-GO file: %s", exc)
            return {}

        df = df.dropna(subset=["GOTermID", "Ontology"])
        df = df[df["Ontology"].isin(self._GO_NAMESPACES)]
        df = df.drop_duplicates(subset=["GOTermID"])
        mapping = {
            row["GOTermID"].strip(): self._GO_NAMESPACES[row["Ontology"].strip()]
            for _, row in df.iterrows()
        }
        logger.info("Loaded GO namespace mapping for %d terms.", len(mapping))
        return mapping

    @staticmethod
    def _parse_entries(raw: str) -> List[List[str]]:
        """
        Split a pipe-separated, caret-delimited CTD field into a list of part arrays.

        Example input: "Arsenic^D001151^MESH|Cadmium^D002104^MESH"
        Returns: [["Arsenic","D001151","MESH"], ["Cadmium","D002104","MESH"]]
        """
        if not raw or pd.isna(raw):
            return []
        return [item.split("^") for item in str(raw).split("|") if item.strip()]

    @staticmethod
    def _normalize_mesh_id(raw_id: str) -> str:
        """Strip whitespace from a CTD ID; return empty string if blank."""
        return raw_id.strip()

    def _expand_stressor_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Return a DataFrame with one row per (pubmedId, xrefMeSH, commonName)
        by exploding the pipe-separated Stressors column.
        """
        records = []
        for _, row in df.iterrows():
            pubmed_id = str(row["PubmedID"]).strip()
            for parts in self._parse_entries(row["Stressors"]):
                if len(parts) < 2 or not parts[1].strip():
                    continue
                xref_mesh = self._normalize_mesh_id(parts[1])
                if xref_mesh:
                    records.append({
                        "pubmedId": pubmed_id,
                        "xrefMeSH": xref_mesh,
                        "commonName": parts[0].strip(),
                        "_raw": row,
                    })
        return pd.DataFrame(records)

    def _build_exposure_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract unique Exposure (stressor) nodes."""
        records = []
        for _, row in df.iterrows():
            for parts in self._parse_entries(row["Stressors"]):
                if len(parts) < 2 or not parts[1].strip():
                    continue
                xref_mesh = self._normalize_mesh_id(parts[1])
                if xref_mesh:
                    records.append({"xrefMeSH": xref_mesh, "commonName": parts[0].strip()})

        if not records:
            return pd.DataFrame()

        nodes = (
            pd.DataFrame(records)
            .drop_duplicates(subset=["xrefMeSH"])
            .reset_index(drop=True)
        )
        nodes["source_database"] = "CTD"
        logger.info("Exposure nodes: %d", len(nodes))
        return nodes

    def _build_disease_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build exposureLinkedToDisease edges from col [7] (Diseases).
        Each row's stressors × diseases are cross-joined.
        """
        records = []
        for _, row in df.iterrows():
            pubmed_id = str(row["PubmedID"]).strip()
            stressors = [
                self._normalize_mesh_id(p[1])
                for p in self._parse_entries(row["Stressors"])
                if len(p) >= 2 and self._normalize_mesh_id(p[1])
            ]
            diseases = [
                self._normalize_mesh_id(p[1])
                for p in self._parse_entries(row["Diseases"])
                if len(p) >= 2 and self._normalize_mesh_id(p[1])
            ]
            for xref_mesh in stressors:
                for disease_id in diseases:
                    records.append({
                        "xrefMeSH": xref_mesh,
                        "disease_id": disease_id,
                        "pubmedId": pubmed_id,
                    })

        if not records:
            logger.warning("No disease-linked exposure rows found.")
            return pd.DataFrame()

        edges = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)
        edges["source_database"] = "CTD"
        logger.info("exposureLinkedToDisease edges: %d", len(edges))
        return edges

    def _build_gene_edges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build exposureInteractsWithGene edges from gene-type entries in col [3] (Receptors).
        Receptor entries with type 'gene' carry NCBI Gene IDs at parts[2].
        """
        records = []
        for _, row in df.iterrows():
            pubmed_id = str(row["PubmedID"]).strip()
            stressors = [
                self._normalize_mesh_id(p[1])
                for p in self._parse_entries(row["Stressors"])
                if len(p) >= 2 and self._normalize_mesh_id(p[1])
            ]
            for parts in self._parse_entries(row["Receptors"]):
                # Format: Name^type^ID^Source^notes
                if len(parts) < 3:
                    continue
                if parts[1].strip().lower() != "gene":
                    continue
                ncbi_gene_id = parts[2].strip()
                if not ncbi_gene_id:
                    continue
                for xref_mesh in stressors:
                    records.append({
                        "xrefMeSH": xref_mesh,
                        "xrefNcbiGene": ncbi_gene_id,
                        "geneSymbol": parts[0].strip(),
                        "pubmedId": pubmed_id,
                    })

        if not records:
            logger.warning("No gene-receptor exposure rows found.")
            return pd.DataFrame()

        edges = pd.DataFrame(records).drop_duplicates().reset_index(drop=True)
        edges["source_database"] = "CTD"
        logger.info("exposureInteractsWithGene edges: %d", len(edges))
        return edges

    def _build_go_edges(
        self, df: pd.DataFrame, go_namespace: Dict[str, str]
    ) -> Dict[str, pd.DataFrame]:
        """
        Build GO-term edges from col [8] (GOTerms), split by namespace.

        GO term IDs are classified via the go_namespace lookup.
        Returns a dict keyed by namespace slug (biological_process, etc.).
        """
        if not go_namespace:
            logger.warning("GO namespace lookup is empty; skipping GO edges.")
            return {}

        records = []
        for _, row in df.iterrows():
            pubmed_id = str(row["PubmedID"]).strip()
            stressors = [
                self._normalize_mesh_id(p[1])
                for p in self._parse_entries(row["Stressors"])
                if len(p) >= 2 and self._normalize_mesh_id(p[1])
            ]
            for parts in self._parse_entries(row["GOTerms"]):
                # Format: Name^GO:ID^GO
                if len(parts) < 2:
                    continue
                go_id = parts[1].strip()
                if not go_id.startswith("GO:"):
                    continue
                namespace_slug = go_namespace.get(go_id)
                if not namespace_slug:
                    continue
                for xref_mesh in stressors:
                    records.append({
                        "xrefMeSH": xref_mesh,
                        "xrefGeneOntology": go_id,
                        "namespace": namespace_slug,
                        "pubmedId": pubmed_id,
                    })

        if not records:
            logger.warning("No GO-term exposure associations found.")
            return {}

        all_go = pd.DataFrame(records)
        result = {}
        for namespace_slug in self._GO_NAMESPACES.values():
            subset = (
                all_go[all_go["namespace"] == namespace_slug]
                .drop(columns=["namespace"])
                .drop_duplicates()
                .reset_index(drop=True)
            )
            if subset.empty:
                continue
            subset["source_database"] = "CTD"
            label = {v: k for k, v in self._GO_NAMESPACES.items()}[namespace_slug]
            logger.info("exposureInteractsWith%s edges: %d",
                        label.replace(" ", ""), len(subset))
            result[namespace_slug] = subset

        return result

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return the column schema for each output table."""
        return {
            "exposure_nodes": {
                "xrefMeSH": "CTD stressor MeSH ID (e.g. D007854 or C004762)",
                "commonName": "Name of the stressor/exposure",
                "source_database": "CTD",
            },
            "exposure_linked_to_disease": {
                "xrefMeSH": "Source stressor MeSH ID",
                "disease_id": "Target disease MeSH ID",
                "pubmedId": "PubMed ID of the supporting study",
                "source_database": "CTD",
            },
            "exposure_interacts_with_gene": {
                "xrefMeSH": "Source stressor MeSH ID",
                "xrefNcbiGene": "Target NCBI Gene ID",
                "geneSymbol": "Gene symbol",
                "pubmedId": "PubMed ID of the supporting study",
                "source_database": "CTD",
            },
            "exposure_interacts_with_biological_process": {
                "xrefMeSH": "Source stressor MeSH ID",
                "xrefGeneOntology": "Target GO term ID (GO:XXXXXXX)",
                "pubmedId": "PubMed ID of the supporting study",
                "source_database": "CTD",
            },
            "exposure_interacts_with_molecular_function": {
                "xrefMeSH": "Source stressor MeSH ID",
                "xrefGeneOntology": "Target GO term ID (GO:XXXXXXX)",
                "pubmedId": "PubMed ID of the supporting study",
                "source_database": "CTD",
            },
            "exposure_interacts_with_cellular_component": {
                "xrefMeSH": "Source stressor MeSH ID",
                "xrefGeneOntology": "Target GO term ID (GO:XXXXXXX)",
                "pubmedId": "PubMed ID of the supporting study",
                "source_database": "CTD",
            },
        }
