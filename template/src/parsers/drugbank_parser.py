"""
DrugBankParser: Parser for DrugBank drug data with authentication support.

DrugBank is a comprehensive database of drug information including
drug targets, interactions, and pharmacology.

Source: https://go.drugbank.com/
Full XML: https://go.drugbank.com/releases/{version}/downloads/all-full-database
Drug links: https://go.drugbank.com/releases/{version}/downloads/all-drug-links

Note: Requires free academic account for access (HTTP Basic Auth).

Download example (curl):
    curl -Lfv -o drugbank_full.zip -u username:password \
        https://go.drugbank.com/releases/latest/downloads/all-full-database

Drug nodes are only created for entries that have at least one external
cross-referenceable identifier (CAS number, ChEMBL ID, PubChem CID,
ChEBI ID, or KEGG Drug ID) so they can be matched to other knowledge sources.

Gene (drug-target) edges are only created for entries where the gene has a
standardised identifier (UniProt accession or HGNC gene symbol).
"""

import io
import logging
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
import xml.etree.ElementTree as ET

import pandas as pd
import requests

from .base_parser import BaseParser

logger = logging.getLogger(__name__)

# DrugBank XML namespace
_NS = "http://www.drugbank.ca"
_NS_MAP = {"db": _NS}

# Output table names
_DRUGS = "drugs"
_DRUG_GENE_EDGES = "drug_gene_edges"


def _tag(local: str) -> str:
    """Return a fully-qualified DrugBank XML tag."""
    return f"{{{_NS}}}{local}"


class DrugBankParser(BaseParser):
    """
    Parser for DrugBank drug data with HTTP Basic Authentication support.

    Attempts to download and parse the full DrugBank XML database for rich
    drug properties and drug-gene edges.  Falls back to the lighter
    drug-links CSV when the XML is unavailable.

    Only Drug nodes that have at least one external cross-referenceable
    identifier (CAS number, ChEMBL ID, PubChem CID, ChEBI ID, or KEGG Drug
    ID) are emitted so they can be matched to other knowledge sources.

    Constructor args (injected by main.py from databases.yaml after env-var
    resolution):
        data_dir    – base directory for raw/cached files
        source_url  – URL template for the drug-links CSV download
                      (e.g. "https://go.drugbank.com/releases/{version}/downloads/all-drug-links")
        version     – DrugBank release tag ("latest" or e.g. "5-1-14")
        username    – DrugBank account username/email
        password    – DrugBank account password
    """

    BASE_URL = "https://go.drugbank.com"

    # External identifier fields used to decide whether a drug is
    # cross-referenceable to other knowledge sources.
    _CROSSREF_FIELDS = ("cas_number", "chembl_id", "pubchem_cid", "chebi_id", "kegg_drug_id")

    def __init__(
        self,
        data_dir: str,
        source_url: Optional[str] = None,
        version: str = "latest",
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        super().__init__(data_dir)

        # Override source_name so files land in data/raw/drugbank/
        self.source_name = "drugbank"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

        self.version = version

        # Credentials — prefer constructor args, fall back to env vars
        self.username = username or os.getenv("DRUGBANK_USERNAME")
        self.password = password or os.getenv("DRUGBANK_PASSWORD")

        # Build download URLs
        ver = version  # "latest" or "5-1-14"
        self._full_db_url = f"{self.BASE_URL}/releases/{ver}/downloads/all-full-database"
        if source_url:
            self._links_url = source_url.format(version=ver)
        else:
            self._links_url = f"{self.BASE_URL}/releases/{ver}/downloads/all-drug-links"

        # HTTP session with Basic Auth if credentials are present
        self.session = requests.Session()
        if self.username and self.password:
            self.session.auth = (self.username, self.password)
            logger.info("DrugBank credentials configured (HTTP Basic Auth).")
        else:
            logger.warning(
                "No DrugBank credentials provided. "
                "Will attempt file-based parsing from cached data."
            )

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """
        Download DrugBank data.

        Priority:
          1. Full XML database (all-full-database) — richest data
          2. Drug-links CSV (all-drug-links) — basic identifiers only
          3. Existing cached files — no network required

        Returns True when at least one usable file is present.
        """
        # 1. Try full XML if credentials are available
        if self.username and self.password:
            if self._download_full_xml():
                return True
            logger.warning(
                "Full XML download failed; trying drug-links CSV fallback."
            )
            if self._download_links_csv():
                return True
        else:
            # No credentials — try drug-links CSV (may also fail without auth)
            if self._download_links_csv():
                return True

        # 2. Check for any cached file
        return self._check_cached_files()

    def _download_full_xml(self) -> bool:
        """Download the full DrugBank XML zip."""
        zip_path = self.source_dir / "drugbank_full.zip"
        xml_path = self.source_dir / "full_database.xml"

        if xml_path.exists() and not self.force:
            logger.info("Full XML already cached: %s", xml_path)
            return True
        if zip_path.exists() and not self.force:
            logger.info("Full XML zip already cached; extracting...")
            return self._extract_full_xml(zip_path, xml_path)

        logger.info("Downloading full DrugBank XML from: %s", self._full_db_url)
        try:
            resp = self.session.get(
                self._full_db_url,
                stream=True,
                timeout=600,
                allow_redirects=True,
            )
            resp.raise_for_status()

            with open(zip_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
            logger.info("Downloaded full XML zip to %s", zip_path)
            return self._extract_full_xml(zip_path, xml_path)

        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            logger.error("HTTP %s downloading full XML: %s", code, exc)
            return False
        except Exception as exc:
            logger.error("Error downloading full XML: %s", exc)
            return False

    def _extract_full_xml(self, zip_path: Path, xml_path: Path) -> bool:
        """Extract the XML from the zip archive."""
        try:
            with zipfile.ZipFile(zip_path) as zf:
                xml_members = [
                    m for m in zf.namelist()
                    if m.endswith(".xml") and "full" in m.lower()
                ]
                if not xml_members:
                    # Accept any XML in the archive
                    xml_members = [m for m in zf.namelist() if m.endswith(".xml")]
                if not xml_members:
                    logger.error("No XML file found in zip: %s", zf.namelist())
                    return False
                member = xml_members[0]
                logger.info("Extracting %s from zip...", member)
                with zf.open(member) as src, open(xml_path, "wb") as dst:
                    dst.write(src.read())
            logger.info("Extracted XML to %s", xml_path)
            return True
        except Exception as exc:
            logger.error("Failed to extract XML: %s", exc)
            return False

    def _download_links_csv(self) -> bool:
        """Download the drug-links CSV (basic identifiers)."""
        csv_path = self.source_dir / "drugs.csv"
        if csv_path.exists() and not self.force:
            logger.info("Drug-links CSV already cached: %s", csv_path)
            return True

        logger.info("Downloading drug-links CSV from: %s", self._links_url)
        try:
            resp = self.session.get(
                self._links_url,
                stream=True,
                timeout=120,
                allow_redirects=True,
            )
            resp.raise_for_status()

            content_type = resp.headers.get("content-type", "")
            if "zip" in content_type or self._links_url.endswith(".zip"):
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_members = [m for m in zf.namelist() if m.endswith(".csv")]
                    if not csv_members:
                        logger.error("No CSV in zip: %s", zf.namelist())
                        return False
                    with zf.open(csv_members[0]) as src:
                        csv_path.write_bytes(src.read())
            else:
                csv_path.write_bytes(resp.content)

            logger.info("Drug-links CSV saved to %s", csv_path)
            return True

        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else "?"
            logger.error("HTTP %s downloading drug-links CSV: %s", code, exc)
            return False
        except Exception as exc:
            logger.error("Error downloading drug-links CSV: %s", exc)
            return False

    def _check_cached_files(self) -> bool:
        """Return True if any usable cached file exists."""
        xml_path = self.source_dir / "full_database.xml"
        csv_path = self.source_dir / "drugs.csv"
        if xml_path.exists():
            logger.info("Using cached full XML: %s", xml_path)
            return True
        if csv_path.exists():
            logger.info("Using cached drug-links CSV: %s", csv_path)
            return True
        logger.error(
            "No DrugBank data available. Provide credentials or place "
            "drugs.csv / full_database.xml in %s", self.source_dir
        )
        return False

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse DrugBank data into DataFrames.

        Returns a dict with one or more of:
          - "drugs"            : Drug node properties
          - "drug_gene_edges"  : Drug-Gene binding edges (chemicalBindsGene)
        """
        xml_path = self.source_dir / "full_database.xml"
        csv_path = self.source_dir / "drugs.csv"

        if xml_path.exists():
            logger.info("Parsing full DrugBank XML: %s", xml_path)
            result = self._parse_full_xml(xml_path)
        elif csv_path.exists():
            logger.info("Parsing drug-links CSV (fallback): %s", csv_path)
            result = self._parse_links_csv(csv_path)
        else:
            logger.error("No DrugBank data file found to parse.")
            return {}

        return self._post_process(result)

    def _post_process(self, result: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Post-process all parsed DataFrames:
          1. Strip embedded newlines/carriage-returns from all string columns.
          2. Rename 'name' → 'drug_name' in the drugs DataFrame.
        """
        if _DRUGS not in result:
            return result

        # --- Strip embedded newlines from all string columns ---
        for table_name, df in result.items():
            for col in df.select_dtypes(include="object").columns:
                df[col] = df[col].str.replace(r"[\r\n]+", " ", regex=True)
            result[table_name] = df

        # --- Rename 'name' → 'drug_name' in drugs ---
        if "name" in result[_DRUGS].columns:
            result[_DRUGS] = result[_DRUGS].rename(columns={"name": "drug_name"})
            logger.info("Renamed 'name' → 'drug_name' in drugs DataFrame.")

        return result

    # ------------------------------------------------------------------
    # Full XML parsing
    # ------------------------------------------------------------------

    def _parse_full_xml(self, xml_path: Path) -> Dict[str, pd.DataFrame]:
        """Parse the full DrugBank XML and return DataFrames."""
        drug_rows: List[Dict] = []
        gene_edge_rows: List[Dict] = []

        logger.info("Streaming DrugBank XML (this may take a while)...")
        context = ET.iterparse(str(xml_path), events=("end",))

        drug_count = 0
        for event, elem in context:
            if elem.tag != _tag("drug"):
                continue
            # Skip nested <drug> elements (e.g. metabolites inside a drug)
            # Top-level drugs have a 'type' attribute (small molecule / biotech)
            parent_tag = elem.get("type")
            if parent_tag is None:
                elem.clear()
                continue

            drug_row = self._extract_drug_row(elem)
            if drug_row:
                drug_rows.append(drug_row)
                db_id = drug_row["drugbank_id"]
                # Gene edges from targets / enzymes / carriers / transporters
                for edge in self._extract_gene_edges(elem, db_id):
                    gene_edge_rows.append(edge)
                drug_count += 1

            elem.clear()

        logger.info("Parsed %d drug entries from XML.", drug_count)

        result: Dict[str, pd.DataFrame] = {}

        if drug_rows:
            drugs_df = pd.DataFrame(drug_rows)
            drugs_df = drugs_df.drop_duplicates(subset=["drugbank_id"])

            # Filter: only keep drugs with at least one external cross-reference
            # identifier so they can be matched to other knowledge sources.
            crossref_mask = drugs_df[list(self._CROSSREF_FIELDS)].apply(
                lambda row: row.astype(str).str.strip().ne("").any(), axis=1
            )
            before = len(drugs_df)
            drugs_df = drugs_df[crossref_mask].copy()
            logger.info(
                "Drug cross-reference filter: %d → %d drugs "
                "(kept entries with CAS/ChEMBL/PubChem/ChEBI/KEGG).",
                before, len(drugs_df),
            )

            result[_DRUGS] = drugs_df
            logger.info("Drug nodes: %d", len(drugs_df))

        if gene_edge_rows:
            gene_df = pd.DataFrame(gene_edge_rows)
            # Only keep edges where the gene has a standardised identifier
            has_id = (
                gene_df["uniprot_id"].str.strip().ne("")
                | gene_df["gene_symbol"].str.strip().ne("")
            )
            before = len(gene_df)
            gene_df = gene_df[has_id].drop_duplicates().copy()
            logger.info(
                "Gene edge filter: %d → %d edges "
                "(kept entries with UniProt ID or gene symbol).",
                before, len(gene_df),
            )
            result[_DRUG_GENE_EDGES] = gene_df
            logger.info("Drug-Gene edges: %d", len(gene_df))

        return result

    def _txt(self, elem: ET.Element, path: str) -> str:
        """Extract text from a child element; return empty string if absent."""
        child = elem.find(path, _NS_MAP)
        if child is not None and child.text:
            return child.text.strip()
        return ""

    def _extract_drug_row(self, drug_elem: ET.Element) -> Optional[Dict]:
        """Extract drug node properties from a <drug> XML element."""
        # Primary DrugBank ID
        db_id = ""
        for id_elem in drug_elem.findall("db:drugbank-id", _NS_MAP):
            if id_elem.get("primary") == "true":
                db_id = (id_elem.text or "").strip()
                break
        if not db_id:
            return None

        # Drug type from attribute
        drug_type = drug_elem.get("type", "")

        # Groups (approved, investigational, etc.)
        groups = "; ".join(
            g.text.strip()
            for g in drug_elem.findall("db:groups/db:group", _NS_MAP)
            if g.text
        )

        # Categories
        categories = "; ".join(
            c.text.strip()
            for c in drug_elem.findall("db:categories/db:category/db:category", _NS_MAP)
            if c.text
        )

        # Calculated properties (molecular formula, weight, SMILES, InChI)
        mol_formula = ""
        mol_weight = ""
        smiles = ""
        inchi = ""
        inchikey = ""
        for prop in drug_elem.findall(
            "db:calculated-properties/db:property", _NS_MAP
        ):
            kind = self._txt(prop, "db:kind")
            value = self._txt(prop, "db:value")
            if kind == "Molecular Formula":
                mol_formula = value
            elif kind == "Molecular Weight":
                mol_weight = value
            elif kind == "SMILES":
                smiles = value
            elif kind == "InChI":
                inchi = value
            elif kind == "InChIKey":
                inchikey = value

        # External identifiers for cross-referencing
        ext_ids: Dict[str, str] = {}
        for ext in drug_elem.findall(
            "db:external-identifiers/db:external-identifier", _NS_MAP
        ):
            resource = self._txt(ext, "db:resource")
            identifier = self._txt(ext, "db:identifier")
            if resource and identifier:
                ext_ids[resource] = identifier

        return {
            "drugbank_id": db_id,
            "name": self._txt(drug_elem, "db:name"),
            # --- data properties (free text — NOT separate RDF classes) ---
            "drugDescription": self._txt(drug_elem, "db:description"),
            "drugType": drug_type,
            "drugGroups": groups,
            "drugCategories": categories,
            "drugIndication": self._txt(drug_elem, "db:indication"),
            "drugPharmacology": self._txt(drug_elem, "db:pharmacodynamics"),
            "drugMechanism": self._txt(drug_elem, "db:mechanism-of-action"),
            "drugToxicity": self._txt(drug_elem, "db:toxicity"),
            "drugHalfLife": self._txt(drug_elem, "db:half-life"),
            "drugState": self._txt(drug_elem, "db:state"),
            # --- structural identifiers ---
            "molecularFormula": mol_formula,
            "molecularWeight": mol_weight,
            "smiles": smiles,
            "inchi": inchi,
            "inchikey": inchikey,
            # --- cross-reference identifiers ---
            "cas_number": self._txt(drug_elem, "db:cas-number"),
            "kegg_drug_id": ext_ids.get("KEGG Drug", ""),
            "pubchem_cid": ext_ids.get("PubChem Compound", ""),
            "chembl_id": ext_ids.get("ChEMBL", ""),
            "chebi_id": ext_ids.get("ChEBI", ""),
            "source_database": "DrugBank",
        }

    def _extract_gene_edges(
        self, drug_elem: ET.Element, db_id: str
    ) -> List[Dict]:
        """
        Extract Drug-Gene edges from targets, enzymes, carriers, transporters.

        Only edges where the gene has a UniProt accession or HGNC gene symbol
        are returned, ensuring cross-referenceability to other sources.

        Returns list of dicts with keys:
          drugbank_id, gene_symbol, uniprot_id, interaction_type
        """
        edges: List[Dict] = []
        interaction_sections = [
            ("db:targets/db:target", "target"),
            ("db:enzymes/db:enzyme", "enzyme"),
            ("db:carriers/db:carrier", "carrier"),
            ("db:transporters/db:transporter", "transporter"),
        ]
        for section_path, itype in interaction_sections:
            for actor in drug_elem.findall(section_path, _NS_MAP):
                gene_symbol = ""
                uniprot_id = ""
                poly = actor.find("db:polypeptide", _NS_MAP)
                if poly is not None:
                    uniprot_id = poly.get("id", "")
                    gene_symbol = self._txt(poly, "db:gene-name")
                # Skip entries without any standardised gene identifier
                if not gene_symbol and not uniprot_id:
                    continue
                edges.append(
                    {
                        "drugbank_id": db_id,
                        "gene_symbol": gene_symbol,
                        "uniprot_id": uniprot_id,
                        "interaction_type": itype,
                        "source_database": "DrugBank",
                    }
                )
        return edges

    # ------------------------------------------------------------------
    # Drug-links CSV parsing (fallback)
    # ------------------------------------------------------------------

    def _parse_links_csv(self, csv_path: Path) -> Dict[str, pd.DataFrame]:
        """
        Parse the drug-links CSV (basic identifiers, no rich properties).

        Column mapping follows the DrugBank all-drug-links export format.
        Only drugs with at least one external cross-reference identifier are
        kept so they can be matched to other knowledge sources.
        """
        try:
            df = pd.read_csv(csv_path, dtype=str)
        except Exception as exc:
            logger.error("Failed to read drug-links CSV: %s", exc)
            return {}

        column_mapping = {
            "DrugBank ID": "drugbank_id",
            "Name": "name",
            "CAS Number": "cas_number",
            "Drug Type": "drugType",
            "KEGG Compound ID": "kegg_compound_id",
            "KEGG Drug ID": "kegg_drug_id",
            "PubChem Compound ID": "pubchem_cid",
            "PubChem Substance ID": "pubchem_sid",
            "ChEBI ID": "chebi_id",
            "PharmGKB ID": "pharmgkb_id",
            "UniProt ID": "uniprot_id",
            "UniProt Title": "uniprot_title",
            "GenBank ID": "genbank_id",
            "ChEMBL ID": "chembl_id",
        }
        existing = {k: v for k, v in column_mapping.items() if k in df.columns}
        df = df.rename(columns=existing)

        # Fill in columns that only come from the full XML
        xml_only_cols = [
            "drugDescription", "drugGroups", "drugCategories",
            "drugIndication", "drugPharmacology", "drugMechanism",
            "drugToxicity", "drugHalfLife", "drugState",
            "molecularFormula", "molecularWeight", "smiles", "inchi", "inchikey",
        ]
        for col in xml_only_cols:
            if col not in df.columns:
                df[col] = ""

        # Ensure cross-reference columns exist before filtering
        for field in self._CROSSREF_FIELDS:
            if field not in df.columns:
                df[field] = ""

        df["source_database"] = "DrugBank"
        df = df.drop_duplicates(subset=["drugbank_id"])

        # Filter: only keep drugs with at least one external cross-reference
        crossref_mask = df[list(self._CROSSREF_FIELDS)].apply(
            lambda row: row.fillna("").astype(str).str.strip().ne("").any(), axis=1
        )
        before = len(df)
        df = df[crossref_mask].copy()
        logger.info(
            "Drug cross-reference filter: %d → %d drugs "
            "(kept entries with CAS/ChEMBL/PubChem/ChEBI/KEGG).",
            before, len(df),
        )

        logger.info("Parsed %d drugs from drug-links CSV.", len(df))
        return {_DRUGS: df}

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """Return the schema for DrugBank output DataFrames."""
        return {
            _DRUGS: {
                "drugbank_id": "DrugBank primary identifier (DB#####)",
                "drug_name": "Drug common name",
                "drugDescription": "Drug description text (data property)",
                "drugType": "Drug type (small molecule / biotech)",
                "drugGroups": "Approval groups (approved, investigational, …)",
                "drugCategories": "ATC / pharmacological categories",
                "drugIndication": "Therapeutic indication text (data property)",
                "drugPharmacology": "Pharmacodynamics text (data property)",
                "drugMechanism": "Mechanism of action text (data property)",
                "drugToxicity": "Toxicity information (data property)",
                "drugHalfLife": "Half-life value",
                "drugState": "Physical state (solid / liquid / gas)",
                "molecularFormula": "Molecular formula",
                "molecularWeight": "Molecular weight (Da)",
                "smiles": "Canonical SMILES string",
                "inchi": "InChI string",
                "inchikey": "InChIKey",
                "cas_number": "CAS Registry Number",
                "kegg_drug_id": "KEGG Drug identifier",
                "pubchem_cid": "PubChem Compound ID",
                "chembl_id": "ChEMBL identifier",
                "chebi_id": "ChEBI identifier",
                "source_database": "Source database name",
            },
            _DRUG_GENE_EDGES: {
                "drugbank_id": "DrugBank ID of the drug",
                "gene_symbol": "HGNC gene symbol of the target/enzyme/carrier/transporter",
                "uniprot_id": "UniProt accession of the polypeptide",
                "interaction_type": "Type of interaction (target/enzyme/carrier/transporter)",
                "source_database": "Source database name",
            },
        }
