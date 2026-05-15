"""
DrugCentral Parser for the knowledge graph.

Connects to a PostgreSQL instance with the DrugCentral dump loaded and
produces five clean TSV files in data/processed/drugcentral/:

  drugs.tsv                 — Drug nodes with cross-reference identifiers
  pharmacologic_classes.tsv — Pharmacologic class nodes
  drug_treats_disease.tsv   — Drug→Disease indication edges (scoped to project diseases)
  drug_in_class.tsv         — Drug→PharmacologicClass membership edges
  chemical_causes_effect.tsv— Drug→ChemicalEffect FAERS disproportionality edges

Public read-only instance (no local install required):
  host=unmtid-dbs.net  port=5433  dbname=drugcentral
  user=drugman  password=dosage   (set DC_USER / DC_PASSWORD env vars)

Alternatively, load the dump locally:
  createdb drugcentral
  gunzip -c drugcentral.sql.gz | psql drugcentral
"""

import logging
import os
from typing import Dict, List, Optional

import pandas as pd

from .base_parser import BaseParser
from config_loader import get_disease_scope

logger = logging.getLogger(__name__)

# DrugCentral SQL dump (latest public release)
_DUMP_URL = (
    "https://unmtid-shinyapps.net/download/DrugCentral/"
    "drugcentral.dump.11012023.sql.gz"
)


class DrugCentralParser(BaseParser):
    """
    Parser for the DrugCentral PostgreSQL database.

    Constructor args are injected by main.py from databases.yaml after
    env-var resolution.  The ``pg_config`` dict is passed as-is to
    psycopg2.connect(); ``llr_threshold`` is an optional additional LLR
    floor applied on top of the per-row threshold stored in the faers table.

    Args:
        data_dir      : Base directory for raw/cached files.
        pg_config     : psycopg2 connection kwargs
                        (host, port, dbname, user, password).
        llr_threshold : Additional minimum LLR value for adverse-effect rows
                        (default 0.0 — keeps all rows where llr > per-row
                        llr_threshold column).
    """

    def __init__(
        self,
        data_dir: str,
        pg_config: Optional[Dict] = None,
        llr_threshold: float = 0.0,
        disease_scope: Optional[Dict] = None,
    ):
        super().__init__(data_dir)
        self.source_name = "drugcentral"
        self.source_dir = self.data_dir / self.source_name
        self.source_dir.mkdir(parents=True, exist_ok=True)

        # Build connection config: start from defaults, overlay supplied dict
        _defaults = {
            "host": "unmtid-dbs.net",
            "port": 5433,
            "dbname": "drugcentral",
        }
        _cfg = dict(pg_config) if pg_config else {}
        # Ensure port is an int (YAML may load it as str)
        if "port" in _cfg:
            _cfg["port"] = int(_cfg["port"])
        self._pg_config = {**_defaults, **_cfg}

        self.llr_threshold = float(llr_threshold)

        _cfg_scope = disease_scope if disease_scope else get_disease_scope()
        self.umls_cuis: List[str] = _cfg_scope.get("umls_cuis", [])
        if not self.umls_cuis:
            logger.warning(
                "No umls_cuis in disease_scope; drug_treats_disease will be empty."
            )

        try:
            import psycopg2  # noqa: F401
            self._pg_available = True
        except ImportError:
            self._pg_available = False
            logger.warning(
                "psycopg2 not installed. "
                "Run: pip install psycopg2-binary"
            )

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self):
        import psycopg2
        return psycopg2.connect(**self._pg_config)

    def _query(self, conn, sql: str, params=None) -> pd.DataFrame:
        """Execute *sql* and return a DataFrame; rolls back on error."""
        try:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                cols = [d[0] for d in cur.description]
                return pd.DataFrame(cur.fetchall(), columns=cols)
        except Exception:
            conn.rollback()
            raise

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    def download_data(self) -> bool:
        """
        Download the DrugCentral SQL dump (skip if already cached).

        The dump must be loaded into PostgreSQL before parse_data() will work.
        Returns True when the dump file is available locally or when the
        remote database is reachable (so the pipeline can proceed).
        """
        dump_path = self.source_dir / "drugcentral.sql.gz"
        if dump_path.exists() and not self.force:
            logger.info("DrugCentral dump already cached: %s", dump_path)
            return True

        # Try to reach the public database first (fast path)
        if self._pg_available:
            try:
                conn = self._connect()
                conn.close()
                logger.info(
                    "Connected to DrugCentral public PostgreSQL instance — "
                    "no local dump needed."
                )
                return True
            except Exception as exc:
                logger.debug("Public DB not reachable (%s); will download dump.", exc)

        logger.info("Downloading DrugCentral SQL dump from %s …", _DUMP_URL)
        result = self.download_file(_DUMP_URL, "drugcentral.sql.gz")
        if result:
            logger.info("Downloaded to %s", result)
            logger.info(
                "Load into PostgreSQL with: "
                "createdb drugcentral && "
                "gunzip -c drugcentral.sql.gz | psql drugcentral"
            )
            return True
        logger.error("Failed to download DrugCentral dump.")
        return False

    # ------------------------------------------------------------------
    # Parse
    # ------------------------------------------------------------------

    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Query DrugCentral and return the six output DataFrames.

        Returns an empty dict if psycopg2 is unavailable or the database
        cannot be reached.
        """
        if not self._pg_available:
            logger.error(
                "psycopg2 is required. Install with: pip install psycopg2-binary"
            )
            return {}

        result: Dict[str, pd.DataFrame] = {}

        try:
            conn = self._connect()
        except Exception as exc:
            logger.error(
                "Cannot connect to DrugCentral PostgreSQL: %s\n"
                "Connection config (without password): %s",
                exc,
                {k: v for k, v in self._pg_config.items() if k != "password"},
            )
            return {}

        try:
            for name, method in [
                ("drugs",                   self._query_drugs),
                ("pharmacologic_classes",   self._query_pharmacologic_classes),
                ("drug_treats_disease",     self._query_drug_treats_disease),
                ("drug_in_class",           self._query_drug_in_class),
                ("chemical_causes_effect",  self._query_chemical_causes_effect),
            ]:
                try:
                    df = method(conn)
                    if df is not None and len(df) > 0:
                        result[name] = df
                        logger.info("  %-30s %d rows", name, len(df))
                    else:
                        logger.warning("  %-30s 0 rows returned", name)
                except Exception as exc:
                    logger.error("  Error building %s: %s", name, exc)
        finally:
            conn.close()

        return result

    # ------------------------------------------------------------------
    # Individual query methods
    # ------------------------------------------------------------------

    def _query_drugs(self, conn) -> Optional[pd.DataFrame]:
        """
        drugs.tsv — structures joined with identifier table.

        Columns: struct_id | drugbank_id | cas_number | drug_name |
                 inchikey | inchi | smiles | molecular_weight |
                 molecular_formula | chebi_id | pubchem_cid |
                 mesh_id | umls_cui | source_database
        """
        sql = """
            SELECT
                s.id                                           AS struct_id,
                MAX(CASE WHEN i.id_type = 'DRUGBANK_ID'
                         THEN i.identifier END)               AS drugbank_id,
                s.cas_reg_no                                   AS cas_number,
                s.name                                         AS drug_name,
                s.inchikey,
                s.inchi,
                s.smiles,
                s.cd_molweight                                 AS molecular_weight,
                s.cd_formula                                   AS molecular_formula,
                s.clogp                                        AS logp,
                s.tpsa,
                s.lipinski                                     AS lipinski_compliance,
                MAX(CASE WHEN i.id_type = 'CHEBI'
                         THEN i.identifier END)               AS chebi_id,
                MAX(CASE WHEN i.id_type = 'PUBCHEM_CID'
                         THEN i.identifier END)               AS pubchem_cid,
                MAX(CASE WHEN i.id_type = 'MESH_DESCRIPTOR_UI'
                         THEN i.identifier END)               AS mesh_id,
                MAX(CASE WHEN i.id_type = 'UMLSCUI'
                         THEN i.identifier END)               AS umls_cui
            FROM structures s
            LEFT JOIN identifier i ON s.id = i.struct_id
            GROUP BY s.id, s.cas_reg_no, s.name, s.inchikey,
                     s.inchi, s.smiles, s.cd_molweight, s.cd_formula,
                     s.clogp, s.tpsa, s.lipinski
        """
        df = self._query(conn, sql)
        df["source_database"] = "drugcentral"
        return df

    def _query_pharmacologic_classes(self, conn) -> Optional[pd.DataFrame]:
        """
        pharmacologic_classes.tsv — distinct classes from pharma_class.

        Columns: pharma_class_id | pharma_class_name | pharma_class_code |
                 source_database
        pharma_class_id = "{source}:{class_code}" for uniqueness.
        """
        sql = """
            SELECT DISTINCT
                class_code                                     AS pharma_class_code,
                name                                           AS pharma_class_name,
                source                                         AS class_source
            FROM pharma_class
            WHERE class_code IS NOT NULL
              AND name       IS NOT NULL
        """
        df = self._query(conn, sql)
        df["pharma_class_id"] = (
            df["class_source"].fillna("DC") + ":" + df["pharma_class_code"]
        )
        df = df.drop_duplicates(subset=["pharma_class_id"])
        df["source_database"] = "drugcentral"
        return df[
            ["pharma_class_id", "pharma_class_name",
             "pharma_class_code", "source_database"]
        ].reset_index(drop=True)

    def _query_drug_treats_disease(self, conn) -> Optional[pd.DataFrame]:
        """
        drug_treats_disease.tsv — indication edges from omop_relationship,
        filtered to the project disease scope (umls_cuis).

        Columns: struct_id | disease_id | indication | source_database
        disease_id = umls_cui.
        """
        if not self.umls_cuis:
            return None

        sql = """
            SELECT
                struct_id,
                umls_cui          AS disease_id,
                relationship_name AS indication
            FROM omop_relationship
            WHERE relationship_name = 'indication'
              AND umls_cui = ANY(%s)
        """
        df = self._query(conn, sql, [self.umls_cuis])
        df["source_database"] = "drugcentral"
        df = df.drop_duplicates(subset=["struct_id", "disease_id"])
        return df[
            ["struct_id", "disease_id", "indication", "source_database"]
        ].reset_index(drop=True)

    def _query_drug_in_class(self, conn) -> Optional[pd.DataFrame]:
        """
        drug_in_class.tsv — drug–class memberships from pharma_class.

        Columns: struct_id | pharma_class_id | source_database
        pharma_class_id = "{source}:{class_code}" (matches pharmacologic_classes.tsv).
        """
        sql = """
            SELECT
                struct_id,
                class_code,
                source AS class_source
            FROM pharma_class
            WHERE class_code  IS NOT NULL
              AND struct_id   IS NOT NULL
        """
        df = self._query(conn, sql)
        df["pharma_class_id"] = (
            df["class_source"].fillna("DC") + ":" + df["class_code"]
        )
        df["source_database"] = "drugcentral"
        df = df.drop_duplicates(subset=["struct_id", "pharma_class_id"])
        return df[
            ["struct_id", "pharma_class_id", "source_database"]
        ].reset_index(drop=True)

    def _query_chemical_causes_effect(self, conn) -> Optional[pd.DataFrame]:
        """
        chemical_causes_effect.tsv — FAERS adverse-effect signals.

        Columns: struct_id | adverse_effect_id | adverse_effect_name |
                 llr | drug_ae | source_database

        Filters: llr > llr_threshold (per-row column) AND drug_ae >= 3.
        An additional floor from self.llr_threshold is applied if > 0.
        """
        sql = """
            SELECT
                struct_id,
                meddra_code   AS adverse_effect_id,
                meddra_name   AS adverse_effect_name,
                llr,
                drug_ae
            FROM faers
            WHERE llr      > llr_threshold
              AND drug_ae >= 3
              AND llr      > %s
        """
        df = self._query(conn, sql, [self.llr_threshold])
        df["source_database"] = "drugcentral"
        return df[
            ["struct_id", "adverse_effect_id", "adverse_effect_name",
             "llr", "drug_ae", "source_database"]
        ].reset_index(drop=True)

    # ------------------------------------------------------------------
    # Schema declaration
    # ------------------------------------------------------------------

    def get_schema(self) -> Dict[str, Dict[str, str]]:
        return {
            "drugs": {
                "struct_id":         "DrugCentral structure ID (integer)",
                "drugbank_id":       "DrugBank ID (DB prefix)",
                "cas_number":        "CAS Registry Number",
                "drug_name":         "INN drug name",
                "inchikey":          "InChIKey",
                "inchi":             "InChI string",
                "smiles":            "SMILES string",
                "molecular_weight":  "Molecular weight (Da)",
                "molecular_formula": "Molecular formula",
                "chebi_id":          "ChEBI identifier",
                "pubchem_cid":       "PubChem CID",
                "mesh_id":           "MeSH Descriptor UI",
                "umls_cui":          "UMLS Concept Unique Identifier",
                "source_database":   "Source database (drugcentral)",
            },
            "pharmacologic_classes": {
                "pharma_class_id":   "Unique class ID ({source}:{class_code})",
                "pharma_class_name": "Pharmacologic class name",
                "pharma_class_code": "Class code (NDFRT, ATC, MeSH, etc.)",
                "source_database":   "Source database (drugcentral)",
            },
            "drug_treats_disease": {
                "struct_id":         "DrugCentral structure ID",
                "disease_id":        "Disease ID (= umls_cui)",
                "relationship_name": "Relationship type (indication)",
                "source_database":   "Source database (drugcentral)",
            },
            "drug_in_class": {
                "struct_id":         "DrugCentral structure ID",
                "pharma_class_id":   "Pharmacologic class ID",
                "source_database":   "Source database (drugcentral)",
            },
            "chemical_causes_effect": {
                "struct_id":           "DrugCentral structure ID",
                "adverse_effect_id":   "MedDRA Preferred Term code",
                "adverse_effect_name": "MedDRA Preferred Term name",
                "llr":                 "Log-likelihood ratio (FAERS signal strength)",
                "drug_ae":             "Drug adverse-event pair count",
                "source_database":     "Source database (drugcentral)",
            },
        }
