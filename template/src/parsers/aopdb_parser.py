"""
AOPDBParser: Parser for AOP-DB (Adverse Outcome Pathway Database).

AOP-DB is a MySQL database containing information about adverse outcome
pathways, which describe causal relationships between molecular initiating
events and adverse outcomes.

Source: https://gaftp.epa.gov/EPADataCommons/ORD/AOP-DB/
Note: This is a large MySQL database that must be downloaded and imported.
"""

import logging
import re
from typing import Dict, Optional, Any, List
import pandas as pd
from .base_parser import BaseParser

AOPDB_PATHWAYS = 'pathways'
AOPDB_GENE_PATHWAY_RELATIONSHIPS = 'gene_pathway_relationships'
AOPDB_DRUGS = 'drugs'

AOPDB_TABLE_MAPPING = {
    AOPDB_PATHWAYS: 'pathway_gene',
    AOPDB_GENE_PATHWAY_RELATIONSHIPS: 'pathway_gene',
    AOPDB_DRUGS: 'chemical_info',
}

logger = logging.getLogger(__name__)


class AOPDBParser(BaseParser):
    """
    Parser for AOP-DB MySQL database.

    This parser connects to a MySQL database and extracts relevant
    adverse outcome pathway data.
    """

    def __init__(self, data_dir: Optional[str] = None,
                 mysql_config: Optional[Dict[str, str]] = None):
        """
        Initialize the AOP-DB parser.
        
        Args:
            data_dir: Directory for cached data
            mysql_config: MySQL connection configuration with keys:
                         'host', 'user', 'password', 'database'
        """
        super().__init__(data_dir)
        self.mysql_config = mysql_config or {}
        self.connection = None
        self._mysql_available = False
        
        # Check if MySQL connector is available
        try:
            import mysql.connector
            self._mysql_available = True
            logger.info("MySQL connector is available")
        except ImportError:
            logger.warning("MySQL connector not available. Install with: pip install mysql-connector-python")
    
    def _get_table_names(self) -> List[str]:
        """
        Get list of available tables in the database.
        
        Returns:
            List of table names.
        """
        if not self.connection:
            return []
        
        try:
            cursor = self.connection.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            cursor.close()
            return tables
        except Exception as e:
            logger.error(f"Failed to get table names: {e}")
            return []
    
    def _connect(self):
        """Open and return a new MySQL connection using self.mysql_config."""
        import mysql.connector
        return mysql.connector.connect(
            host=self.mysql_config.get('host', 'localhost'),
            user=self.mysql_config.get('user', 'root'),
            password=self.mysql_config.get('password', ''),
            database=self.mysql_config.get('database', 'aopdb')
        )

    def download_data(self) -> bool:
        """
        Check for AOP-DB database.

        Since AOP-DB is a large MySQL database, this method only checks
        if the database is accessible.

        Returns:
            True if database is accessible, False otherwise.
        """
        logger.info("Checking for AOP-DB database...")
        logger.info("Note: AOP-DB must be downloaded and imported into MySQL from:")
        logger.info("  https://gaftp.epa.gov/EPADataCommons/ORD/AOP-DB/")
        logger.info("  File: AOP-DB_v2.zip (7.2 GB compressed)")

        if not self._mysql_available:
            logger.error("MySQL connector not available")
            return False

        if not self.mysql_config:
            logger.error("MySQL configuration not provided")
            logger.info("Please provide mysql_config with: host, user, password, database")
            return False

        try:
            self.connection = self._connect()
            logger.info("✓ Successfully connected to AOP-DB")

            tables = self._get_table_names()
            logger.info(f"Found {len(tables)} tables in database")
            logger.debug(f"Available tables: {', '.join(tables)}")

            return True

        except Exception as e:
            logger.error(f"Failed to connect to AOP-DB: {e}")
            return False
    
    def parse_data(self) -> Dict[str, pd.DataFrame]:
        """
        Parse AOP-DB data.
        
        Extracts key tables from the AOP-DB MySQL database.
        Uses dynamic table name detection to handle schema variations.
        
        Returns:
            Dictionary where KEY = source filename stem; VALUE = pandas DataFrame for different AOP entities.
        """
        logger.info("Parsing AOP-DB data...")
        
        if not self.connection:
            if not self._mysql_available or not self.mysql_config:
                logger.error("Not connected to AOP-DB and cannot reconnect (missing config or connector).")
                return {}
            try:
                self.connection = self._connect()
                logger.info("✓ Reconnected to AOP-DB for parsing")
            except Exception as e:
                logger.error(f"Failed to connect to AOP-DB: {e}")
                return {}
        
        result = {}
        
        # Get available tables
        available_tables = self._get_table_names()
        logger.info(f"Available tables: {available_tables}")

        # Queries for each output key
        query = {
            AOPDB_PATHWAYS: """
                SELECT path_name,
                    GROUP_CONCAT(DISTINCT path_id ORDER BY path_id) as path_id,
                    CONCAT('AOPDB - ', GROUP_CONCAT(DISTINCT ext_source ORDER BY ext_source)) as ext_source
                FROM(
                    SELECT DISTINCT path_id,
                    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(path_name, '<sub>', ''), '</sub>', ''), '<i>', ''), '</i>', ''), ' - Homo sapiens (human)', '')) as path_name,
                    ext_source
                    FROM pathway_gene
                    WHERE tax_id = 9606)data
                GROUP BY path_name
                ORDER BY path_name
            """,
            AOPDB_GENE_PATHWAY_RELATIONSHIPS: """
                SELECT DISTINCT entrez,
                    path_id,
                    TRIM(REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(path_name, '<sub>', ''), '</sub>', ''), '<i>', ''), '</i>', ''), ' - Homo sapiens (human)', '')) as path_name
                FROM pathway_gene
                WHERE tax_id = 9606
                ORDER BY entrez, path_id
            """,
            AOPDB_DRUGS: """
                SELECT * FROM chemical_info
                ORDER BY ChemicalName
            """,
        }
        
        # Query each table type
        for result_key, table_name in AOPDB_TABLE_MAPPING.items():
            if table_name in available_tables:
                try:
                    df = pd.read_sql(query[result_key], self.connection)
                    df['source_database'] = 'AOPDB'
                    if result_key == AOPDB_PATHWAYS:
                        df['path_iri'] = (
                            df['path_name']
                            .str.strip()
                            .str.lower()
                            .apply(lambda s: re.sub(r"[^a-z0-9_\-\.:]", "_", s))
                        )
                    result[result_key] = df
                    logger.info(f"✓ Parsed {len(df)} rows from {table_name} (as {result_key})")
                except Exception as e:
                    logger.warning(f"Failed to query {table_name}: {e}")
        
        return result
    
    def get_schema(self) -> Dict[str, Dict[str, str]]:
        """
        Get the schema for AOP-DB data.
        
        Returns:
            Dictionary describing the schema for AOP entities.
        """
        return {
            AOPDB_PATHWAYS: {
                'path_id': 'Pathway identifier',
                'path_name': 'Pathway name (human-readable)',
                'path_iri': 'URI-safe pathway name used as ontology IRI fragment',
                'ext_source': 'External source',
                'source_database': 'AOP-DB'
            },
            AOPDB_GENE_PATHWAY_RELATIONSHIPS: {
                'entrez': 'Entrez identifier',
                'path_id': 'Pathway identifier',
                'path_name': 'Pathway name',
                'source_database': 'AOP-DB'
            },
            AOPDB_DRUGS: {
                'chemical_id': 'Chemical identifier',
                'chemical_name': 'Chemical name',
                'chemical_type': 'Chemical type',
                'chemical_description': 'Chemical description',
                'source_database': 'AOP-DB'
            }
        }
    
    def close(self):
        """Close database connection."""
        if self.connection:
            self.connection.close()
            logger.info("Closed database connection")
    
    def __del__(self):
        """Cleanup: close connection on deletion."""
        self.close()
