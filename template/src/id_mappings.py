"""
Cross-ontology ID mapping module for the KG pipeline.

Provides mappings between different identifier systems:
- EFO → DOID (Experimental Factor Ontology to Disease Ontology)
- MESH → DOID (Medical Subject Headings to Disease Ontology)
- BTO → UBERON (BRENDA Tissue Ontology to Uberon)
- ENSP → NCBIGene (Ensembl Protein to NCBI Gene)

Mappings are loaded from TSV files in data/mappings/ or built from
cross-references in existing node data.
"""

import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Set

logger = logging.getLogger(__name__)


class IDMapper:
    """
    Central ID mapping service for cross-ontology resolution.

    Builds mappings from:
    1. Explicit mapping files (data/mappings/*.tsv)
    2. Cross-references extracted from node TSV files
    """

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.mappings_dir = self.data_dir / "mappings"
        self.mappings_dir.mkdir(parents=True, exist_ok=True)

        # Mapping dictionaries: source_id -> target_id
        self.efo_to_doid: Dict[str, str] = {}
        self.mesh_to_doid: Dict[str, str] = {}
        self.bto_to_uberon: Dict[str, str] = {}
        self.ensp_to_ncbigene: Dict[str, str] = {}
        self.umls_to_doid: Dict[str, str] = {}

    def load_all_mappings(self, processed_dir: Path):
        """Load all available mappings from files and node cross-references."""
        logger.info("Loading ID mappings...")

        # Load from explicit mapping files
        self._load_mapping_file("efo_to_doid.tsv", self.efo_to_doid)
        self._load_mapping_file("mesh_to_doid.tsv", self.mesh_to_doid)
        self._load_mapping_file("bto_to_uberon.tsv", self.bto_to_uberon)
        self._load_mapping_file("ensp_to_ncbigene.tsv", self.ensp_to_ncbigene)

        # Extract mappings from node cross-references
        self._extract_disease_xrefs(processed_dir)
        self._extract_gene_xrefs(processed_dir)
        self._extract_bodypart_xrefs(processed_dir)

        logger.info(f"Loaded mappings: EFO→DOID: {len(self.efo_to_doid)}, "
                   f"MESH→DOID: {len(self.mesh_to_doid)}, "
                   f"BTO→UBERON: {len(self.bto_to_uberon)}, "
                   f"ENSP→NCBIGene: {len(self.ensp_to_ncbigene)}, "
                   f"UMLS→DOID: {len(self.umls_to_doid)}")

    def _load_mapping_file(self, filename: str, target_dict: Dict[str, str]):
        """Load a TSV mapping file into a dictionary."""
        filepath = self.mappings_dir / filename
        if not filepath.exists():
            return

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    source_id = row.get('source_id', '').strip()
                    target_id = row.get('target_id', '').strip()
                    if source_id and target_id:
                        target_dict[source_id] = target_id
                        target_dict[source_id.lower()] = target_id
            logger.info(f"Loaded {len(target_dict)} mappings from {filename}")
        except Exception as e:
            logger.warning(f"Failed to load {filename}: {e}")

    def _extract_disease_xrefs(self, processed_dir: Path):
        """Extract MESH/UMLS → DOID mappings from Disease Ontology data."""
        do_path = processed_dir / "disease_ontology" / "slim_terms.tsv"
        if not do_path.exists():
            return

        try:
            with open(do_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    doid = row.get('doid', '').strip()
                    if not doid:
                        continue

                    # Full DOID ID format
                    doid_full = f"DOID:{doid}" if not doid.startswith("DOID:") else doid

                    # UMLS CUI mapping
                    umls = row.get('umls_cui', '').strip()
                    if umls:
                        self.umls_to_doid[umls] = doid_full
                        self.umls_to_doid[f"UMLS:{umls}"] = doid_full

                    # MeSH ID mapping
                    mesh = row.get('mesh_id', '').strip()
                    if mesh:
                        self.mesh_to_doid[mesh] = doid_full
                        self.mesh_to_doid[f"MESH:{mesh}"] = doid_full

                    # EFO ID mapping
                    efo = row.get('efo_id', '').strip()
                    if efo:
                        normalized = efo.replace("EFO:", "EFO_")
                        self.efo_to_doid[efo] = doid_full
                        self.efo_to_doid[normalized] = doid_full

            logger.info(f"Extracted {len(self.umls_to_doid)} UMLS→DOID, "
                       f"{len(self.mesh_to_doid)} MESH→DOID, "
                       f"{len(self.efo_to_doid)} EFO→DOID mappings from Disease Ontology")
        except Exception as e:
            logger.warning(f"Failed to extract disease xrefs: {e}")

    def _extract_gene_xrefs(self, processed_dir: Path):
        """Extract ENSP → NCBIGene mappings from gene data."""
        gene_path = processed_dir / "ncbigene" / "genes.tsv"
        if not gene_path.exists():
            return

        try:
            with open(gene_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    gene_id = row.get('GeneID', '').strip()
                    if not gene_id:
                        continue

                    ncbi_full = f"NCBIGene:{gene_id}"

                    # Ensembl gene ID (ENSG) - already handled by exporter
                    # But let's also try to find protein mappings if available
                    # Note: NCBI Gene doesn't directly provide ENSP mappings
                    # This would need external UniProt/Ensembl mapping files

            logger.info("Gene xref extraction complete (ENSP mapping requires external data)")
        except Exception as e:
            logger.warning(f"Failed to extract gene xrefs: {e}")

    def _extract_bodypart_xrefs(self, processed_dir: Path):
        """Extract BTO → UBERON mappings from Uberon data."""
        uberon_path = processed_dir / "uberon" / "uberon_nodes.tsv"
        if not uberon_path.exists():
            return

        try:
            with open(uberon_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    uberon_id = row.get('uberon_id', '').strip()
                    if not uberon_id:
                        continue

                    uberon_full = f"UBERON:{uberon_id}" if not uberon_id.startswith("UBERON:") else uberon_id

                    # BTO mapping would need to come from external BTO-UBERON mapping file
                    # Uberon doesn't directly include BTO xrefs

            logger.info("BodyPart xref extraction complete (BTO mapping requires external data)")
        except Exception as e:
            logger.warning(f"Failed to extract bodypart xrefs: {e}")

    def map_to_doid(self, disease_id: str) -> Optional[str]:
        """
        Map any disease ID to DOID format.

        Supports: EFO, MESH, UMLS, MONDO, and raw DOID IDs.
        """
        if not disease_id:
            return None

        # Already DOID
        if disease_id.startswith("DOID:"):
            return disease_id

        # Strip common prefixes and try mappings
        clean_id = disease_id

        # EFO format: EFO_0000318 or EFO:0000318
        if disease_id.startswith("EFO"):
            if disease_id in self.efo_to_doid:
                return self.efo_to_doid[disease_id]
            # Try normalized form
            normalized = disease_id.replace("EFO_", "EFO:").replace("EFO:", "")
            if f"EFO:{normalized}" in self.efo_to_doid:
                return self.efo_to_doid[f"EFO:{normalized}"]

        # MESH format: MESH:D000123 or D000123
        if disease_id.startswith("MESH:") or disease_id.startswith("MeSH:"):
            clean_id = disease_id.split(":", 1)[1]
        if clean_id in self.mesh_to_doid:
            return self.mesh_to_doid[clean_id]
        if f"MESH:{clean_id}" in self.mesh_to_doid:
            return self.mesh_to_doid[f"MESH:{clean_id}"]

        # UMLS format
        if disease_id.startswith("UMLS:") or disease_id.startswith("C"):
            clean_id = disease_id.replace("UMLS:", "")
            if clean_id in self.umls_to_doid:
                return self.umls_to_doid[clean_id]

        return None

    def map_to_uberon(self, tissue_id: str) -> Optional[str]:
        """Map BTO or other tissue ID to UBERON format."""
        if not tissue_id:
            return None

        if tissue_id.startswith("UBERON:"):
            return tissue_id

        if tissue_id.startswith("BTO:"):
            if tissue_id in self.bto_to_uberon:
                return self.bto_to_uberon[tissue_id]

        return None

    def map_to_ncbigene(self, gene_id: str) -> Optional[str]:
        """Map ENSP or other gene ID to NCBIGene format."""
        if not gene_id:
            return None

        if gene_id.startswith("NCBIGene:"):
            return gene_id

        if gene_id.startswith("ENSP"):
            if gene_id in self.ensp_to_ncbigene:
                return self.ensp_to_ncbigene[gene_id]

        return None


def download_efo_doid_mappings(output_path: Path) -> bool:
    """
    Download EFO to DOID mappings from OxO (Ontology Xref Service) or EFO OWL.

    Note: This requires network access and may need API keys for large downloads.
    For now, we'll create a stub that can be populated manually or via API.
    """
    logger.info("EFO→DOID mapping download not implemented - use manual mapping file")
    return False


def download_mesh_doid_mappings(output_path: Path) -> bool:
    """
    Download MeSH to DOID mappings from Disease Ontology xrefs.

    Disease Ontology includes MeSH xrefs that can be extracted.
    """
    logger.info("MESH→DOID mapping download not implemented - use manual mapping file")
    return False
