"""
TSV-based Memgraph exporter for the KG pipeline.

Reads processed TSV files and ontology mappings to generate
Memgraph-compatible CSV files and import.cypher script.

CRITICAL: Edge IDs must exactly match node IDs. This exporter builds
lookup tables from node CSVs and uses them to resolve edge endpoint IDs.

Cross-ontology ID resolution uses mapping files from data/mappings/:
- mesh_to_doid.tsv: MESH → DOID disease mappings
- umls_to_doid.tsv: UMLS → DOID disease mappings
- mondo_to_doid.tsv: MONDO → DOID disease mappings (if available)
"""

import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import yaml

logger = logging.getLogger(__name__)


class TSVMemgraphExporter:
    """
    Export processed TSV files to Memgraph-compatible CSV format.

    Reads TSV files from data/processed/<source>/ and ontology mappings
    to generate typed CSV files and import.cypher script.

    ID Resolution Strategy:
    - Nodes: created with id = {id_prefix}:{id_column_value}
    - Edges: endpoint IDs resolved via node lookup tables that map
      various identifier types (geneId, geneSymbol, xrefDrugBank, etc.)
      back to the canonical node ID.
    """

    def __init__(self, processed_dir: str, ontology_mappings_file: str, output_dir: str):
        self.processed_dir = Path(processed_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Locate data directory (parent of processed_dir)
        self.data_dir = self.processed_dir.parent
        self.mappings_dir = self.data_dir / "mappings"

        with open(ontology_mappings_file, 'r') as f:
            self.ontology_mappings = yaml.safe_load(f)

        self.node_configs = {}
        self.edge_configs = {}
        self._parse_configs()

        # Node lookup tables: node_type -> {identifier_value -> canonical_node_id}
        # We build multiple lookups per node type for different ID columns
        self.node_lookups: Dict[str, Dict[str, str]] = defaultdict(dict)

        # Cross-ontology ID mappings
        self.id_mappings: Dict[str, Dict[str, str]] = {
            'mesh_to_doid': {},
            'umls_to_doid': {},
            'mondo_to_doid': {},
            'efo_to_doid': {},
        }
        self._load_id_mappings()

        # Track all nodes and edges
        self.nodes_by_type: Dict[str, List[Dict]] = defaultdict(list)
        self.edges_by_type: Dict[str, List[Dict]] = defaultdict(list)

    def _load_id_mappings(self):
        """Load cross-ontology ID mapping files."""
        mapping_files = [
            ('mesh_to_doid', 'mesh_to_doid.tsv'),
            ('umls_to_doid', 'umls_to_doid.tsv'),
            ('mondo_to_doid', 'mondo_to_doid.tsv'),
            ('efo_to_doid', 'efo_to_doid.tsv'),
        ]

        for mapping_name, filename in mapping_files:
            filepath = self.mappings_dir / filename
            if not filepath.exists():
                continue

            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    for row in reader:
                        source_id = row.get('source_id', '').strip()
                        target_id = row.get('target_id', '').strip()
                        if source_id and target_id:
                            self.id_mappings[mapping_name][source_id] = target_id
                            self.id_mappings[mapping_name][source_id.lower()] = target_id
                logger.info(f"Loaded {len(self.id_mappings[mapping_name])} {mapping_name} mappings")
            except Exception as e:
                logger.warning(f"Failed to load {filename}: {e}")

    def _parse_configs(self):
        """Separate node and edge configurations."""
        for config_key, config in self.ontology_mappings.items():
            if config is None or config.get('skip', False):
                continue

            data_type = config.get('data_type')
            if data_type == 'node':
                self.node_configs[config_key] = config
            elif data_type == 'relationship':
                self.edge_configs[config_key] = config

    def export(self) -> dict:
        """
        Export TSVs to Memgraph CSV format.

        Two-pass approach:
        1. Export all nodes and build comprehensive lookup tables
        2. Export all edges using lookup tables to resolve IDs
        """
        output_files = []
        total_nodes = 0
        total_edges = 0
        node_columns: Dict[str, List[str]] = {}
        rel_types: List[str] = []

        # ====================================================================
        # PASS 1: Export all nodes and build lookup tables
        # ====================================================================
        logger.info("Pass 1: Exporting nodes and building lookup tables...")

        # Group node configs by owl_class to merge into single CSV per type
        nodes_by_class: Dict[str, List[Dict]] = defaultdict(list)
        columns_by_class: Dict[str, Set[str]] = defaultdict(set)

        for config_key, config in self.node_configs.items():
            node_type = config.get('owl_class')
            source_name = config_key.split('.')[0]
            source_filename = config.get('source_filename')

            if not source_filename:
                continue

            tsv_path = self.processed_dir / source_name / source_filename
            if not tsv_path.exists():
                logger.warning(f"TSV not found: {tsv_path}")
                continue

            logger.info(f"Processing nodes: {config_key} ({node_type})")
            nodes, cols = self._process_node_file(tsv_path, config, node_type)

            if nodes:
                nodes_by_class[node_type].extend(nodes)
                columns_by_class[node_type].update(cols)
                logger.info(f"  Added {len(nodes)} {node_type} nodes, lookup size: {len(self.node_lookups[node_type])}")

        # Write merged node CSVs (deduped by id)
        for node_type, all_nodes in nodes_by_class.items():
            # Deduplicate by id
            seen_ids = set()
            unique_nodes = []
            for node in all_nodes:
                if node['id'] not in seen_ids:
                    seen_ids.add(node['id'])
                    unique_nodes.append(node)

            columns = ['id'] + sorted(columns_by_class[node_type] - {'id'})
            csv_path = self.output_dir / f"nodes_{node_type}.csv"
            self._write_node_csv(csv_path, unique_nodes, columns, node_type)
            node_columns[node_type] = columns
            total_nodes += len(unique_nodes)
            output_files.append(str(csv_path))
            logger.info(f"Exported {len(unique_nodes)} {node_type} nodes")

        # Log lookup table sizes
        logger.info("Lookup table sizes:")
        for node_type, lookup in self.node_lookups.items():
            logger.info(f"  {node_type}: {len(lookup)} entries")

        # ====================================================================
        # PASS 2: Export edges using lookup tables
        # ====================================================================
        logger.info("\nPass 2: Exporting edges with ID resolution...")

        # Group edge configs by owl_relationship
        edges_by_rel: Dict[str, List[Dict]] = defaultdict(list)

        for config_key, config in self.edge_configs.items():
            rel_type = config.get('owl_relationship')
            source_name = config_key.split('.')[0]
            source_filename = config.get('source_filename')

            if not source_filename:
                continue

            tsv_path = self.processed_dir / source_name / source_filename
            if not tsv_path.exists():
                logger.warning(f"TSV not found: {tsv_path}")
                continue

            logger.info(f"Processing edges: {config_key} ({rel_type})")
            edges, stats = self._process_edge_file(tsv_path, config)

            if edges:
                edges_by_rel[rel_type].extend(edges)
                logger.info(f"  Resolved {len(edges)} edges ({stats})")

        # Write merged edge CSVs (deduped)
        for rel_type, all_edges in edges_by_rel.items():
            # Deduplicate
            seen = set()
            unique_edges = []
            for edge in all_edges:
                key = (edge[':START_ID'], edge[':END_ID'])
                if key not in seen:
                    seen.add(key)
                    unique_edges.append(edge)

            csv_path = self.output_dir / f"edges_{rel_type}.csv"
            self._write_edge_csv(csv_path, unique_edges, rel_type)
            total_edges += len(unique_edges)
            output_files.append(str(csv_path))
            rel_types.append(rel_type)
            logger.info(f"Exported {len(unique_edges)} {rel_type} edges")

        # Write Cypher import script
        cypher_path = self._write_cypher_script(node_columns, rel_types)
        output_files.append(str(cypher_path))

        logger.info(f"\nTotal: {total_nodes} nodes, {total_edges} edges")

        return {
            "nodes_count": total_nodes,
            "edges_count": total_edges,
            "output_files": output_files,
            "cypher_script": str(cypher_path),
        }

    def _process_node_file(self, tsv_path: Path, config: Dict, node_type: str) -> Tuple[List[Dict], Set[str]]:
        """
        Process a node TSV file and build lookup tables.

        Creates lookup entries for:
        - The primary ID (id_column value)
        - All property values that could be used as alternate IDs
        """
        nodes = []
        columns = set(['id'])

        id_column = config.get('id_column')
        id_prefix = config.get('id_prefix', '')
        property_map = config.get('property_map', {})

        if not id_column:
            logger.error(f"No id_column in config for {tsv_path}")
            return [], set()

        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    raw_id = row.get(id_column, '').strip()
                    if not raw_id:
                        continue

                    # Create canonical node ID, stripping existing prefix to avoid doubling
                    if id_prefix:
                        expected_prefix = f"{id_prefix}:"
                        if raw_id.startswith(expected_prefix):
                            node_id = raw_id
                        else:
                            node_id = f"{id_prefix}:{raw_id}"
                    else:
                        node_id = raw_id

                    node = {'id': node_id}

                    # Map properties and build lookups
                    for tsv_col, owl_prop in property_map.items():
                        value = row.get(tsv_col, '').strip()
                        if value:
                            node[owl_prop] = value
                            columns.add(owl_prop)

                            # Add to lookup table
                            # Key: the property value, Value: canonical node ID
                            self.node_lookups[node_type][value] = node_id

                            # Also add lowercase version for case-insensitive matching
                            self.node_lookups[node_type][value.lower()] = node_id

                    # Also add the raw ID to lookups
                    self.node_lookups[node_type][raw_id] = node_id
                    self.node_lookups[node_type][raw_id.lower()] = node_id

                    # And the full node ID
                    self.node_lookups[node_type][node_id] = node_id
                    self.node_lookups[node_type][node_id.lower()] = node_id

                    nodes.append(node)

        except Exception as e:
            logger.error(f"Error processing {tsv_path}: {e}")
            return [], set()

        return nodes, columns

    def _resolve_id(self, node_type: str, raw_value: str) -> Optional[str]:
        """
        Resolve a raw ID value to a canonical node ID.

        Tries multiple strategies:
        1. Direct lookup in node lookups
        2. Case-insensitive lookup
        3. Cross-ontology mapping (for Disease: MESH/UMLS/MONDO/EFO → DOID)
        4. With common prefixes removed/added
        """
        if not raw_value:
            return None

        lookup = self.node_lookups.get(node_type, {})

        # Try direct match
        if raw_value in lookup:
            return lookup[raw_value]

        # Try lowercase
        if raw_value.lower() in lookup:
            return lookup[raw_value.lower()]

        # Cross-ontology mapping for Disease type
        if node_type == 'Disease':
            mapped_id = self._map_disease_id(raw_value)
            if mapped_id and mapped_id in lookup:
                return lookup[mapped_id]
            if mapped_id and mapped_id.lower() in lookup:
                return lookup[mapped_id.lower()]

        # Try stripping common prefixes
        for prefix in ['MESH:', 'MeSH:', 'GO:', 'HP:', 'DOID:', 'UBERON:', 'NCBIGene:', 'DrugBank:', 'HGNC:', 'MONDO:', 'EFO:', 'EFO_', 'UMLS:', 'MedGen:']:
            if raw_value.startswith(prefix):
                stripped = raw_value[len(prefix):]
                if stripped in lookup:
                    return lookup[stripped]
                if stripped.lower() in lookup:
                    return lookup[stripped.lower()]

        # Try adding common prefixes
        common_prefixes = {
            'Gene': ['NCBIGene:', ''],
            'Drug': ['DrugBank:', 'CTD:', 'DrugCentral:', ''],
            'Disease': ['DOID:', 'DOID:DOID:', ''],
            'BiologicalProcess': ['GO:', 'GO:GO:', ''],
            'MolecularFunction': ['GO:', 'GO:GO:', ''],
            'CellularComponent': ['GO:', 'GO:GO:', ''],
            'Phenotype': ['HP:', 'HP:HP:', ''],
            'Pathway': ['Reactome:', 'R-HSA-', ''],
            'BodyPart': ['UBERON:', 'UBERON:UBERON:', ''],
            'Anatomy': ['UBERON:', 'UBERON:UBERON:', ''],
            'Variant': ['ClinVar:', ''],
        }

        for prefix in common_prefixes.get(node_type, ['']):
            test_id = f"{prefix}{raw_value}"
            if test_id in lookup:
                return lookup[test_id]
            if test_id.lower() in lookup:
                return lookup[test_id.lower()]

        return None

    def _map_disease_id(self, disease_id: str) -> Optional[str]:
        """
        Map a disease ID from various ontologies to DOID format.

        Supports: MESH, UMLS, MONDO, EFO → DOID
        """
        if not disease_id:
            return None

        # Already DOID format
        if disease_id.startswith('DOID:'):
            return disease_id

        # Try MESH mapping
        mesh_map = self.id_mappings.get('mesh_to_doid', {})
        if disease_id.startswith('MESH:') or disease_id.startswith('MeSH:'):
            mesh_id = disease_id.split(':', 1)[1]
            if mesh_id in mesh_map:
                return mesh_map[mesh_id]
            if f"MESH:{mesh_id}" in mesh_map:
                return mesh_map[f"MESH:{mesh_id}"]
        elif disease_id.startswith('D') or disease_id.startswith('C'):
            # Bare MESH ID
            if disease_id in mesh_map:
                return mesh_map[disease_id]

        # Try UMLS mapping
        umls_map = self.id_mappings.get('umls_to_doid', {})
        if disease_id.startswith('UMLS:') or disease_id.startswith('MedGen:'):
            umls_id = disease_id.split(':', 1)[1]
            if umls_id in umls_map:
                return umls_map[umls_id]
        elif disease_id.startswith('C') and len(disease_id) >= 7:
            # Bare UMLS CUI
            if disease_id in umls_map:
                return umls_map[disease_id]

        # Try MONDO mapping
        mondo_map = self.id_mappings.get('mondo_to_doid', {})
        if disease_id.startswith('MONDO:') or disease_id.startswith('MONDO_'):
            normalized = disease_id.replace('MONDO_', 'MONDO:')
            if normalized in mondo_map:
                return mondo_map[normalized]
            # Also try with double prefix (MONDO:MONDO:...)
            if disease_id in mondo_map:
                return mondo_map[disease_id]

        # Try EFO mapping
        efo_map = self.id_mappings.get('efo_to_doid', {})
        if disease_id.startswith('EFO:') or disease_id.startswith('EFO_'):
            if disease_id in efo_map:
                return efo_map[disease_id]
            # Try alternate format
            normalized = disease_id.replace('EFO_', 'EFO:')
            if normalized in efo_map:
                return efo_map[normalized]

        return None

    def _process_edge_file(self, tsv_path: Path, config: Dict) -> Tuple[List[Dict], str]:
        """
        Process an edge TSV file, resolving IDs via lookup tables.
        Includes edge properties from property_map and source_label.
        """
        edges = []
        source_node_type = config.get('source_node_type')
        target_node_type = config.get('target_node_type')
        source_id_column = config.get('source_id_column')
        target_id_column = config.get('target_id_column')
        property_map = config.get('property_map', {})
        source_label = config.get('source_label', '')

        if not source_id_column or not target_id_column:
            logger.error(f"Missing id columns in config for {tsv_path}")
            return [], "missing config"

        total_rows = 0
        resolved = 0
        source_fails = 0
        target_fails = 0

        try:
            with open(tsv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for row in reader:
                    total_rows += 1

                    source_raw = row.get(source_id_column, '').strip()
                    target_raw = row.get(target_id_column, '').strip()

                    if not source_raw or not target_raw:
                        continue

                    # Resolve IDs through lookup tables
                    start_id = self._resolve_id(source_node_type, source_raw)
                    end_id = self._resolve_id(target_node_type, target_raw)

                    if not start_id:
                        source_fails += 1
                        continue
                    if not end_id:
                        target_fails += 1
                        continue

                    edge = {
                        ':START_ID': start_id,
                        ':END_ID': end_id,
                    }

                    if source_label:
                        edge['source'] = source_label

                    for tsv_col, owl_prop in property_map.items():
                        value = row.get(tsv_col, '').strip()
                        if value:
                            edge[owl_prop] = value

                    edges.append(edge)
                    resolved += 1

        except Exception as e:
            logger.error(f"Error processing {tsv_path}: {e}")
            return [], f"error: {e}"

        stats = f"{resolved}/{total_rows} resolved"
        if source_fails:
            stats += f", {source_fails} source misses"
        if target_fails:
            stats += f", {target_fails} target misses"

        return edges, stats

    def _write_node_csv(self, csv_path: Path, nodes: List[Dict],
                       columns: List[str], node_type: str):
        """Write nodes to CSV file."""
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            header = columns + [':LABEL']
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', extrasaction='ignore')
            writer.writeheader()

            for node in nodes:
                row = {col: node.get(col, '') for col in columns}
                row[':LABEL'] = node_type
                writer.writerow(row)

    def _write_edge_csv(self, csv_path: Path, edges: List[Dict], rel_type: str):
        """Write edges to CSV file with all edge properties."""
        if not edges:
            return

        prop_keys = set()
        for edge in edges:
            prop_keys.update(k for k in edge if k not in (':START_ID', ':END_ID'))

        header = [':START_ID', ':END_ID', ':TYPE'] + sorted(prop_keys)

        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=header, delimiter=',', extrasaction='ignore')
            writer.writeheader()

            for edge in edges:
                row = {col: edge.get(col, '') for col in header}
                row[':TYPE'] = rel_type
                writer.writerow(row)

    def _write_cypher_script(self, node_columns: Dict[str, List[str]],
                           rel_types: List[str]) -> Path:
        """Generate import.cypher script with edge properties."""
        cypher_path = self.output_dir / "import.cypher"

        # Collect edge property columns per rel_type from exported CSVs
        edge_props: Dict[str, List[str]] = {}
        for rel_type in rel_types:
            csv_path = self.output_dir / f"edges_{rel_type}.csv"
            if csv_path.exists():
                with open(csv_path, 'r') as ef:
                    reader = csv.reader(ef)
                    header = next(reader, [])
                    props = [c for c in header if c not in (':START_ID', ':END_ID', ':TYPE')]
                    edge_props[rel_type] = props

        with open(cypher_path, 'w') as f:
            f.write("// Knowledge graph import script — generated by TSVMemgraphExporter\n")
            f.write("// docker run -it -p 7687:7687 -p 3000:3000 -v /abs/path/to/data/output:/import-data memgraph/memgraph-platform\n")
            f.write("// docker exec -i <container_id> mgconsole < /abs/path/to/data/output/import.cypher\n")
            f.write("// Note: LOAD CSV parses all values as strings. Use ToInteger()/ToFloat() for numeric comparisons.\n\n")

            f.write("// ============================================================================\n")
            f.write("// CREATE INDEXES\n")
            f.write("// ============================================================================\n\n")

            for node_type in sorted(node_columns.keys()):
                f.write(f"CREATE INDEX ON :{node_type}(id);\n")

            f.write("\n// ============================================================================\n")
            f.write("// LOAD NODES\n")
            f.write("// ============================================================================\n\n")

            for node_type in sorted(node_columns.keys()):
                csv_file = f"nodes_{node_type}.csv"
                f.write(f"LOAD CSV FROM '/import-data/{csv_file}' WITH HEADER AS row\n")
                f.write(f"CREATE (n:{node_type} {{id: row.id}})\n")
                f.write(f"SET n += row;\n\n")

            f.write("// ============================================================================\n")
            f.write("// LOAD EDGES\n")
            f.write("// ============================================================================\n\n")

            for rel_type in sorted(rel_types):
                csv_file = f"edges_{rel_type}.csv"
                props = edge_props.get(rel_type, [])
                f.write(f"LOAD CSV FROM '/import-data/{csv_file}' WITH HEADER AS row\n")
                f.write(f"MATCH (start {{id: row.`:START_ID`}})\n")
                f.write(f"MATCH (end {{id: row.`:END_ID`}})\n")
                if props:
                    set_parts = ', '.join(f"r.{p} = row.{p}" for p in props)
                    f.write(f"CREATE (start)-[r:{rel_type}]->(end)\n")
                    f.write(f"SET {set_parts};\n\n")
                else:
                    f.write(f"CREATE (start)-[r:{rel_type}]->(end);\n\n")

        return cypher_path
