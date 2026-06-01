#!/usr/bin/env python3
"""
Load exported CSV files into Memgraph using IN_MEMORY_ANALYTICAL storage mode.

Reads node and edge CSVs from data/output/ and loads them via the neo4j driver
using batched UNWIND operations. Switches to IN_MEMORY_ANALYTICAL mode for bulk
loading (no WAL, no MVCC overhead), then back to IN_MEMORY_TRANSACTIONAL.

Usage:
    python scripts/load_csvs_to_memgraph.py
"""

import csv
import logging
import os
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv
from neo4j import GraphDatabase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

BATCH_SIZE = 10000
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "output"


def connect():
    load_dotenv()
    uri = os.environ.get("MEMGRAPH_URI", "bolt://localhost:7687")
    user = os.environ.get("MEMGRAPH_USERNAME", "")
    pwd = os.environ.get("MEMGRAPH_PASSWORD", "")
    driver = GraphDatabase.driver(uri, auth=(user, pwd))
    driver.verify_connectivity()
    logger.info(f"Connected to Memgraph at {uri}")
    return driver


def run(driver, query, **kwargs):
    with driver.session() as s:
        return s.run(query, **kwargs).consume()


def batch_load(driver, query, rows, label="", source_label=None):
    total = len(rows)
    created = 0
    params = {}
    if source_label:
        params["source_label"] = source_label

    for i in range(0, total, BATCH_SIZE):
        batch = rows[i : i + BATCH_SIZE]
        with driver.session() as s:
            result = s.run(query, rows=batch, **params)
            summary = result.consume()
            created += summary.counters.nodes_created + summary.counters.relationships_created

    return created


def read_csv(filepath):
    rows = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            cleaned = {k: v for k, v in row.items() if v != ""}
            rows.append(cleaned)
    return rows


def main():
    driver = connect()

    logger.info("Switching to IN_MEMORY_ANALYTICAL storage mode...")
    run(driver, "STORAGE MODE IN_MEMORY_ANALYTICAL")

    logger.info("Setting transaction timeout to 600s...")
    try:
        run(driver, "SET GLOBAL TRANSACTION TIMEOUT 600")
    except Exception:
        pass

    logger.info("Dropping all existing data...")
    run(driver, "MATCH (n) DETACH DELETE n")

    logger.info("Dropping existing indexes...")
    with driver.session() as s:
        result = s.run("SHOW INDEX INFO")
        indexes = list(result)
    for idx in indexes:
        label = idx.get("label", "")
        prop = idx.get("property", "")
        if label and prop:
            try:
                run(driver, f"DROP INDEX ON :{label}({prop})")
            except Exception:
                pass

    node_files = sorted(OUTPUT_DIR.glob("nodes_*.csv"))
    edge_files = sorted(OUTPUT_DIR.glob("edges_*.csv"))

    logger.info(f"Found {len(node_files)} node files, {len(edge_files)} edge files")

    # Create indexes first
    logger.info("Creating indexes on all node types...")
    for nf in node_files:
        node_type = nf.stem.replace("nodes_", "")
        try:
            run(driver, f"CREATE INDEX ON :{node_type}(id)")
            logger.info(f"  Index: :{node_type}(id)")
        except Exception as e:
            if "already exists" not in str(e).lower():
                logger.warning(f"  Failed index :{node_type}(id): {e}")

    # Load nodes
    logger.info("=" * 60)
    logger.info("Loading nodes...")
    logger.info("=" * 60)
    total_nodes = 0

    for nf in node_files:
        node_type = nf.stem.replace("nodes_", "")
        t0 = time.time()
        rows = read_csv(nf)
        if not rows:
            continue

        cols = list(rows[0].keys())
        cols_no_label = [c for c in cols if c != ":LABEL"]

        prop_assignments = ", ".join(f"{c}: row.{c}" for c in cols_no_label)
        query = f"UNWIND $rows AS row CREATE (n:{node_type} {{{prop_assignments}}})"

        created = batch_load(driver, query, rows)
        elapsed = time.time() - t0
        logger.info(f"  {node_type}: {len(rows):,} rows -> {created:,} nodes ({elapsed:.1f}s)")
        total_nodes += len(rows)

    logger.info(f"Total nodes loaded: {total_nodes:,}")

    # Build rel_type → (source_label, target_label) map from ontology_mappings.yaml
    mappings_file = Path(__file__).parent.parent / "config" / "ontology_mappings.yaml"
    with open(mappings_file, "r") as f:
        mappings = yaml.safe_load(f)

    rel_node_types = {}
    for key, cfg in mappings.items():
        if not isinstance(cfg, dict) or cfg.get("data_type") != "relationship":
            continue
        if cfg.get("skip"):
            continue
        rel = cfg.get("owl_relationship", "")
        src_type = cfg.get("source_node_type", "")
        tgt_type = cfg.get("target_node_type", "")
        if rel and src_type and tgt_type and rel not in rel_node_types:
            rel_node_types[rel] = (src_type, tgt_type)

    # Load edges
    logger.info("=" * 60)
    logger.info("Loading edges...")
    logger.info("=" * 60)
    total_edges = 0

    for ef in edge_files:
        rel_type = ef.stem.replace("edges_", "")
        t0 = time.time()
        rows = read_csv(ef)
        if not rows:
            continue

        sample = rows[0]
        prop_cols = [c for c in sample.keys() if c not in (":START_ID", ":END_ID", ":TYPE")]

        set_parts = []
        for pc in prop_cols:
            set_parts.append(f"r.{pc} = row.{pc}")
        set_clause = f"SET {', '.join(set_parts)}" if set_parts else ""

        src_label, tgt_label = rel_node_types.get(rel_type, ("", ""))
        src_match = f"(s:{src_label} {{id: row.`:START_ID`}})" if src_label else "(s {id: row.`:START_ID`})"
        tgt_match = f"(t:{tgt_label} {{id: row.`:END_ID`}})" if tgt_label else "(t {id: row.`:END_ID`})"

        query = (
            f"UNWIND $rows AS row "
            f"MATCH {src_match} "
            f"MATCH {tgt_match} "
            f"CREATE (s)-[r:{rel_type}]->(t) "
            f"{set_clause}"
        )

        created = batch_load(driver, query, rows)
        elapsed = time.time() - t0
        logger.info(f"  {rel_type}: {len(rows):,} rows -> {created:,} edges ({elapsed:.1f}s)")
        total_edges += created

    logger.info(f"Total edges loaded: {total_edges:,}")

    # Switch back to transactional mode
    logger.info("Switching back to IN_MEMORY_TRANSACTIONAL storage mode...")
    run(driver, "STORAGE MODE IN_MEMORY_TRANSACTIONAL")

    # Verify
    logger.info("=" * 60)
    logger.info("Verification")
    logger.info("=" * 60)
    with driver.session() as s:
        node_count = s.run("MATCH (n) RETURN count(n) AS cnt").single()["cnt"]
        edge_count = s.run("MATCH ()-[r]->() RETURN count(r) AS cnt").single()["cnt"]

        logger.info(f"Total nodes: {node_count:,}")
        logger.info(f"Total edges: {edge_count:,}")

        labels_result = s.run("MATCH (n) RETURN DISTINCT labels(n) AS l, count(n) AS cnt ORDER BY cnt DESC")
        for rec in labels_result:
            label = rec["l"][0] if rec["l"] else "?"
            logger.info(f"  {label}: {rec['cnt']:,}")

        rel_result = s.run("MATCH ()-[r]->() RETURN DISTINCT type(r) AS t, count(r) AS cnt ORDER BY cnt DESC")
        logger.info("Relationship types:")
        for rec in rel_result:
            logger.info(f"  {rec['t']}: {rec['cnt']:,}")

        source_result = s.run("MATCH ()-[r]->() WHERE r.source IS NOT NULL RETURN DISTINCT r.source AS src, count(r) AS cnt ORDER BY cnt DESC")
        logger.info("Source labels:")
        for rec in source_result:
            logger.info(f"  {rec['src']}: {rec['cnt']:,}")

    driver.close()
    logger.info("Done.")


if __name__ == "__main__":
    main()
