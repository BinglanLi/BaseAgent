"""
Load CTD chemical-disease therapeutic edges into Memgraph as drugTreatsDisease.

Matching strategy:
  - CTD chemical MeSH ID -> Drug.xrefMeSH
  - CTD disease MeSH ID -> Disease.xrefUmlsCUI (via disease name fuzzy match)

Downloads CTD_chemicals_diseases.tsv.gz if not cached, filters for
DirectEvidence='therapeutic', maps IDs, and loads edges.
"""

import csv
import gzip
import logging
from collections import defaultdict
from pathlib import Path

import requests
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CTD_CACHE = PROJECT_ROOT / "data" / "raw" / "ctd"
CTD_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
CTD_FILE = "CTD_chemicals_diseases.tsv.gz"
TREATS_TSV = PROJECT_ROOT / "data" / "processed" / "ctd" / "chemical_treats_disease.tsv"

MEMGRAPH_URI = "bolt://localhost:7687"


def get_driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))


def download_if_needed():
    """Download CTD chemical-disease file if not cached."""
    path = CTD_CACHE / CTD_FILE
    if path.exists():
        log.info(f"Using cached {path}")
        return path

    CTD_CACHE.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading {CTD_URL} ...")
    resp = requests.get(CTD_URL, timeout=120, allow_redirects=True)
    resp.raise_for_status()
    path.write_bytes(resp.content)
    log.info(f"Downloaded {len(resp.content) / 1e6:.1f} MB to {path}")
    return path


def parse_therapeutic_edges(gz_path):
    """Parse CTD file for DirectEvidence='therapeutic' rows."""
    log.info("Parsing therapeutic edges...")
    rows = []
    with gzip.open(gz_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 10:
                continue
            if fields[5] != "therapeutic":
                continue
            chem_id = fields[1].strip()
            if chem_id and not chem_id.startswith("MESH:"):
                chem_id = f"MESH:{chem_id}"
            rows.append({
                "chemical_id": chem_id,
                "chemical_name": fields[0].strip(),
                "disease_id": fields[4].strip(),
                "disease_name": fields[3].strip(),
                "pubmed_ids": fields[9] if len(fields) > 9 else "",
            })

    log.info(f"Parsed {len(rows):,} therapeutic rows")

    # Deduplicate
    seen = set()
    unique = []
    for r in rows:
        key = (r["chemical_id"], r["disease_id"])
        if key not in seen:
            seen.add(key)
            unique.append(r)

    log.info(f"After dedup: {len(unique):,} unique chemical-disease pairs")
    return unique


def save_tsv(rows):
    """Save parsed treats edges as TSV for pipeline compatibility."""
    TREATS_TSV.parent.mkdir(parents=True, exist_ok=True)
    with open(TREATS_TSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["chemical_id", "disease_id", "chemical_name", "disease_name", "pubmed_ids", "source_database"],
            delimiter="\t",
        )
        writer.writeheader()
        for r in rows:
            r["source_database"] = "CTD"
            writer.writerow(r)
    log.info(f"Saved TSV: {TREATS_TSV}")


def build_drug_mesh_map(driver):
    """Build MeSH ID -> Drug graph ID mapping."""
    mesh_to_drug = {}
    with driver.session() as s:
        result = s.run("MATCH (d:Drug) RETURN d")
        for rec in result:
            node = dict(rec["d"])
            mesh = node.get("xrefMeSH", "")
            drug_id = node.get("id", "")
            if mesh and drug_id:
                mesh_to_drug[mesh] = drug_id
    log.info(f"Drug MeSH mapping: {len(mesh_to_drug):,} entries")
    return mesh_to_drug


def build_disease_name_map(driver):
    """Build disease name -> Disease graph ID mapping.

    CTD uses MeSH disease IDs but our Disease nodes use DOID.
    Match by disease name (case-insensitive) as the bridge.
    Also build a synonym-based lookup.
    """
    name_to_ids = defaultdict(list)
    with driver.session() as s:
        result = s.run("MATCH (d:Disease) RETURN d")
        for rec in result:
            node = dict(rec["d"])
            disease_id = node.get("id", "")
            name = (node.get("diseaseName") or "").lower().strip()
            if name and disease_id:
                name_to_ids[name].append(disease_id)
            # Also index synonyms
            synonyms = node.get("synonyms", "") or ""
            for syn in synonyms.split("|"):
                syn = syn.strip().lower()
                if syn and disease_id:
                    name_to_ids[syn].append(disease_id)

    log.info(f"Disease name mapping: {len(name_to_ids):,} name/synonym entries")
    return name_to_ids


def load_edges(driver, rows, drug_mesh_map, disease_name_map):
    """Match CTD rows to graph nodes and load drugTreatsDisease edges."""
    batch = []
    matched_drugs = set()
    matched_diseases = set()
    skipped_no_drug = 0
    skipped_no_disease = 0

    for r in rows:
        drug_graph_id = drug_mesh_map.get(r["chemical_id"])
        if not drug_graph_id:
            skipped_no_drug += 1
            continue

        disease_name = r["disease_name"].lower().strip()
        disease_ids = disease_name_map.get(disease_name, [])
        if not disease_ids:
            skipped_no_disease += 1
            continue

        matched_drugs.add(r["chemical_id"])
        matched_diseases.add(disease_name)
        for did in disease_ids:
            batch.append({"drug_id": drug_graph_id, "disease_id": did})

    log.info(f"\nEdges to load: {len(batch):,}")
    log.info(f"Matched drugs: {len(matched_drugs):,}")
    log.info(f"Matched diseases: {len(matched_diseases):,}")
    log.info(f"Skipped (no Drug match): {skipped_no_drug:,}")
    log.info(f"Skipped (no Disease match): {skipped_no_disease:,}")

    # Deduplicate (drug_id, disease_id) pairs
    seen = set()
    unique_batch = []
    for b in batch:
        key = (b["drug_id"], b["disease_id"])
        if key not in seen:
            seen.add(key)
            unique_batch.append(b)
    log.info(f"After dedup: {len(unique_batch):,} unique edges")

    # Filter out existing drugTreatsDisease edges
    existing = set()
    with driver.session() as s:
        result = s.run(
            "MATCH (d:Drug)-[:drugTreatsDisease]->(dis:Disease) "
            "RETURN d.id AS drug_id, dis.id AS disease_id"
        )
        for rec in result:
            existing.add((rec["drug_id"], rec["disease_id"]))

    new_batch = [b for b in unique_batch if (b["drug_id"], b["disease_id"]) not in existing]
    log.info(f"Already exist: {len(unique_batch) - len(new_batch):,}")
    log.info(f"New edges to load: {len(new_batch):,}")

    total = 0
    for i in range(0, len(new_batch), 1000):
        chunk = new_batch[i : i + 1000]
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $batch AS row
                MATCH (d:Drug {id: row.drug_id})
                MATCH (dis:Disease {id: row.disease_id})
                CREATE (d)-[:drugTreatsDisease {source: "CTD"}]->(dis)
                RETURN count(*) AS created
                """,
                batch=chunk,
            )
            total += result.single()["created"]

    log.info(f"\nLoaded {total:,} CTD drugTreatsDisease edges")
    return total


def main():
    gz_path = download_if_needed()
    rows = parse_therapeutic_edges(gz_path)
    save_tsv(rows)

    driver = get_driver()
    try:
        drug_mesh_map = build_drug_mesh_map(driver)
        disease_name_map = build_disease_name_map(driver)
        ctd_count = load_edges(driver, rows, drug_mesh_map, disease_name_map)

        log.info("\n" + "=" * 50)
        log.info("Final drugTreatsDisease summary")
        log.info("=" * 50)
        with driver.session() as s:
            result = s.run(
                "MATCH ()-[r:drugTreatsDisease]->() "
                "RETURN r.source AS source, count(r) AS c ORDER BY c DESC"
            )
            total = 0
            for rec in result:
                log.info(f"  {rec['source']}: {rec['c']:,}")
                total += rec["c"]
            log.info(f"  TOTAL: {total:,}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
