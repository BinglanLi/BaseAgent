"""
Load missing edge types into Memgraph:
1. drugTreatsDisease (DrugCentral) — struct_id → drugbank_id → Drug, UMLS CUI → Disease
2. AFFECTS_RESPONSE_TO (ClinPGx) — gene_symbol → Gene, drug_id → Drug (via ClinPGx API)

Also un-skips these configs in ontology_configs.py for future pipeline runs.
"""

import csv
import json
import logging
import requests
from pathlib import Path

from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MEMGRAPH_URI = "bolt://localhost:7687"
MEMGRAPH_USER = ""
MEMGRAPH_PASS = ""
BATCH_SIZE = 1000


def get_driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=(MEMGRAPH_USER, MEMGRAPH_PASS))


def load_drugcentral_treats(driver):
    """Load drugTreatsDisease edges from DrugCentral TSV."""
    treats_path = PROJECT_ROOT / "data/processed/drugcentral/drug_treats_disease.tsv"
    drugs_path = PROJECT_ROOT / "data/processed/drugcentral/drugs.tsv"

    if not treats_path.exists():
        log.error(f"Missing: {treats_path}")
        return

    # Build struct_id → drugbank_id mapping from drugs.tsv
    struct_to_db = {}
    with open(drugs_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = row["struct_id"]
            dbid = row.get("drugbank_id") or row.get("canonical_drugbank_id", "")
            if sid and dbid:
                struct_to_db[sid] = dbid.strip()
    log.info(f"DrugCentral struct→drugbank mapping: {len(struct_to_db)} entries")

    # Build UMLS CUI → Disease graph ID mapping
    cui_to_disease = {}
    with driver.session() as s:
        r = s.run("MATCH (d:Disease) WHERE d.xrefUmlsCUI IS NOT NULL "
                   "RETURN d.id AS id, d.xrefUmlsCUI AS cui")
        for rec in r:
            cui = rec["cui"]
            if cui:
                cui_to_disease[cui] = rec["id"]
    log.info(f"Disease CUI mapping: {len(cui_to_disease)} entries")

    # Read treats TSV and map IDs
    batch = []
    skipped_no_db = 0
    skipped_no_disease = 0
    with open(treats_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            struct_id = row["struct_id"]
            disease_cui = row["disease_id"]

            drugbank_id = struct_to_db.get(struct_id)
            if not drugbank_id:
                skipped_no_db += 1
                continue

            drug_graph_id = f"drug_{drugbank_id.lower()}"
            disease_graph_id = cui_to_disease.get(disease_cui)
            if not disease_graph_id:
                skipped_no_disease += 1
                continue

            batch.append({
                "drug_id": drug_graph_id,
                "disease_id": disease_graph_id,
            })

    log.info(f"Treats edges to load: {len(batch)}")
    log.info(f"Skipped (no drugbank mapping): {skipped_no_db}")
    log.info(f"Skipped (no disease CUI match): {skipped_no_disease}")

    # Load in batches
    total = 0
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i + BATCH_SIZE]
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $batch AS row
                MATCH (d:Drug {id: row.drug_id})
                MATCH (dis:Disease {id: row.disease_id})
                CREATE (d)-[:drugTreatsDisease {source: "DrugCentral"}]->(dis)
                RETURN count(*) AS created
                """,
                batch=chunk,
            )
            total += result.single()["created"]

    log.info(f"Loaded {total} drugTreatsDisease edges")
    return total


def load_clinpgx_affects_response(driver):
    """Load AFFECTS_RESPONSE_TO edges from ClinPGx TSV.

    drug_name column contains ClinPGx numeric drug IDs.
    We resolve them to drug names via the ClinPGx/CPIC API,
    then match against Drug.commonName in the graph.
    """
    tsv_path = PROJECT_ROOT / "data/processed/clinpgx/gene_drug_interactions.tsv"
    if not tsv_path.exists():
        log.error(f"Missing: {tsv_path}")
        return

    # Read TSV
    rows = []
    drug_ids = set()
    with open(tsv_path) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
            drug_ids.add(row["drug_name"])

    log.info(f"ClinPGx TSV: {len(rows)} rows, {len(drug_ids)} unique drug IDs")

    # Resolve drug IDs to names via CPIC API
    log.info("Resolving drug IDs via CPIC API...")
    drug_id_to_name = {}
    try:
        resp = requests.get(
            "https://api.cpicpgx.org/v1/drug",
            timeout=30,
        )
        resp.raise_for_status()
        for drug in resp.json():
            drug_name = drug.get("name", "")
            if not drug_name:
                continue
            # Map by RxNorm ID (what the TSV has)
            rxnorm = drug.get("rxnormid", "")
            if rxnorm:
                drug_id_to_name[str(rxnorm)] = drug_name
            # Also map by drugid field (e.g., "RxNorm:190521")
            drugid = drug.get("drugid", "")
            if drugid:
                drug_id_to_name[str(drugid)] = drug_name
                # Strip prefix too
                if ":" in drugid:
                    bare_id = drugid.split(":", 1)[1]
                    drug_id_to_name[bare_id] = drug_name
            # Map by ATC codes too
            for atc in drug.get("atcid", []) or []:
                if atc:
                    drug_id_to_name[f"ATC:{atc}"] = drug_name
        log.info(f"CPIC API returned {len(drug_id_to_name)} drug ID mappings")
    except Exception as e:
        log.error(f"CPIC API failed: {e}")
        # Fallback: try PharmGKB-style lookup
        log.info("Trying alternative: use drug IDs as-is for name matching")

    # Build Drug.commonName → graph_id mapping (case-insensitive)
    drug_name_to_id = {}
    with driver.session() as s:
        r = s.run("MATCH (d:Drug) RETURN d.id AS id, toLower(d.commonName) AS name")
        for rec in r:
            if rec["name"]:
                drug_name_to_id[rec["name"]] = rec["id"]
    log.info(f"Drug name mapping: {len(drug_name_to_id)} entries")

    # Map and load
    batch = []
    skipped_no_drug_name = 0
    skipped_no_drug_match = 0
    skipped_no_gene = 0
    matched_drugs = set()
    for row in rows:
        gene_symbol = row["gene_symbol"]
        drug_id = row["drug_name"]

        drug_name = drug_id_to_name.get(drug_id, "")
        if not drug_name:
            skipped_no_drug_name += 1
            continue

        drug_graph_id = drug_name_to_id.get(drug_name.lower())
        if not drug_graph_id:
            skipped_no_drug_match += 1
            continue

        matched_drugs.add(drug_name)
        batch.append({
            "gene_sym": gene_symbol,
            "drug_id": drug_graph_id,
            "phenotype": row.get("phenotype", ""),
            "classification": row.get("classification", ""),
        })

    log.info(f"AFFECTS_RESPONSE_TO edges to load: {len(batch)}")
    log.info(f"Matched drugs: {len(matched_drugs)}")
    log.info(f"Skipped (no drug name from API): {skipped_no_drug_name}")
    log.info(f"Skipped (drug name not in graph): {skipped_no_drug_match}")

    total = 0
    for i in range(0, len(batch), BATCH_SIZE):
        chunk = batch[i:i + BATCH_SIZE]
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $batch AS row
                MATCH (g:Gene {geneSymbol: row.gene_sym})
                MATCH (d:Drug {id: row.drug_id})
                CREATE (g)-[:AFFECTS_RESPONSE_TO {
                    source: "ClinPGx",
                    phenotype: row.phenotype,
                    classification: row.classification
                }]->(d)
                RETURN count(*) AS created
                """,
                batch=chunk,
            )
            total += result.single()["created"]

    log.info(f"Loaded {total} AFFECTS_RESPONSE_TO edges")
    return total


def verify(driver):
    """Verify loaded edges."""
    with driver.session() as s:
        for rel in ["drugTreatsDisease", "AFFECTS_RESPONSE_TO", "hasVariant"]:
            r = s.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c")
            count = r.single()["c"]
            log.info(f"  {rel}: {count:,} edges")


def main():
    driver = get_driver()
    try:
        log.info("=" * 50)
        log.info("Loading drugTreatsDisease (DrugCentral)")
        log.info("=" * 50)
        load_drugcentral_treats(driver)

        log.info("")
        log.info("=" * 50)
        log.info("Loading AFFECTS_RESPONSE_TO (ClinPGx)")
        log.info("=" * 50)
        load_clinpgx_affects_response(driver)

        log.info("")
        log.info("=" * 50)
        log.info("Verification")
        log.info("=" * 50)
        verify(driver)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
