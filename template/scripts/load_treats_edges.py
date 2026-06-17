"""
Load drugTreatsDisease edges from two sources:
1. DrugCentral treats TSV (struct_id -> DrugBank ID, UMLS CUI -> Disease)
2. ClinicalTrials.gov Phase 3/4 trials (inferred drug-disease treatment pairs)
"""

import csv
import logging
from collections import defaultdict
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger()

MEMGRAPH_URI = "bolt://localhost:7687"


def get_driver():
    return GraphDatabase.driver(MEMGRAPH_URI, auth=("", ""))


def build_cui_mapping(driver):
    """Build CUI -> list of Disease graph IDs, handling pipe-separated CUI fields."""
    cui_to_ids = defaultdict(list)
    with driver.session() as s:
        result = s.run("MATCH (d:Disease) RETURN d")
        for rec in result:
            node = dict(rec["d"])
            cui_field = node.get("xrefUmlsCUI", "")
            disease_id = node.get("id", "")
            if cui_field and disease_id:
                for cui in cui_field.split("|"):
                    cui = cui.strip()
                    if cui:
                        cui_to_ids[cui].append(disease_id)
    return cui_to_ids


def load_drugcentral_treats(driver, cui_to_ids):
    """Load drugTreatsDisease from DrugCentral TSV."""
    log.info("=" * 50)
    log.info("DrugCentral drugTreatsDisease")
    log.info("=" * 50)

    struct_to_db = {}
    with open("data/processed/drugcentral/drugs.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["struct_id"]
            dbid = row.get("drugbank_id") or row.get("canonical_drugbank_id", "")
            if sid and dbid:
                struct_to_db[sid] = dbid.strip()
    log.info(f"struct->drugbank mapping: {len(struct_to_db)}")

    batch = []
    skipped_no_db = 0
    skipped_no_disease = 0
    unmatched_cuis = set()

    with open("data/processed/drugcentral/drug_treats_disease.tsv") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            drugbank_id = struct_to_db.get(row["struct_id"])
            if not drugbank_id:
                skipped_no_db += 1
                continue

            disease_ids = cui_to_ids.get(row["disease_id"], [])
            if not disease_ids:
                skipped_no_disease += 1
                unmatched_cuis.add(row["disease_id"])
                continue

            drug_graph_id = f"drug_{drugbank_id.lower()}"
            for did in disease_ids:
                batch.append({"drug_id": drug_graph_id, "disease_id": did})

    log.info(f"Edges to load: {len(batch)}")
    log.info(f"Skipped (no DrugBank ID): {skipped_no_db}")
    log.info(f"Skipped (no Disease match): {skipped_no_disease}")
    if unmatched_cuis:
        log.info(f"Unmatched CUIs: {unmatched_cuis}")

    total = 0
    for i in range(0, len(batch), 1000):
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $batch AS row
                MATCH (d:Drug {id: row.drug_id})
                MATCH (dis:Disease {id: row.disease_id})
                CREATE (d)-[:drugTreatsDisease {source: "DrugCentral"}]->(dis)
                RETURN count(*) AS created
                """,
                batch=batch[i : i + 1000],
            )
            total += result.single()["created"]

    log.info(f"Loaded {total} drugTreatsDisease edges (DrugCentral)\n")
    return total


def load_clinical_trials_treats(driver):
    """Derive drugTreatsDisease from ClinicalTrials Phase 3/4 trials.

    Logic: If a drug is tested in a Phase 3 or 4 trial for a disease,
    that's strong evidence of a treatment relationship.
    """
    log.info("=" * 50)
    log.info("ClinicalTrials.gov drugTreatsDisease (Phase 3/4)")
    log.info("=" * 50)

    with driver.session() as s:
        result = s.run(
            """
            MATCH (t:ClinicalTrial)-[:STUDIES_CONDITION]->(dis:Disease)
            MATCH (t)-[:TESTS_INTERVENTION]->(drug:Drug)
            WHERE t.phase IN ['PHASE3', 'PHASE4', 'PHASE2|PHASE3']
            WITH DISTINCT drug, dis
            RETURN drug.id AS drug_id, dis.id AS disease_id
            """
        )
        pairs = [{"drug_id": rec["drug_id"], "disease_id": rec["disease_id"]} for rec in result]

    log.info(f"Unique Phase 3/4 Drug-Disease pairs: {len(pairs)}")

    # Filter out pairs that already have a drugTreatsDisease edge
    with driver.session() as s:
        result = s.run(
            """
            MATCH (d:Drug)-[:drugTreatsDisease]->(dis:Disease)
            RETURN d.id AS drug_id, dis.id AS disease_id
            """
        )
        existing = {(rec["drug_id"], rec["disease_id"]) for rec in result}

    new_pairs = [p for p in pairs if (p["drug_id"], p["disease_id"]) not in existing]
    log.info(f"Already have drugTreatsDisease: {len(pairs) - len(new_pairs)}")
    log.info(f"New edges to load: {len(new_pairs)}")

    total = 0
    for i in range(0, len(new_pairs), 1000):
        with driver.session() as s:
            result = s.run(
                """
                UNWIND $batch AS row
                MATCH (d:Drug {id: row.drug_id})
                MATCH (dis:Disease {id: row.disease_id})
                CREATE (d)-[:drugTreatsDisease {source: "ClinicalTrials.gov", evidence: "Phase3/4"}]->(dis)
                RETURN count(*) AS created
                """,
                batch=new_pairs[i : i + 1000],
            )
            total += result.single()["created"]

    log.info(f"Loaded {total} drugTreatsDisease edges (ClinicalTrials)\n")
    return total


def main():
    driver = get_driver()
    try:
        # Clean slate
        with driver.session() as s:
            r = s.run("MATCH ()-[r:drugTreatsDisease]->() DELETE r RETURN count(r)")
            deleted = r.single()[0]
            if deleted:
                log.info(f"Cleared {deleted} existing drugTreatsDisease edges\n")

        cui_to_ids = build_cui_mapping(driver)
        log.info(f"CUI mapping: {len(cui_to_ids)} unique CUIs\n")

        dc_count = load_drugcentral_treats(driver, cui_to_ids)
        ct_count = load_clinical_trials_treats(driver)

        log.info("=" * 50)
        log.info("Final verification")
        log.info("=" * 50)
        with driver.session() as s:
            r = s.run(
                """
                MATCH ()-[r:drugTreatsDisease]->()
                RETURN r.source AS source, count(r) AS c
                ORDER BY c DESC
                """
            )
            total = 0
            for rec in r:
                log.info(f"  {rec['source']}: {rec['c']} edges")
                total += rec["c"]
            log.info(f"  TOTAL: {total} drugTreatsDisease edges")

    finally:
        driver.close()


if __name__ == "__main__":
    main()
