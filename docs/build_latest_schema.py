#!/usr/bin/env python3
"""
Build a self-contained version of baseagent_source_schema_template.html that works
when opened directly from the filesystem (file:// protocol).

Usage:
    python docs/build_latest_schema.py

Output:
    docs/baseagent_source_schema_latest.html
"""

import csv
import json
from pathlib import Path

DOCS = Path(__file__).parent
CSV_PATH = DOCS / "baseagent_databases.csv"
TEMPLATE_PATH = DOCS / "baseagent_source_schema_template.html"
OUTPUT_PATH = DOCS / "baseagent_source_schema_latest.html"

PLACEHOLDER = "const DB_DATA = null; // @INJECT"

# Must mirror NODE_LABEL_TO_ID / EDGE_LABEL_TO_ID in baseagent_source_schema_template.html
NODE_LABEL_TO_ID = {
    "Gene":                "nt_Gene",
    "Disease":             "nt_Disease",
    "Drug":                "nt_Drug",
    "Chemical":            "nt_Drug",   # Drug and Chemical are synonymous in this ontology
    "Pathway":             "nt_Pathway",
    "TranscriptionFactor": "nt_TranscriptionFactor",
    "BiologicalProcess":   "nt_BiologicalProcess",
    "MolecularFunction":   "nt_MolecularFunction",
    "CellularComponent":   "nt_CellularComponent",
    "BodyPart":            "nt_BodyPart",
    "Symptom":             "nt_Symptom",
    "DrugClass":           "nt_DrugClass",
}

EDGE_LABEL_TO_ID = {
    "chemicalAssociatesWithDisease":        "et_CAD",
    "chemicalBindsGene":                    "et_CBG",
    "chemicalCausesEffect":                 "et_CCE",
    "chemicalIncreasesExpression":          "et_CIE",
    "chemicalDecreasesExpression":          "et_CDE",
    "diseaseLocalizesToAnatomy":            "et_DLA",
    "diseaseAssociatesWithDisease":         "et_DAD",
    "drugInClass":                          "et_DIC",
    "drugTreatsDisease":                    "et_DTD",
    "geneAssociatesWithDisease":            "et_GAD",
    "geneCovariesWithGene":                 "et_GCG",
    "geneHasMolecularFunction":             "et_GMF",
    "geneInPathway":                        "et_GIP",
    "geneInteractsWithGene":                "et_GIG",
    "geneParticipatesInBiologicalProcess":  "et_GBP",
    "geneAssociatedWithCellularComponent":  "et_GCC",
    "geneRegulatesGene":                    "et_GRG",
    "symptomManifestationOfDisease":        "et_SMD",
    "transcriptionFactorInteractsWithGene": "et_TFG",
    "pathwayContainsGene":                  "et_PCG",
    "anatomyExpressesGene":                 "et_AEG",
}


def transform_row(row: dict) -> dict | None:
    """Apply the same transformation as parseCSV() in baseagent_source_schema_template.html."""
    db_id = row.get("ID", "").strip()
    if not db_id:
        return None

    raw_nodes = row.get("Biomedical Entities (Node types)", "").strip()
    raw_edges = row.get("Biomedical Relationships (Edge types)", "").strip()

    nodes = [NODE_LABEL_TO_ID[s.strip()] for s in raw_nodes.split(",")
             if raw_nodes and s.strip() in NODE_LABEL_TO_ID]
    edges = [EDGE_LABEL_TO_ID[s.strip()] for s in raw_edges.split(",")
             if raw_edges and s.strip() in EDGE_LABEL_TO_ID]

    return {
        "id":          db_id,
        "label":       row.get("Label", "").strip(),
        "integration": row.get("Integration Path", "").strip(),
        "active":      row.get("Active", "").strip() == "Yes",
        "status":      row.get("Status", "").strip() or "N/A",
        "version":     row.get("Latest Version", "").strip() or "N/A",
        "parent":      row.get("Sub-source Of", "").strip() or None,
        "nodes":       nodes,
        "edges":       edges,
    }


def main():
    with CSV_PATH.open(newline="", encoding="utf-8") as f:
        dbs = [r for row in csv.DictReader(f) if (r := transform_row(row))]

    template = TEMPLATE_PATH.read_text(encoding="utf-8")

    if PLACEHOLDER not in template:
        raise RuntimeError(f"Placeholder not found in template: {PLACEHOLDER!r}")

    injected = f"const DB_DATA = {json.dumps(dbs, ensure_ascii=False, indent=2)};"
    patched = template.replace(PLACEHOLDER, injected, 1)

    OUTPUT_PATH.write_text(patched, encoding="utf-8")
    print(f"Written: {OUTPUT_PATH}  ({len(dbs)} databases)")


if __name__ == "__main__":
    main()
