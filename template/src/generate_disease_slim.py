"""
generate_disease_slim.py

Generates data/raw/diseaseontology/slim-terms.tsv by combining:
  1. TopNodes_DOcancerslim from doid.obo (broad cancer categories)
  2. MONDO terms with >30 unique GWAS Catalog studies, mapped to DOID
     via xref fields in mondo.obo (downloaded if absent)

Run from project root:
  python src/generate_disease_slim.py

Output columns: doid, name, source, pathophysiology
  - TopNodes_DOcancerslim terms: source=TopNodes_DOcancerslim, pathophysiology=neoplastic
  - GWAS-derived terms: source=GWAS_Catalog, pathophysiology="" (blank)
  - GWAS terms also in cancer slim keep the cancer slim values (neoplastic)
"""

import csv
import io
import re
import sys
from collections import defaultdict
from pathlib import Path

import obonet
import requests

DOID_OBO = Path("data/raw/diseaseontology/doid.obo")
MONDO_OBO = Path("data/raw/mondo/mondo.obo")
MONDO_OBO_URL = "http://purl.obolibrary.org/obo/mondo.obo"
OUTPUT = Path("data/raw/diseaseontology/slim-terms.tsv")
GWAS_URL = "https://www.ebi.ac.uk/gwas/api/search/downloads/studies/v1.0.3.1"
MIN_STUDIES = 20


def load_doid_graph():
    return obonet.read_obo(DOID_OBO)


def load_mondo_graph():
    if not MONDO_OBO.exists():
        print(f"Downloading mondo.obo from {MONDO_OBO_URL} ...", file=sys.stderr)
        MONDO_OBO.parent.mkdir(parents=True, exist_ok=True)
        resp = requests.get(MONDO_OBO_URL, timeout=300, stream=True)
        resp.raise_for_status()
        with open(MONDO_OBO, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  Saved to {MONDO_OBO}", file=sys.stderr)
    return obonet.read_obo(MONDO_OBO)


def topnodes_cancer_slim(doid_graph):
    """Return {doid: name} for TopNodes_DOcancerslim subset."""
    return {
        node_id: data.get("name", "")
        for node_id, data in doid_graph.nodes(data=True)
        if node_id.startswith("DOID:")
        and "TopNodes_DOcancerslim" in data.get("subset", [])
    }


def mondo_to_doid_map(mondo_graph):
    """
    Build {MONDO_XXXXXXX: [DOID:A, DOID:B, ...]} from DOID xrefs in mondo.obo.

    MONDO terms carry xref: DOID:XXXXXXX entries (obonet returns them as plain
    strings). One MONDO may map to multiple DOIDs; all are kept so downstream
    code can add every corresponding DOID to the disease slim.
    """
    mapping = defaultdict(list)
    for node_id, data in mondo_graph.nodes(data=True):
        if not node_id.startswith("MONDO:"):
            continue
        # MONDO:XXXXXXX → MONDO_XXXXXXX to match GWAS URI extraction
        mondo_key = node_id.replace("MONDO:", "MONDO_")
        for xref in data.get("xref", []):
            if xref.startswith("DOID:"):
                mapping[mondo_key].append(xref)

    return dict(mapping)


def gwas_mondo_study_counts():
    """
    Download GWAS Catalog studies file and return {MONDO_XXXXXXX: study_count}.
    Streams line-by-line to avoid loading ~tens of MB into memory at once.
    """
    print(f"Downloading GWAS Catalog from {GWAS_URL} ...", file=sys.stderr)
    resp = requests.get(GWAS_URL, timeout=120, stream=True)
    resp.raise_for_status()

    study_sets = defaultdict(set)
    header = None
    trait_col = None
    study_col = None

    # Wrap raw socket as a text stream; decode line by line
    text_stream = io.TextIOWrapper(resp.raw, encoding="utf-8", errors="replace")
    reader = csv.reader(text_stream, delimiter="\t")

    for row in reader:
        if header is None:
            header = row
            try:
                trait_col = header.index("MAPPED_TRAIT_URI")
                study_col = header.index("STUDY ACCESSION")
            except ValueError as exc:
                raise RuntimeError(
                    f"Expected GWAS columns not found in header: {header}"
                ) from exc
            continue

        if len(row) <= max(trait_col, study_col):
            continue

        study_acc = row[study_col].strip()
        if not study_acc:
            continue

        for uri in row[trait_col].split(","):
            m = re.search(r"MONDO_(\d+)", uri.strip())
            if m:
                study_sets[f"MONDO_{m.group(1)}"].add(study_acc)

    return {k: len(v) for k, v in study_sets.items()}


def main():
    print("Loading doid.obo ...", file=sys.stderr)
    doid_graph = load_doid_graph()

    cancer_terms = topnodes_cancer_slim(doid_graph)
    print(f"TopNodes_DOcancerslim: {len(cancer_terms)} terms", file=sys.stderr)

    print("Loading mondo.obo ...", file=sys.stderr)
    mondo_graph = load_mondo_graph()

    print("Building MONDO→DOID xref map ...", file=sys.stderr)
    m2d = mondo_to_doid_map(mondo_graph)
    total_doids = sum(len(v) for v in m2d.values())
    print(f"  {len(m2d)} MONDO terms → {total_doids} DOID entries", file=sys.stderr)

    gwas_counts = gwas_mondo_study_counts()
    frequent = {k: v for k, v in gwas_counts.items() if v > MIN_STUDIES}
    print(
        f"MONDO terms with >{MIN_STUDIES} unique studies: {len(frequent)}",
        file=sys.stderr,
    )

    # Map frequent MONDO terms to all corresponding DOIDs
    gwas_doids = {}  # {doid: name}
    unmapped = []
    for mondo_id, count in sorted(frequent.items(), key=lambda x: -x[1]):
        doids = m2d.get(mondo_id, [])
        if doids:
            for doid in doids:
                if doid in doid_graph.nodes:
                    gwas_doids[doid] = doid_graph.nodes[doid].get("name", "")
        else:
            unmapped.append((mondo_id, count))

    if unmapped:
        print(
            f"  {len(unmapped)} MONDO terms could not be mapped to any DOID:",
            file=sys.stderr,
        )
        for mid, cnt in unmapped[:15]:
            print(f"    {mid} ({cnt} studies)", file=sys.stderr)

    # Build rows: GWAS first, cancer slim overwrites overlapping DOIDs
    rows = {}
    for doid, name in gwas_doids.items():
        rows[doid] = {
            "doid": doid,
            "name": name,
            "source": "GWAS_Catalog",
            "pathophysiology": "",
        }

    for doid, name in cancer_terms.items():
        rows[doid] = {
            "doid": doid,
            "name": name,
            "source": "TopNodes_DOcancerslim",
            "pathophysiology": "neoplastic",
        }

    print(f"Total unique disease terms: {len(rows)}", file=sys.stderr)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["doid", "name", "source", "pathophysiology"],
            delimiter="\t",
        )
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda r: r["doid"]):
            writer.writerow(row)

    print(f"Wrote {len(rows)} terms to {OUTPUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
