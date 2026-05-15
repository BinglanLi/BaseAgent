"""
eval_after_parser.py — Tier 1/2/3 metrics computed from processed TSV files.

Metrics implemented (see docs/eval_metrics.md):
  Tier 1: Direct source download, Source database extraction,
          TSV structural integrity, Extracted record counts,
          Filter pass rate, Duplication rate per ontology
  Tier 2: Null/empty field rate per property,
          Identifier format validity rate per namespace,
          Property value constraint violations, Source schema conformance
  Tier 3: Extraction timestamp per source

Output JSON schema (one object per metric):
  name         — metric name from eval_metrics.md
  data_type    — binary | integer | float | date
  tier         — 1, 2, or 3
  result       — the computed value
  source       — data source name (e.g. disgenet)
  mapping      — ontology mapping key, if applicable
  (extra keys) — column, ontology_property, note, etc.

Usage:
    python eval/eval_after_parser.py
    python eval/eval_after_parser.py --output report.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
PROCESSED_DIR = ROOT / "data" / "processed"

# Regex patterns keyed by ontology data_property name.
# Applied to the source column that maps to each property.
IDENTIFIER_PATTERNS: dict[str, str] = {
    "xrefNcbiGene":       r"^\d+$",
    "xrefHGNC":           r"^HGNC:\d+$",
    "xrefEnsembl":        r"^ENSG\d+$",
    "xrefUmlsCUI":        r"^C\d{7}$",
    "xrefDiseaseOntology": r"^\d+(\.\d+)?$",
    "xrefDrugbank":       r"^DB\d{5}$",
    "xrefMeSH":           r"^(D|C)\d+$",
    "xrefDTXSID":         r"^DTXSID\d+$",
    "xrefOMIM":           r"^\d+(\.\d+)?$",
    "xrefUberon":         r"^UBERON:\d+$",
}


def load_configs() -> tuple[dict, dict, dict]:
    project = yaml.safe_load((CONFIG_DIR / "project.yaml").read_text())["project"]
    mappings_raw = yaml.safe_load((CONFIG_DIR / "ontology_mappings.yaml").read_text())
    mappings = mappings_raw.get("mappings", mappings_raw)
    mappings = {k: v for k, v in mappings.items() if v is not None}
    databases = yaml.safe_load((CONFIG_DIR / "databases.yaml").read_text())["databases"]
    return project, mappings, databases


def _metric(name: str, data_type: str, result, tier: int, **kwargs) -> dict:
    entry = {"name": name, "data_type": data_type, "tier": tier, "result": result}
    # exclude None-valued extra fields to keep output clean
    entry.update({k: v for k, v in kwargs.items() if v is not None})
    return entry


def _is_direct_download(db_config: dict) -> bool:
    """True if source fetches from a live API or web download rather than a precomputed file."""
    notes = (db_config or {}).get("notes", "").lower()
    return "github" not in notes


def _apply_filter(df: pd.DataFrame, parse_config: dict) -> pd.DataFrame:
    col = parse_config.get("filter_column")
    val = parse_config.get("filter_value")
    if col and val is not None and col in df.columns:
        return df[df[col].astype(str) == str(val)]
    return df


def _count_bad_rows(tsv_path: Path) -> int:
    """Count rows whose tab-delimited field count differs from the header."""
    with open(tsv_path, "rb") as fh:
        header = fh.readline()
        expected = len(header.rstrip(b"\n").split(b"\t"))
        bad = sum(
            1 for line in fh
            if line.strip() and len(line.rstrip(b"\n").split(b"\t")) != expected
        )
    return bad


def eval_source(source_name: str, mappings: dict, databases: dict) -> list[dict]:
    metrics: list[dict] = []
    db_config = databases.get(source_name, {})

    metrics.append(_metric(
        "Direct source download", "binary",
        _is_direct_download(db_config),
        tier=1, source=source_name,
    ))

    source_mappings = {
        k: v for k, v in mappings.items()
        if k.startswith(source_name + ".") and not v.get("skip", False)
    }

    for mapping_key, mapping in source_mappings.items():
        parse_config = mapping.get("parse_config", {})
        tsv_path = PROCESSED_DIR / source_name / mapping["source_filename"]

        # --- Tier 1: Source database extraction ---
        tsv_exists = tsv_path.exists()
        df = None
        if tsv_exists:
            try:
                df = pd.read_csv(tsv_path, sep="\t", low_memory=False, on_bad_lines="skip")
            except Exception as exc:
                metrics.append(_metric(
                    "Source database extraction", "binary", False,
                    tier=1, source=source_name, mapping=mapping_key,
                    note=f"read error: {exc}",
                ))
                continue

        extraction_pass = df is not None and len(df) > 0
        metrics.append(_metric(
            "Source database extraction", "binary", extraction_pass,
            tier=1, source=source_name, mapping=mapping_key,
        ))

        if df is None:
            continue

        # --- Tier 1: TSV structural integrity ---
        bad_rows = _count_bad_rows(tsv_path)
        metrics.append(_metric(
            "TSV structural integrity", "binary", bad_rows == 0,
            tier=1, source=source_name, mapping=mapping_key,
            note=f"{bad_rows} rows with mismatched field count" if bad_rows else None,
        ))

        # --- Tier 1: Extracted record counts ---
        metrics.append(_metric(
            "Extracted record counts", "integer", len(df),
            tier=1, source=source_name, mapping=mapping_key,
        ))

        # --- Tier 1: Filter pass rate (only when filter is configured) ---
        df_filtered = _apply_filter(df, parse_config)
        if parse_config.get("filter_column") and len(df) > 0:
            metrics.append(_metric(
                "Filter pass rate", "float",
                round(len(df_filtered) / len(df), 4),
                tier=1, source=source_name, mapping=mapping_key,
                note=f"{len(df_filtered)} of {len(df)} rows pass filter",
            ))

        # --- Tier 1: Duplication rate per ontology (nodes only) ---
        if mapping["data_type"] == "node":
            iri_col = parse_config.get("iri_column_name")
            if iri_col and iri_col in df.columns and len(df) > 0:
                valid_mask = df[iri_col].notna() & (
                    df[iri_col].astype(str).str.strip() != ""
                )
                iri_vals = df.loc[valid_mask, iri_col].astype(str).str.strip()
                dup_count = int(iri_vals.duplicated().sum())
                metrics.append(_metric(
                    "Duplication rate per ontology", "float",
                    round(dup_count / len(iri_vals), 4) if len(iri_vals) > 0 else None,
                    tier=1, source=source_name, mapping=mapping_key,
                    iri_column=iri_col, duplicate_count=dup_count,
                ))

        # --- Tier 2: Null/empty field rate per property ---
        data_property_map = parse_config.get("data_property_map", {})
        for src_col, ont_prop in data_property_map.items():
            if src_col not in df.columns:
                continue
            null_mask = df[src_col].isna() | (df[src_col].astype(str).str.strip() == "")
            null_rate = round(float(null_mask.mean()), 4)
            metrics.append(_metric(
                "Null/empty field rate per property", "float", null_rate,
                tier=2, source=source_name, mapping=mapping_key,
                column=src_col, ontology_property=ont_prop,
            ))

        # --- Tier 2: Identifier format validity rate per namespace ---
        for src_col, ont_prop in data_property_map.items():
            if ont_prop not in IDENTIFIER_PATTERNS or src_col not in df.columns:
                continue
            pattern = IDENTIFIER_PATTERNS[ont_prop]
            non_null = df[src_col].dropna().astype(str).str.strip()
            non_null = non_null[non_null != ""]
            if len(non_null) == 0:
                continue
            valid_rate = round(float(non_null.str.match(pattern).mean()), 4)
            metrics.append(_metric(
                "Identifier format validity rate per namespace", "float", valid_rate,
                tier=2, source=source_name, mapping=mapping_key,
                column=src_col, ontology_property=ont_prop, pattern=pattern,
            ))

        # --- Tier 2: Property value constraint violations ---
        # Heuristic: columns where >90% of non-null values are numeric
        # but some values are not numeric are flagged as violations.
        violations = 0
        for col in df.columns:
            if df[col].dtype != object:
                continue
            non_null = df[col].dropna().astype(str).str.strip()
            non_null = non_null[non_null != ""]
            if len(non_null) == 0:
                continue
            numeric_mask = non_null.str.match(r"^-?\d+(\.\d+)?([eE][+-]?\d+)?$")
            if numeric_mask.mean() > 0.9:
                violations += int((~numeric_mask).sum())
        metrics.append(_metric(
            "Property value constraint violations", "integer", violations,
            tier=2, source=source_name, mapping=mapping_key,
        ))

        # --- Tier 2: Source schema conformance ---
        required_cols: set[str] = set(data_property_map.keys())
        if mapping["data_type"] == "node":
            iri_col = parse_config.get("iri_column_name")
            if iri_col:
                required_cols.add(iri_col)
            merge_src = (parse_config.get("merge_column") or {}).get("source_column_name")
            if merge_src:
                required_cols.add(merge_src)
        else:
            for col_key in ("subject_column_name", "object_column_name"):
                col = parse_config.get(col_key)
                if col:
                    required_cols.add(col)
        missing = sorted(required_cols - set(df.columns))
        metrics.append(_metric(
            "Source schema conformance", "binary", len(missing) == 0,
            tier=2, source=source_name, mapping=mapping_key,
            missing_columns=missing if missing else None,
        ))

    # --- Tier 3: Extraction timestamp per source ---
    source_dir = PROCESSED_DIR / source_name
    if source_dir.exists():
        tsvs = list(source_dir.glob("*.tsv"))
        if tsvs:
            latest_mtime = max(p.stat().st_mtime for p in tsvs)
            ts = datetime.fromtimestamp(latest_mtime, tz=timezone.utc).strftime("%Y-%m-%d")
            metrics.append(_metric(
                "Extraction timestamp per source", "date", ts,
                tier=3, source=source_name,
            ))

    return metrics


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compute after-parser evaluation metrics from processed TSV files."
    )
    ap.add_argument("--output", metavar="FILE", help="Write JSON to FILE (default: stdout)")
    args = ap.parse_args()

    _, mappings, databases = load_configs()
    sources = sorted({k.split(".")[0] for k in mappings})

    all_metrics: list[dict] = []
    for source in sources:
        all_metrics.extend(eval_source(source, mappings, databases))

    report = {
        "run_timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "metrics": all_metrics,
    }
    output = json.dumps(report, indent=2, default=str)

    if args.output:
        Path(args.output).write_text(output)
        print(f"Report written to {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    main()
