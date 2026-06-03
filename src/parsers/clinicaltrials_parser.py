"""
ClinicalTrials.gov Parser for CardioKB
Loads ClinicalTrial nodes and STUDIES_CONDITION / TESTS_INTERVENTION edges.
Supports: --download --parse --load --all
"""
import argparse, csv, os, sys, time, requests
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:7688"
OUT_DIR = "./data/processed/clinicaltrials"
os.makedirs(OUT_DIR, exist_ok=True)

CVD_CONDITIONS = [
    "heart failure", "coronary artery disease", "myocardial infarction",
    "atrial fibrillation", "cardiomyopathy", "aortic stenosis",
    "hypertension", "stroke", "atherosclerosis", "cardiac arrest",
    "ventricular tachycardia", "pulmonary hypertension", "heart valve disease",
    "endocarditis", "pericarditis", "aortic aneurysm", "peripheral artery disease",
    "deep vein thrombosis", "pulmonary embolism", "congenital heart disease",
    "arrhythmia", "angina pectoris",
]
BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
TRIALS_TSV = os.path.join(OUT_DIR, "trials_new.tsv")
COND_TSV   = os.path.join(OUT_DIR, "trial_conditions.tsv")
INTV_TSV   = os.path.join(OUT_DIR, "trial_interventions.tsv")


def fetch_trials_for_condition(cond, max_pages=5):
    trials, token, page = [], None, 0
    while page < max_pages:
        params = {"query.cond": cond, "pageSize": 200, "format": "json"}
        if token: params["pageToken"] = token
        try:
            r = requests.get(BASE_URL, params=params, timeout=30)
            r.raise_for_status(); data = r.json()
        except Exception as e:
            print(f"  err {cond}/{page}: {e}"); break
        for s in data.get("studies", []):
            p = s.get("protocolSection", {})
            idm = p.get("identificationModule", {})
            sm  = p.get("statusModule", {})
            dm  = p.get("designModule", {})
            cm  = p.get("conditionsModule", {})
            am  = p.get("armsInterventionsModule", {})
            phases = dm.get("phases", [])
            drug_iv = [iv.get("name","") for iv in am.get("interventions", [])
                       if iv.get("type","").upper() in ("DRUG","BIOLOGICAL","COMBINATION_PRODUCT")]
            trials.append({
                "trialId": idm.get("nctId",""),
                "title":   idm.get("briefTitle",""),
                "status":  sm.get("overallStatus",""),
                "phase":   "|".join(phases) if phases else "NA",
                "startDate":      sm.get("startDateStruct", {}).get("date",""),
                "completionDate": sm.get("completionDateStruct", {}).get("date",""),
                "conditions":    "|".join(cm.get("conditions", [])),
                "interventions": "|".join(drug_iv),
                "source": "ClinicalTrials.gov",
            })
        token = data.get("nextPageToken"); page += 1
        if not token: break
        time.sleep(0.3)
    return trials


def download():
    print("[clinicaltrials] download")
    all_t = {}
    for c in CVD_CONDITIONS:
        ts = fetch_trials_for_condition(c)
        for t in ts:
            if t["trialId"]: all_t[t["trialId"]] = t
        print(f"  {c}: total unique {len(all_t)}")
        time.sleep(0.3)
    with open(TRIALS_TSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trialId","title","status","phase","startDate",
                                          "completionDate","conditions","interventions","source"], delimiter="\t")
        w.writeheader(); w.writerows(all_t.values())
    print(f"  wrote {TRIALS_TSV}  rows={len(all_t)}")


def parse():
    print("[clinicaltrials] parse")
    if not os.path.exists(TRIALS_TSV):
        print("  missing trials TSV - run --download first"); return
    with open(TRIALS_TSV) as f, \
         open(COND_TSV,"w",newline="") as cf, \
         open(INTV_TSV,"w",newline="") as ivf:
        rd = csv.DictReader(f, delimiter="\t")
        cw = csv.writer(cf, delimiter="\t"); cw.writerow(["trialId","condition"])
        iw = csv.writer(ivf, delimiter="\t"); iw.writerow(["trialId","intervention"])
        cn = iv_n = 0
        for r in rd:
            tid = r["trialId"]
            for cond in (r.get("conditions") or "").split("|"):
                cond = cond.strip()
                if cond: cw.writerow([tid, cond]); cn += 1
            for ivx in (r.get("interventions") or "").split("|"):
                ivx = ivx.strip()
                if ivx: iw.writerow([tid, ivx]); iv_n += 1
    print(f"  condition rows: {cn}   intervention rows: {iv_n}")


def load():
    print("[clinicaltrials] load")
    drv = GraphDatabase.driver(NEO4J_URI, auth=None)
    with drv.session() as s:
        # Trials
        trials = []
        with open(TRIALS_TSV) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["trialId"]: trials.append(r)
        before_n = s.run("MATCH (n:ClinicalTrial) RETURN count(n) AS c").single()["c"]
        s.run("""
            UNWIND $rows AS r
            MERGE (t:ClinicalTrial {trialId: r.trialId})
            ON CREATE SET t.title=r.title, t.status=r.status, t.phase=r.phase,
                          t.startDate=r.startDate, t.completionDate=r.completionDate,
                          t.source=r.source
        """, rows=trials)
        after_n = s.run("MATCH (n:ClinicalTrial) RETURN count(n) AS c").single()["c"]
        print(f"  ClinicalTrial: {before_n} -> {after_n}  (+{after_n-before_n})")

        # Disease lookup: name + pipe-delimited synonyms (case-insensitive) -> doid
        d_rows = list(s.run("MATCH (d:Disease) RETURN d.xrefDiseaseOntology AS doid, "
                            "coalesce(d.diseaseName, d.name) AS name, d.synonyms AS syn"))
        name_to_doid = {}
        for row in d_rows:
            doid = row["doid"]
            if not doid: continue
            nm = row["name"]
            if nm: name_to_doid[nm.lower()] = doid
            syn = row["syn"]
            if syn:
                for x in str(syn).split("|"):
                    x = x.strip()
                    if x: name_to_doid[x.lower()] = doid

        import re
        _WORD_SUBS = [
            ("arterial", "artery"), ("venous", "vein"), ("cerebral", "brain"),
            ("hepatic", "liver"), ("renal", "kidney"), ("pulmonary", "lung"),
            ("cardiac", "heart"), ("coronary", "coronary artery"),
        ]
        def _resolve(cond):
            key = cond.lower().strip()
            candidates = [key]
            for old, new in _WORD_SUBS:
                if old in key: candidates.append(key.replace(old, new))
            stripped = re.sub(r"\s*\([^)]*\)\s*$", "", key).strip()
            if stripped != key: candidates.append(stripped)
            if "," in key:
                parts = [p.strip() for p in key.split(",", 1)]
                candidates.append(f"{parts[1]} {parts[0]}")
            expanded = []
            for c in candidates:
                expanded.append(c)
                if c.endswith("s"): expanded.append(c[:-1])
                else: expanded.append(c + "s")
            for c in expanded:
                d = name_to_doid.get(c)
                if d: return d
            return None

        # STUDIES_CONDITION
        cond_rows = []
        with open(COND_TSV) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                doid = _resolve(r["condition"] or "")
                if doid:
                    cond_rows.append({"trialId": r["trialId"], "doid": doid})
        before_e = s.run("MATCH ()-[r:STUDIES_CONDITION]->() RETURN count(r) AS c").single()["c"]
        s.run("""
            UNWIND $rows AS r
            MATCH (t:ClinicalTrial {trialId: r.trialId})
            MATCH (d:Disease {xrefDiseaseOntology: r.doid})
            MERGE (t)-[e:STUDIES_CONDITION]->(d)
            ON CREATE SET e.source='ClinicalTrials.gov'
            ON MATCH  SET e.source=coalesce(e.source,'ClinicalTrials.gov')
        """, rows=cond_rows)
        after_e = s.run("MATCH ()-[r:STUDIES_CONDITION]->() RETURN count(r) AS c").single()["c"]
        print(f"  STUDIES_CONDITION: {before_e} -> {after_e}  (+{after_e-before_e})  (input rows: {len(cond_rows)})")

        # TESTS_INTERVENTION - match Drug by commonName (case-insensitive)
        dr_rows = list(s.run("MATCH (d:Drug) WHERE d.commonName IS NOT NULL "
                             "RETURN d.commonName AS nm"))
        name_to_drug = {row["nm"].lower(): row["nm"] for row in dr_rows if row["nm"]}
        iv_rows = []
        with open(INTV_TSV) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                nm = name_to_drug.get((r["intervention"] or "").lower().strip())
                if nm:
                    iv_rows.append({"trialId": r["trialId"], "name": nm})
        before_t = s.run("MATCH ()-[r:TESTS_INTERVENTION]->() RETURN count(r) AS c").single()["c"]
        s.run("""
            UNWIND $rows AS r
            MATCH (t:ClinicalTrial {trialId: r.trialId})
            MATCH (d:Drug {commonName: r.name})
            MERGE (t)-[e:TESTS_INTERVENTION]->(d)
            ON CREATE SET e.source='ClinicalTrials.gov'
            ON MATCH  SET e.source=coalesce(e.source,'ClinicalTrials.gov')
        """, rows=iv_rows)
        after_t = s.run("MATCH ()-[r:TESTS_INTERVENTION]->() RETURN count(r) AS c").single()["c"]
        print(f"  TESTS_INTERVENTION: {before_t} -> {after_t}  (+{after_t-before_t})  (input rows: {len(iv_rows)})")
    drv.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--parse",    action="store_true")
    ap.add_argument("--load",     action="store_true")
    ap.add_argument("--all",      action="store_true")
    a = ap.parse_args()
    if a.all or a.download: download()
    if a.all or a.parse:    parse()
    if a.all or a.load:     load()
    if not any([a.all,a.download,a.parse,a.load]):
        print("usage: --download | --parse | --load | --all")
