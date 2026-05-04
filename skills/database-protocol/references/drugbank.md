# DrugBank Operational Reference

---

## Setup

**Requirements:**
- Free academic account at https://go.drugbank.com/users/sign_up
- Set credentials as env vars `DRUGBANK_USERNAME` and `DRUGBANK_PASSWORD`

**databases.yaml entry:**
```yaml
drugbank:
  enabled: true
  args:
    version: "latest"
    username_env: DRUGBANK_USERNAME
    password_env: DRUGBANK_PASSWORD
```

`version` can be `"latest"` or a pinned release tag (e.g. `"5-1-14"`). `source_url` overrides the drug-links CSV download URL; the full XML URL is always `https://go.drugbank.com/releases/{version}/downloads/all-full-database`.

**Two downloads per release:**
- **Full XML** (`all-full-database`): rich drug properties and drug-gene interactions; ~1 GB zip
- **Drug-links CSV** (`all-drug-links`): basic cross-reference identifiers only; no pharmacology text or structural data

Without credentials the source can only be used from previously cached files in `data/raw/drugbank/`.

---

## Source Data

### Full XML (`full_database.xml`)

XML namespace: `http://www.drugbank.ca`. Drug entries are top-level `<drug>` elements with a `type` attribute (`small molecule` or `biotech`). Nested `<drug>` elements (e.g. metabolites) do not carry a `type` attribute.

**Drug node fields:**

| XML location | Field | Notes |
|---|---|---|
| `<drugbank-id primary="true">` | `drugbank_id` | Primary ID; format `DB#####` |
| `<name>` | `name` | Common drug name |
| `<description>` | `drugDescription` | Free text; may contain HTML entities |
| `type` attribute | `drugType` | `small molecule` or `biotech` |
| `<groups><group>` | `drugGroups` | Semicolon-separated: `approved`, `investigational`, `withdrawn`, `experimental`, `nutraceutical`, `illicit`, `vet_approved` |
| `<categories><category><category>` | `drugCategories` | ATC / pharmacological categories |
| `<indication>` | `drugIndication` | Free-text indication prose; not a structured disease identifier |
| `<pharmacodynamics>` | `drugPharmacology` | |
| `<mechanism-of-action>` | `drugMechanism` | |
| `<toxicity>` | `drugToxicity` | |
| `<half-life>` | `drugHalfLife` | Mixed units (no standard format) |
| `<state>` | `drugState` | `solid`, `liquid`, or `gas` |
| `<calculated-properties>` kind=`Molecular Formula` | `molecularFormula` | |
| `<calculated-properties>` kind=`Molecular Weight` | `molecularWeight` | Daltons |
| `<calculated-properties>` kind=`SMILES` | `smiles` | Canonical SMILES |
| `<calculated-properties>` kind=`InChI` | `inchi` | |
| `<calculated-properties>` kind=`InChIKey` | `inchikey` | |
| `<cas-number>` | `cas_number` | CAS Registry Number |
| `<external-identifiers>` resource=`KEGG Drug` | `kegg_drug_id` | |
| `<external-identifiers>` resource=`PubChem Compound` | `pubchem_cid` | |
| `<external-identifiers>` resource=`ChEMBL` | `chembl_id` | |
| `<external-identifiers>` resource=`ChEBI` | `chebi_id` | |

**Drug-gene interaction fields** — drawn from `<targets>`, `<enzymes>`, `<carriers>`, `<transporters>` sections:

| XML location | Field | Notes |
|---|---|---|
| `<polypeptide><gene-name>` | `gene_symbol` | HGNC gene symbol |
| `<polypeptide id="...">` attribute | `uniprot_id` | UniProt accession |
| section tag | `interaction_type` | `target`, `enzyme`, `carrier`, or `transporter` |

### Drug-links CSV (`drugs.csv`)

Lighter export with basic identifiers only. Columns: `DrugBank ID`, `Name`, `CAS Number`, `Drug Type`, `KEGG Compound ID`, `KEGG Drug ID`, `PubChem Compound ID`, `PubChem Substance ID`, `ChEBI ID`, `PharmGKB ID`, `UniProt ID`, `UniProt Title`, `GenBank ID`, `ChEMBL ID`. Does not include pharmacology text, SMILES, InChI, calculated properties, or drug-gene interaction data.

---

## Known Gotchas

**Nested `<drug>` elements** — the XML embeds metabolite entries as `<drug>` children inside parent entries. These lack a `type` attribute. Use presence of `type` to identify top-level drug entries.

**`drugIndication` is free text, not a disease link** — it stores prose, not a structured disease identifier. For structured drug-disease associations use DrugCentral `omop_relationship`.

**External identifier resource names are exact strings** — `"KEGG Drug"`, `"PubChem Compound"`, `"ChEMBL"`, `"ChEBI"` must be matched verbatim from the `<resource>` element. Note: the CSV export uses a separate column `KEGG Compound ID` (distinct from `KEGG Drug ID`).

**`version: "latest"` is not reproducible** — the URL always resolves to the current release. Pin to a specific version tag (e.g. `"5-1-14"`) for reproducible builds.
