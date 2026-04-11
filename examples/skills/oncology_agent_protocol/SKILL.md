---
name: oncology_agent_protocol
description: Protocol for the oncology agent: how to define disease scope and maintain the OWL ontology
tools:
  - get_my_feedback
  - inspect_ontology_types
  - read_project_config
  - write_project_config
  - post_feedback
---

## Your role
You own `config/project.yaml`. Your job is to define the disease scope and ensure the
base OWL ontology covers all node and edge types required by the pipeline.

## Disease scope fields
- `primary_terms`: lowercase search strings used by API-based parsers (DisGeNET).
- `umls_cuis`, `doid_ids`, `mesh_ids`: cross-references used by parsers that filter
  by disease identifier.
- `drug_names`: known drugs for this disease, used by DrugBank post-filtering.

## Before editing project.yaml
1. Call `get_my_feedback("oncology_agent")` — check for `missing_entity` feedback
   from the mapping agent.
2. Call `inspect_ontology_types()` to see what classes and properties already exist.
3. Only modify `project.yaml`; never edit Python parser source.

## After editing
Call `post_feedback("oncology_agent", "mapping_agent", "general", "info",
"project.yaml updated — re-validate ontology mappings")`.
