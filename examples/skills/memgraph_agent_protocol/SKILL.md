---
name: memgraph_agent_protocol
description: Protocol for the memgraph agent: exporting the populated ontology to Memgraph
tools:
  - get_pipeline_status
  - run_export_graph
  - list_output_files
  - read_output_file
  - post_feedback
---

## Your role
You run the `export_graph` step, which converts the populated RDF ontology into
typed CSV files and a Cypher LOAD CSV import script for Memgraph.

## Running export
1. Confirm `populate` step is completed: call `get_pipeline_status()`.
2. Call `run_export_graph()`.
3. Verify outputs: call `list_output_files()` — expect nodes_*.csv, edges_*.csv,
   and import.cypher.

## Validating the Cypher script
Call `read_output_file("import.cypher")` and check:
- Each node type has a CREATE CONSTRAINT or CREATE INDEX statement.
- LOAD CSV paths match the actual CSV filenames in data/output/.
- Node labels match the ontology class names.

## If export_graph fails
Check whether `data/output/alzkb_v2_populated.rdf` exists.
If not, the populate step did not run or failed — post feedback to mapping_agent.
