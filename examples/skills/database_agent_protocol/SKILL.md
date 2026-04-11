---
name: database_agent_protocol
description: Protocol for the database agent: evaluating and enabling data sources in databases.yaml
tools:
  - get_my_feedback
  - check_env_vars
  - list_parser_classes
  - read_databases_config
  - write_databases_config
  - post_feedback
---

## Your role
You own `config/databases.yaml`. You decide which data sources are enabled and how
they are accessed.

## Before enabling a source
1. Call `get_my_feedback("database_agent")` — check for `source_unavailable` feedback.
2. Call `check_env_vars()` — confirm required credentials are present before
   enabling sources that need authentication (drugbank, disgenet, aopdb).
3. Call `list_parser_classes()` — the `parser_class` value must appear in this list.

## disease_scope_mode values
- `none`: parser is disease-agnostic (GO, NCBI Gene, Uberon) — always safe to enable.
- `api_query`: parser queries the API with terms from project.yaml — requires API key.
- `post_filter`: downloads all data then filters — large download, use carefully.

## Changing enabled status
Set `enabled: true` or `enabled: false` in databases.yaml. Never delete entries.
