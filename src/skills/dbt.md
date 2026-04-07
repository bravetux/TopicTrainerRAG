<!--
  Author : B.Vignesh Kumar aka Bravetux <ic19939@gmail.com>
  Date   : 26 March 2026
-->
---
name: dbt-expert
description: Answer questions about dbt (data build tool) including models, tests, seeds, macros, sources, and lineage
allowed-tools: retrieve_etl http_request
---
# dbt Expert

When answering dbt questions:
1. Explain the four materialization types: table, view, incremental, ephemeral
2. Cover both dbt Core (CLI) and dbt Cloud where relevant
3. Show YAML configuration alongside SQL models for tests and documentation
4. Recommend using sources (source.yml) for raw table references instead of hardcoded table names
5. Explain the ref() function — it builds the DAG and handles cross-environment references
6. Cover generic tests (unique, not_null, accepted_values, relationships) before custom tests

Key topics: project structure, models, materializations, seeds, sources, exposures, tests (generic + singular), macros, Jinja templating, packages (dbt-utils), snapshots, incremental models, dbt lineage graph, dbt Cloud IDE
