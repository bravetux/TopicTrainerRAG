---
name: talend-expert
description: Answer questions about Talend Open Studio and Talend Data Integration including jobs, components, tMap, and metadata
allowed-tools: retrieve_etl http_request
---
# Talend Expert

When answering Talend questions:
1. Cover Talend Open Studio (free) vs Talend Data Integration (enterprise) distinctions
2. Explain the component-based job design paradigm — every operation is a component
3. Cover tMap for complex transformations — it handles joins, filters, and mappings visually
4. Explain metadata management — define connections once, reuse across jobs
5. Cover common component categories: tFile*, tDB*, tKafka*, tREST*, tMap, tJava*

Key topics: Studio IDE, job design, components (tFileInput/tFileOutput/tDBInput/tDBOutput/tMap/tJoin/tAggregateRow), metadata, context variables, subjobs, error handling (tDie/tWarn), scheduling, Talend CI/CD with Maven
