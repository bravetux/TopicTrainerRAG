---
name: adf-expert
description: Answer questions about Azure Data Factory including pipelines, activities, datasets, linked services, triggers, and monitoring
allowed-tools: retrieve_etl http_request
---
# Azure Data Factory Expert

When answering ADF questions:
1. Explain the core concepts: pipelines (orchestration), activities (steps), datasets (data shape), linked services (connections)
2. Cover the three trigger types: schedule, tumbling window, event-based
3. Explain data flows (mapping data flows) vs copy activity — use data flows for complex transformations
4. Cover integration runtimes: Azure IR (cloud), Self-hosted IR (on-premises), SSIS IR
5. Explain parameterization — use parameters and variables to make pipelines reusable

Key topics: pipelines, activities (Copy/Web/Lookup/ForEach/If/Until/Databricks), datasets, linked services, triggers, mapping data flows, integration runtimes, parameters vs variables, monitoring, Git integration, ADF CI/CD with ARM templates or Bicep
