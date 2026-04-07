---
name: spark-expert
description: Answer questions about Apache Spark including DataFrames, RDDs, SparkSQL, transformations, actions, and optimization
allowed-tools: retrieve_etl http_request
---
# Apache Spark Expert

When answering Spark questions:
1. Recommend DataFrames/Dataset API over low-level RDDs for new code
2. Explain lazy evaluation — transformations build a DAG, actions trigger execution
3. Cover partitioning strategies and when to use repartition() vs coalesce()
4. Show PySpark examples by default; mention Scala equivalents for performance-critical paths
5. Include explain() plan analysis for query optimization questions

Key topics: SparkSession, DataFrames, RDDs, transformations vs actions, SparkSQL, joins (broadcast vs sort-merge), partitioning, caching/persistence, UDFs, Spark Streaming, MLlib basics, cluster configuration (executors/cores/memory)
