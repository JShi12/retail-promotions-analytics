# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Data-quality report
# MAGIC **Goal:** show whether the latest successful run can be trusted: which table-level checks passed, how many rows each data-quality (DQ) rule touched, and how big each table is.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Check*: a table-level pass/fail test. Severity `error` stops the run; `warning` is recorded only.
# MAGIC - *DQ rule action*: `flagged` (row kept, problem noted), `modified` (value changed, e.g. a blank set to null), `dropped` (exact duplicate removed).
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Latest successful run
# MAGIC 2. Checks, failed first
# MAGIC 3. Rows flagged, modified or dropped
# MAGIC 4. Table sizes by layer

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Latest successful run

# COMMAND ----------

dbutils.widgets.text("catalog", "retail")
catalog = dbutils.widgets.get("catalog")
latest = spark.sql(
    f"SELECT run_id FROM {catalog}.ops.pipeline_runs WHERE status = 'succeeded' "
    "ORDER BY finished_at DESC LIMIT 1"
).first()["run_id"]
print("latest run:", latest)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Checks, failed first

# COMMAND ----------

display(
    spark.sql(f"""
    SELECT layer, table_name, check_id, severity, passed, observed, expected
    FROM {catalog}.ops.dq_check_results WHERE run_id = '{latest}'
    ORDER BY passed, layer, table_name, check_id""")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Rows flagged, modified or dropped

# COMMAND ----------

display(
    spark.sql(f"""
    SELECT table_name, rule_id, action, n_rows
    FROM {catalog}.gold.dq_summary WHERE run_id = '{latest}'
    ORDER BY table_name, n_rows DESC""")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Table sizes by layer

# COMMAND ----------

rows = []
for layer in ["bronze", "silver", "gold"]:
    for t in spark.sql(f"SHOW TABLES IN {catalog}.{layer}").collect():
        rows.append((layer, t.tableName, spark.table(f"{catalog}.{layer}.{t.tableName}").count()))
display(spark.createDataFrame(rows, "layer string, table_name string, rows long"))
