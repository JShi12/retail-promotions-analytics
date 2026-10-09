# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Data-quality report
# MAGIC Latest run: table-level check results, row-level DQ log counts, and table sizes per layer.

# COMMAND ----------

dbutils.widgets.text("catalog", "retail")
catalog = dbutils.widgets.get("catalog")
latest = spark.sql(
    f"SELECT run_id FROM {catalog}.ops.pipeline_runs WHERE status = 'succeeded' ORDER BY finished_at DESC LIMIT 1"
).first()["run_id"]
print("latest run:", latest)

# COMMAND ----------

# MAGIC %md ## Checks (failed first)

# COMMAND ----------

display(
    spark.sql(f"""
    SELECT layer, table_name, check_id, severity, passed, observed, expected
    FROM {catalog}.ops.dq_check_results WHERE run_id = '{latest}'
    ORDER BY passed, layer, table_name, check_id""")
)

# COMMAND ----------

# MAGIC %md ## Rows flagged, modified or dropped

# COMMAND ----------

display(
    spark.sql(f"""
    SELECT table_name, rule_id, action, n_rows
    FROM {catalog}.gold.dq_summary WHERE run_id = '{latest}'
    ORDER BY table_name, n_rows DESC""")
)

# COMMAND ----------

# MAGIC %md ## Table sizes

# COMMAND ----------

rows = []
for layer in ["bronze", "silver", "gold"]:
    for t in spark.sql(f"SHOW TABLES IN {catalog}.{layer}").collect():
        rows.append((layer, t.tableName, spark.table(f"{catalog}.{layer}.{t.tableName}").count()))
display(spark.createDataFrame(rows, "layer string, table_name string, rows long"))
