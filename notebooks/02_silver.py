# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver
# MAGIC **Goal:** turn bronze into typed, deduplicated tables where every data-quality (DQ) problem is flagged and logged rather than silently fixed or dropped.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Flag, don't drop*: only exact duplicate rows are removed. Every other problem row is kept, and its rule names are added to the row's `dq_flags` array.
# MAGIC - *DQ log (`ops.dq_log`)*: one row per flagged, modified or dropped record, with the rule and the reason.
# MAGIC - *OR rule*: `causal_data` lists 15,245 product-store-weeks twice with conflicting codes. They are collapsed to one row that counts as on display (or in the mailer) if either source row says so.
# MAGIC - *Checks*: table-level pass/fail tests (row reconciliation, unique keys, valid references, documented codes). An error-level failure stops the run.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Settings and run ID
# MAGIC 2. Build silver and run checks
# MAGIC 3. DQ log counts for this run

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Settings and run ID

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from retail_promo import pipeline
from retail_promo.config import Settings
from retail_promo.io import UnityCatalogIO

dbutils.widgets.text("catalog", "retail")
dbutils.widgets.text("run_id", "")  # blank = the run started by 01_bronze
settings = Settings(catalog=dbutils.widgets.get("catalog"))
io = UnityCatalogIO(spark, settings.catalog, settings.landing_dir)
run_id = dbutils.widgets.get("run_id") or pipeline.latest_started_run(io)
print("run_id:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Build silver and run checks

# COMMAND ----------

results = pipeline.run_silver(spark, io, run_id, settings)
display(spark.createDataFrame([r.__dict__ for r in results]))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. DQ log counts for this run
# MAGIC Rows per table, rule and action (`flagged`, `modified`, `dropped`).

# COMMAND ----------

display(
    spark.table(f"{settings.catalog}.ops.dq_log")
    .where(f"run_id = '{run_id}'")
    .groupBy("table_name", "rule_id", "action")
    .count()
    .orderBy("table_name", "rule_id")
)
