# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver
# MAGIC Types every column with `try_cast`, drops exact duplicates, flags every other problem row (flag, don't drop), and writes one `ops.dq_log` row per flagged, modified or dropped record. Conflicting `causal_data` duplicates are collapsed by the OR rule. Error-level checks stop the run.

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

results = pipeline.run_silver(spark, io, run_id, settings)
display(spark.createDataFrame([r.__dict__ for r in results]))

# COMMAND ----------

display(
    spark.table(f"{settings.catalog}.ops.dq_log")
    .where(f"run_id = '{run_id}'")
    .groupBy("table_name", "rule_id", "action")
    .count()
    .orderBy("table_name", "rule_id")
)
