# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze
# MAGIC **Goal:** load the 8 raw CSV files exactly as received into bronze Delta tables, so every later number can be traced back to a raw row.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Bronze*: raw data with every column kept as a string (nothing converted silently), column names lower-cased, plus load metadata (`_source_file`, `_ingested_at`, `_run_id`).
# MAGIC - *Run ID (`run_id`)*: one identifier per pipeline run, stamped on every table and log row. This notebook starts a new run; `02_silver` and `03_gold` pick it up automatically.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Settings and run ID
# MAGIC 2. Load bronze and check row counts against the raw files

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
dbutils.widgets.text("run_id", "")  # a job passes one id to every task; blank = new id
settings = Settings(catalog=dbutils.widgets.get("catalog"))
run_id = dbutils.widgets.get("run_id") or pipeline.new_run_id()
io = UnityCatalogIO(spark, settings.catalog, settings.landing_dir)
print("run_id:", run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Load bronze and check row counts against the raw files
# MAGIC One check per table: bronze rows must equal the data lines in the raw file. A failure stops the run.

# COMMAND ----------

pipeline.record_run(spark, io, run_id, "started")
results = pipeline.run_bronze(spark, io, run_id)
display(spark.createDataFrame([r.__dict__ for r in results]))
