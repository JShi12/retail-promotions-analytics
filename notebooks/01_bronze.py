# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze
# MAGIC Loads the 8 raw CSVs from the landing volume as-is (every column a string) plus load metadata, then checks that row counts match the raw files.

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

pipeline.record_run(spark, io, run_id, "started")
results = pipeline.run_bronze(spark, io, run_id)
display(spark.createDataFrame([r.__dict__ for r in results]))
