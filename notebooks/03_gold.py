# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold
# MAGIC Builds the business-ready tables:
# MAGIC - `household_features`: one row per household (Phase 3 segmentation)
# MAGIC - `product_week_panel`: one row per product per week with traffic-weighted display/mailer share (Phase 2)
# MAGIC - `campaign_household`: one row per household per campaign, features strictly before the campaign start (Phase 4)
# MAGIC - `dq_summary`: DQ log counts for the dashboard

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

results = pipeline.run_gold(spark, io, run_id, settings)
pipeline.record_run(spark, io, run_id, "succeeded")
display(spark.createDataFrame([r.__dict__ for r in results]))
