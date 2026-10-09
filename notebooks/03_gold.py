# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold
# MAGIC **Goal:** build the business-ready tables that the analysis phases and the dashboard read.
# MAGIC
# MAGIC **Key terms**
# MAGIC - `household_features`: one row per household over the whole period — spend, shopping frequency, category mix, promotion and coupon use, demographics. Used for segmentation (Phase 3).
# MAGIC - `product_week_panel`: one row per product per week (weeks 9–101, products bought in at least 26 weeks), with units sold and promotion exposure. Used for promotion lift (Phase 2).
# MAGIC - *Traffic-weighted share*: `display_share` / `mailer_share` add up the weights of the stores featuring the product that week, where a store's weight is its share of that week's transaction lines.
# MAGIC - `campaign_household`: one row per household per campaign received, with features computed only from purchases **before** the campaign start day and the label `redeemed`. Used for the response model (Phase 4).
# MAGIC - `dq_summary`: data-quality (DQ) log counts per run, table and rule, for the dashboard.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Settings and run ID
# MAGIC 2. Build gold and run checks

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
# MAGIC ## 2. Build gold and run checks
# MAGIC Checks: one row per household / product-week / household-campaign, promotion shares at most 1, and no campaign feature using data on or after the campaign start.

# COMMAND ----------

results = pipeline.run_gold(spark, io, run_id, settings)
pipeline.record_run(spark, io, run_id, "succeeded")
display(spark.createDataFrame([r.__dict__ for r in results]))
