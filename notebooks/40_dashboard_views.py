# Databricks notebook source
# MAGIC %md
# MAGIC # 40 · Dashboard views
# MAGIC **Goal:** create the SQL views behind the Databricks AI/BI dashboard, so every chart is a plain `SELECT` over gold and operations (ops) tables and shows the same numbers as the notebooks.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *View*: a saved SQL query that behaves like a table; it stores no data and always reflects the current gold tables. All views live in the `dashboard` schema.
# MAGIC - *Gold tables read*: `weekly_kpis`, `household_features`, `promo_lift_by_category`, `household_segments`, `campaign_household`, `campaign_response_scores`, `dq_summary`; ops tables: `pipeline_runs`, `dq_check_results`.
# MAGIC - *Back-test*: the targeting view scores past campaigns with out-of-fold model scores; it shows how well the model would have targeted them, not whom to target next.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Create the views
# MAGIC 2. Check every view returns rows

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the views
# MAGIC The SQL lives in `src/retail_promo/dashboard/views.py`, where local tests check that the views reproduce the notebooks' calculations (capture curve, segment profiles).

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))

from retail_promo._version import check_fresh
from retail_promo.dashboard import views

dbutils.widgets.text("catalog", "retail")
catalog = dbutils.widgets.get("catalog")
check_fresh()  # stops if this session is running code older than the Git folder

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.dashboard")
created = views.create_views(
    spark, gold=f"{catalog}.gold", ops=f"{catalog}.ops", dashboard=f"{catalog}.dashboard"
)
print(f"{len(created)} views created in {catalog}.dashboard")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check every view returns rows
# MAGIC An empty view means an upstream table is missing or empty; the cell stops if any view is empty.

# COMMAND ----------

counts = [(name, spark.table(f"{catalog}.dashboard.{name}").count()) for name in created]
display(spark.createDataFrame(counts, "view string, rows long"))
empty = [name for name, n in counts if n == 0]
assert not empty, f"Empty views: {empty}"
