# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC **Goal:** create the Unity Catalog objects the pipeline writes to, and confirm the 8 raw dunnhumby CSV files are in place before the first run.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Catalog*: the top level of Unity Catalog's `catalog.schema.table` naming. This project uses `retail`; if Free Edition does not allow creating catalogs, set the `catalog` widget to the default `workspace`.
# MAGIC - *Schemas*: one per pipeline layer (`raw`, `bronze`, `silver`, `gold`) plus `ops` for data-quality (DQ) logs and run records.
# MAGIC - *Volume*: a governed folder for files. The raw CSVs are uploaded to the `raw.landing` volume.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Create the catalog, schemas and volume — run once.
# MAGIC 2. Check the raw files — re-run after uploading the CSVs (Catalog → `raw` → `landing` → *Upload to this volume*).

# COMMAND ----------

dbutils.widgets.text("catalog", "retail")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Create the catalog, schemas and volume

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
for schema in ["raw", "bronze", "silver", "gold", "ops"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.raw.landing")
display(spark.sql(f"SHOW SCHEMAS IN {catalog}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Check the raw files
# MAGIC All 8 files must be present under their original names (`transaction_data.csv`, `causal_data.csv`, ...).

# COMMAND ----------

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "..", "src")))
from retail_promo.config import RAW_FILES

landing = f"/Volumes/{catalog}/raw/landing"
present = {f.name for f in dbutils.fs.ls(landing)}
missing = [f"{stem}.csv" for stem in RAW_FILES if f"{stem}.csv" not in present]
print("Missing files:", missing or "none")
assert not missing, f"Upload these to {landing}: {missing}"
