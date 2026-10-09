# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Creates the Unity Catalog objects the pipeline writes to and checks that the 8 raw CSVs are in the landing volume.
# MAGIC
# MAGIC Run once. Then upload the CSVs (Catalog → `raw` schema → `landing` volume → *Upload to this volume*) and re-run the last cell.

# COMMAND ----------

dbutils.widgets.text("catalog", "retail")
catalog = dbutils.widgets.get("catalog")

# COMMAND ----------

# Free Edition may not allow new catalogs; if this fails, set the widget to the default `workspace` catalog.
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
for schema in ["raw", "bronze", "silver", "gold", "ops"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.raw.landing")
display(spark.sql(f"SHOW SCHEMAS IN {catalog}"))

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
