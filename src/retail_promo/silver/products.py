from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.config import NON_MERCH_COMMODITIES, NON_MERCH_DEPARTMENTS
from retail_promo.silver._typing import cast_columns, trim_strings

TABLE = "products"
KEY = ["product_id"]
HIERARCHY = ["department", "commodity_desc", "sub_commodity_desc"]


def build(bronze: DataFrame, run_id: str) -> tuple[DataFrame, DataFrame]:
    df = dq.init_flags(bronze.drop("_source_file", "_ingested_at", "_run_id"))
    df = cast_columns(df, {"product_id": "int", "manufacturer": "int"})
    df = trim_strings(df, ["brand", "curr_size_of_product", *HIERARCHY])

    blank_hierarchy = F.lit(False)
    for col in HIERARCHY:
        blank_hierarchy = blank_hierarchy | F.coalesce(F.col(col), F.lit("")).eqNullSafe("")
    df = dq.add_flag(df, blank_hierarchy, "blank_hierarchy")
    for col in HIERARCHY:
        df = df.withColumn(
            col, F.when(F.coalesce(F.col(col), F.lit("")) == "", "UNKNOWN").otherwise(F.col(col))
        )

    df = dq.add_flag(df, F.col("curr_size_of_product") == "", "blank_size_to_null")
    df = df.withColumn("curr_size_of_product", F.nullif(F.col("curr_size_of_product"), F.lit("")))

    df = df.withColumn(
        "is_non_merchandise",
        F.col("department").isin(*sorted(NON_MERCH_DEPARTMENTS))
        | F.col("commodity_desc").isin(*sorted(NON_MERCH_COMMODITIES)),
    ).withColumn("is_private_label", F.col("brand") == "Private")

    return df, dq.flags_to_log(df, TABLE, KEY, run_id)
