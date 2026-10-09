"""campaign_desc, campaign_table, coupon and coupon_redempt."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from retail_promo import dq
from retail_promo.config import LAST_DAY
from retail_promo.silver._typing import cast_columns, trim_strings

META = ["_source_file", "_ingested_at", "_run_id"]


def build_campaign_desc(bronze: DataFrame, run_id: str) -> tuple[DataFrame, DataFrame]:
    df = dq.init_flags(bronze.drop(*META))
    df = cast_columns(df, {"campaign": "int", "start_day": "int", "end_day": "int"})
    df = trim_strings(df, ["description"])
    df = dq.add_flag(df, F.col("start_day") > F.col("end_day"), "campaign_window_invalid")
    df = dq.add_flag(df, F.col("end_day") > LAST_DAY, "campaign_window_beyond_data")
    df = df.withColumn("duration_days", F.col("end_day") - F.col("start_day") + 1)
    return df, dq.flags_to_log(df, "campaign_desc", ["campaign"], run_id)


def build_campaign_table(
    bronze: DataFrame, campaign_desc: DataFrame, run_id: str
) -> tuple[DataFrame, DataFrame]:
    table = "campaign_table"
    key = ["household_key", "campaign"]
    df, dup_log = dq.drop_exact_duplicates(bronze.drop(*META), table, run_id)
    df = cast_columns(dq.init_flags(df), {"household_key": "int", "campaign": "int"})
    df = trim_strings(df, ["description"])
    desc = campaign_desc.select("campaign", F.col("description").alias("_desc_type"))
    df = df.join(desc, "campaign", "left")
    df = dq.add_flag(
        df, ~F.col("description").eqNullSafe(F.col("_desc_type")), "campaign_type_mismatch"
    )
    df = df.drop("_desc_type")
    return df, dup_log.unionByName(dq.flags_to_log(df, table, key, run_id))


def build_coupons(bronze: DataFrame, run_id: str) -> tuple[DataFrame, DataFrame]:
    table = "coupons"
    df, dup_log = dq.drop_exact_duplicates(bronze.drop(*META), table, run_id)
    df = cast_columns(
        dq.init_flags(df), {"coupon_upc": "bigint", "product_id": "int", "campaign": "int"}
    )
    return df, dup_log.unionByName(
        dq.flags_to_log(df, table, ["coupon_upc", "product_id", "campaign"], run_id)
    )


def build_redemptions(
    bronze: DataFrame, campaign_desc: DataFrame, run_id: str
) -> tuple[DataFrame, DataFrame]:
    table = "coupon_redemptions"
    key = ["household_key", "day", "coupon_upc", "campaign"]
    types = {"household_key": "int", "day": "int", "coupon_upc": "bigint", "campaign": "int"}
    df, dup_log = dq.drop_exact_duplicates(bronze.drop(*META), table, run_id)
    df = cast_columns(dq.init_flags(df), types)

    n_campaigns = F.count(F.lit(1)).over(Window.partitionBy("household_key", "day", "coupon_upc"))
    df = dq.add_flag(df, n_campaigns > 1, "upc_multi_campaign_redemption")

    window = campaign_desc.select("campaign", "start_day", "end_day")
    df = df.join(window, "campaign", "left")
    df = dq.add_flag(
        df, ~F.col("day").between(F.col("start_day"), F.col("end_day")), "redemption_outside_window"
    )
    df = df.drop("start_day", "end_day")
    return df, dup_log.unionByName(dq.flags_to_log(df, table, key, run_id))
