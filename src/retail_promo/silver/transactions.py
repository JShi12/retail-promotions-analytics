"""transaction_data: typed receipt lines with prices, discount depth and promotion status."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.silver._typing import MONEY, cast_columns

TABLE = "transactions"
KEY = ["basket_id", "product_id"]  # unique per Phase 0 inspection; checked in silver checks

TYPES = {
    "household_key": "int",
    "basket_id": "bigint",
    "day": "int",
    "product_id": "int",
    "quantity": "int",
    "sales_value": MONEY,
    "store_id": "int",
    "retail_disc": MONEY,
    "trans_time": "int",
    "week_no": "int",
    "coupon_disc": MONEY,
    "coupon_match_disc": MONEY,
}


def build(
    bronze: DataFrame,
    products: DataFrame,
    causal: DataFrame,
    run_id: str,
    promo_first_week: int = 9,
    promo_last_week: int = 101,
) -> tuple[DataFrame, DataFrame]:
    df = dq.init_flags(bronze.drop("_source_file", "_ingested_at", "_run_id"))
    df = cast_columns(df, TYPES)

    df = df.join(
        products.select("product_id", "is_non_merchandise"), "product_id", "left"
    ).withColumn("is_non_merchandise", F.coalesce("is_non_merchandise", F.lit(False)))
    df = dq.add_flag(df, F.col("is_non_merchandise"), "non_merchandise")
    df = dq.add_flag(df, F.col("quantity") == 0, "quantity_zero")
    df = dq.add_flag(df, F.col("sales_value") == 0, "sales_zero")
    df = dq.add_flag(
        df, (F.col("quantity") == 0) & (F.col("sales_value") > 0), "quantity_zero_sales_positive"
    )
    df = dq.add_flag(df, F.col("retail_disc") > 0, "retail_disc_positive")
    df = dq.add_flag(
        df, F.col("week_no") != F.floor((F.col("day") + 1) / 7) + 1, "week_day_mismatch"
    )

    # Prices (user guide). Discounts are stored as negatives.
    qty = F.when(F.col("quantity") > 0, F.col("quantity").cast("decimal(12,2)"))
    shelf_value = F.col("sales_value") - F.col("retail_disc") - F.col("coupon_match_disc")
    df = (
        df.withColumn("shelf_price", (shelf_value / qty).cast("decimal(12,4)"))
        .withColumn(
            "paid_price",
            ((F.col("sales_value") + F.col("coupon_disc")) / qty).cast("decimal(12,4)"),
        )
        .withColumn(
            "discount_depth",
            F.when(
                (F.col("sales_value") - F.col("retail_disc")) > 0,
                (-F.col("retail_disc") / (F.col("sales_value") - F.col("retail_disc"))).cast(
                    "decimal(8,4)"
                ),
            ),
        )
        .withColumn("has_loyalty_discount", F.col("retail_disc") < 0)
        .withColumn("has_coupon", F.col("coupon_disc") < 0)
    )

    # Promotion status. Only known for causal stores in causal weeks; elsewhere it is null.
    causal_stores = causal.select("store_id").distinct().withColumn("_causal_store", F.lit(True))
    promos = causal.select(*["product_id", "store_id", "week_no"], "on_display", "on_mailer")
    df = df.join(F.broadcast(causal_stores), "store_id", "left").join(
        promos, ["product_id", "store_id", "week_no"], "left"
    )
    observed = F.coalesce("_causal_store", F.lit(False)) & F.col("week_no").between(
        promo_first_week, promo_last_week
    )
    df = (
        df.withColumn("promo_observed", observed)
        .withColumn("on_display", F.when(observed, F.coalesce("on_display", F.lit(False))))
        .withColumn("on_mailer", F.when(observed, F.coalesce("on_mailer", F.lit(False))))
        .drop("_causal_store")
    )

    log = dq.flags_to_log(df, TABLE, KEY, run_id)
    columns = [*TYPES, "is_non_merchandise", "shelf_price", "paid_price", "discount_depth"]
    columns += [
        "has_loyalty_discount",
        "has_coupon",
        "promo_observed",
        "on_display",
        "on_mailer",
        "dq_flags",
    ]
    return df.select(*columns), log
