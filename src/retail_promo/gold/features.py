"""Household behaviour features, computed from transaction lines up to a reference day.

One function serves both uses:
- Phase 3 segmentation: all lines, reference day = day after the last transaction day.
- Phase 4 campaign response: lines strictly before each campaign's start day, reference day =
  the start day. The caller filters `day < ref_day`; `last_day < ref_day` is checked in gold.
"""

from __future__ import annotations

from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as F

# Department -> feature suffix. MEAT and MEAT-PCKGD are combined.
DEPARTMENT_SHARES: dict[str, list[str]] = {
    "grocery": ["GROCERY"],
    "drug_gm": ["DRUG GM"],
    "produce": ["PRODUCE"],
    "meat": ["MEAT", "MEAT-PCKGD"],
    "deli": ["DELI"],
    "pastry": ["PASTRY"],
    "nutrition": ["NUTRITION"],
}

REQUIRED_LINE_COLUMNS = [
    "household_key", "basket_id", "day", "week_no", "product_id", "store_id", "quantity",
    "sales_value", "retail_disc", "coupon_disc", "is_non_merchandise", "has_loyalty_discount",
    "has_coupon", "promo_observed", "on_display", "on_mailer", "department", "is_private_label",
]  # fmt: skip


def lines_with_products(transactions: DataFrame, products: DataFrame) -> DataFrame:
    return transactions.join(
        products.select("product_id", "department", "is_private_label"), "product_id", "left"
    )


def _ratio(num: Column, den: Column) -> Column:
    return F.try_divide(num.cast("double"), den.cast("double"))


def aggregate_features(lines: DataFrame, group_cols: list[str], ref_day_col: str) -> DataFrame:
    """Aggregate lines (already restricted to day < ref_day) to one row per group."""
    merch = ~F.col("is_non_merchandise")
    sales = F.col("sales_value")
    merch_sales = F.when(merch, sales).otherwise(F.lit(0))
    ref = F.col(ref_day_col)

    def merch_sum(cond: Column) -> Column:
        return F.sum(F.when(merch & cond, sales).otherwise(F.lit(0)))

    aggs = [
        F.countDistinct("basket_id").alias("n_baskets"),
        F.countDistinct("week_no").alias("n_active_weeks"),
        F.min("day").alias("first_day"),
        F.max("day").alias("last_day"),
        F.countDistinct("store_id").alias("n_stores"),
        F.countDistinct(F.when(merch, F.col("product_id"))).alias("n_products"),
        F.sum(merch_sales).alias("total_spend"),
        F.sum(sales).alias("_all_spend"),
        F.sum(F.when(merch, F.col("quantity")).otherwise(F.lit(0))).alias("_units"),
        merch_sum(F.col("is_private_label")).alias("_private_spend"),
        F.avg(F.when(merch, F.col("has_loyalty_discount").cast("int"))).alias(
            "loyalty_discount_line_share"
        ),
        F.sum(F.when(merch, -F.col("retail_disc")).otherwise(F.lit(0))).alias("_loyalty_disc"),
        merch_sum(F.coalesce(F.col("on_display") | F.col("on_mailer"), F.lit(False))).alias(
            "_promo_spend"
        ),
        merch_sum(F.col("promo_observed")).alias("_observed_spend"),
        F.avg(F.when(merch, F.col("has_coupon").cast("int"))).alias("coupon_line_share"),
        F.sum(-F.col("coupon_disc")).alias("total_coupon_discount"),
        merch_sum(F.col("day") >= ref - 28).alias("spend_last_4w"),
        merch_sum(F.col("day") >= ref - 84).alias("spend_last_12w"),
        merch_sum((F.col("day") >= ref - 168) & (F.col("day") < ref - 84)).alias("spend_prev_12w"),
    ]
    for name, departments in DEPARTMENT_SHARES.items():
        aggs.append(merch_sum(F.col("department").isin(*departments)).alias(f"_dept_{name}"))

    out = lines.groupBy(*group_cols, ref_day_col).agg(*aggs)
    out = (
        out.withColumn("recency_days", ref - F.col("last_day"))
        .withColumn("tenure_days", ref - F.col("first_day"))
        .withColumn(
            "baskets_per_week",
            _ratio(F.col("n_baskets"), F.greatest(F.col("tenure_days") / 7, F.lit(1.0))),
        )
        .withColumn("avg_basket_spend", _ratio(F.col("total_spend"), F.col("n_baskets")))
        .withColumn("avg_units_per_basket", _ratio(F.col("_units"), F.col("n_baskets")))
        .withColumn("private_label_share", _ratio(F.col("_private_spend"), F.col("total_spend")))
        .withColumn(
            "avg_discount_depth",
            _ratio(F.col("_loyalty_disc"), F.col("total_spend") + F.col("_loyalty_disc")),
        )
        .withColumn("promo_spend_share", _ratio(F.col("_promo_spend"), F.col("_observed_spend")))
        .withColumn(
            "non_merch_spend_share",
            _ratio(F.col("_all_spend") - F.col("total_spend"), F.col("_all_spend")),
        )
        .withColumn("spend_trend_12w", _ratio(F.col("spend_last_12w"), F.col("spend_prev_12w")))
    )
    for name in DEPARTMENT_SHARES:
        out = out.withColumn(f"share_{name}", _ratio(F.col(f"_dept_{name}"), F.col("total_spend")))
    money = [
        "total_spend",
        "total_coupon_discount",
        "spend_last_4w",
        "spend_last_12w",
        "spend_prev_12w",
    ]
    for col in money:
        out = out.withColumn(col, F.col(col).cast("double"))
    return out.drop(*[c for c in out.columns if c.startswith("_")])
