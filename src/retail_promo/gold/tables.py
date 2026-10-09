"""The four gold tables."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

from retail_promo.config import LAST_DAY
from retail_promo.gold.features import aggregate_features, lines_with_products

DEMOGRAPHIC_COLUMNS = [
    "classification_1", "classification_2", "classification_3", "classification_4", "classification_5",
    "homeowner_desc", "kid_category_desc", "classification_1_ord", "classification_3_ord",
    "classification_4_ord", "classification_5_ord", "kids_ord",
]  # fmt: skip


def household_features(
    transactions: DataFrame, products: DataFrame, demographics: DataFrame
) -> DataFrame:
    """One row per household over the whole period (Phase 3)."""
    lines = lines_with_products(transactions, products).withColumn("ref_day", F.lit(LAST_DAY + 1))
    features = aggregate_features(lines, ["household_key"], "ref_day")
    demo = demographics.select("household_key", *DEMOGRAPHIC_COLUMNS).withColumn(
        "has_demographics", F.lit(True)
    )
    return features.join(demo, "household_key", "left").withColumn(
        "has_demographics", F.coalesce("has_demographics", F.lit(False))
    )


def campaign_household(
    transactions: DataFrame,
    products: DataFrame,
    campaign_table: DataFrame,
    campaign_desc: DataFrame,
    coupons: DataFrame,
    redemptions: DataFrame,
) -> DataFrame:
    """One row per (household, campaign) received, with features from before the campaign start
    and the label `redeemed` (Phase 4)."""
    pairs = campaign_table.select("household_key", "campaign").join(
        campaign_desc.select(
            "campaign",
            F.col("description").alias("campaign_type"),
            "start_day",
            "end_day",
            "duration_days",
        ),
        "campaign",
    )

    # Behaviour strictly before the start day.
    lines = lines_with_products(transactions, products).join(
        F.broadcast(pairs.select("household_key", "campaign", F.col("start_day").alias("ref_day"))),
        "household_key",
    )
    pre_lines = lines.filter(F.col("day") < F.col("ref_day"))
    features = aggregate_features(pre_lines, ["household_key", "campaign"], "ref_day").drop(
        "ref_day"
    )

    # Pre-campaign spend on the products this campaign's coupons cover.
    coupon_products = coupons.select("campaign", "product_id").distinct()
    relevance = (
        pre_lines.filter(~F.col("is_non_merchandise"))
        .join(coupon_products, ["campaign", "product_id"])
        .groupBy("household_key", "campaign")
        .agg(
            F.sum("sales_value").cast("double").alias("pre_spend_on_coupon_products"),
            F.countDistinct("product_id").alias("pre_n_coupon_products_bought"),
        )
    )

    # Earlier campaigns: completed before this start, and redemptions before this start.
    other = campaign_table.select("household_key", F.col("campaign").alias("_other")).join(
        campaign_desc.select(
            F.col("campaign").alias("_other"), F.col("end_day").alias("_other_end")
        ),
        "_other",
    )
    prior_campaigns = (
        pairs.join(other, "household_key")
        .filter(F.col("_other_end") < F.col("start_day"))
        .groupBy("household_key", "campaign")
        .agg(F.countDistinct("_other").alias("prior_campaigns_completed"))
    )
    red = redemptions.select(
        "household_key", F.col("campaign").alias("_rc"), F.col("day").alias("_rday")
    )
    prior_red = (
        pairs.join(red, "household_key")
        .filter(F.col("_rday") < F.col("start_day"))
        .groupBy("household_key", "campaign")
        .agg(
            F.count(F.lit(1)).alias("prior_redemptions"),
            F.countDistinct("_rc").alias("prior_redeemed_campaigns"),
        )
    )

    label = redemptions.groupBy("household_key", "campaign").agg(
        F.count(F.lit(1)).alias("n_redemptions")
    )
    campaign_size = coupons.groupBy("campaign").agg(
        F.countDistinct("coupon_upc").alias("n_coupons"),
        F.countDistinct("product_id").alias("n_coupon_products"),
    )

    out = (
        pairs.join(features, ["household_key", "campaign"], "left")
        .join(relevance, ["household_key", "campaign"], "left")
        .join(prior_campaigns, ["household_key", "campaign"], "left")
        .join(prior_red, ["household_key", "campaign"], "left")
        .join(label, ["household_key", "campaign"], "left")
        .join(campaign_size, "campaign", "left")
    )
    zero_fill = [
        "pre_spend_on_coupon_products", "pre_n_coupon_products_bought", "prior_campaigns_completed",
        "prior_redemptions", "prior_redeemed_campaigns", "n_redemptions",
    ]  # fmt: skip
    for col in zero_fill:
        out = out.withColumn(col, F.coalesce(F.col(col), F.lit(0)))
    return (
        out.withColumn(
            "pre_coupon_product_spend_share",
            F.try_divide("pre_spend_on_coupon_products", "total_spend"),
        )
        .withColumn("has_prior_redemption", F.col("prior_redemptions") > 0)
        .withColumn("redeemed", (F.col("n_redemptions") > 0).cast("int"))
    )


def product_week_panel(
    transactions: DataFrame,
    causal: DataFrame,
    products: DataFrame,
    first_week: int,
    last_week: int,
    min_weeks: int,
) -> DataFrame:
    """One row per (product, week) for products bought in >= min_weeks weeks (Phase 2).

    Scope: merchandise lines, the stores present in causal_data, weeks first_week..last_week.
    Weeks with no purchases are kept with units = 0 and price null.
    display_share / mailer_share = sum of traffic weights of the stores featuring the product,
    where a store's weight is its share of in-scope transaction lines that week.
    """
    stores = causal.select("store_id").distinct()
    scope = (
        transactions.join(F.broadcast(stores), "store_id")
        .filter(F.col("week_no").between(first_week, last_week))
        .filter(~F.col("is_non_merchandise"))
    )

    weights = (
        scope.groupBy("week_no", "store_id")
        .agg(F.count(F.lit(1)).alias("_lines"))
        .withColumn(
            "store_weight", F.col("_lines") / F.sum("_lines").over(Window.partitionBy("week_no"))
        )
        .drop("_lines")
    )

    bought = scope.filter(F.col("quantity") > 0)
    sales = bought.groupBy("product_id", "week_no").agg(
        F.sum("quantity").alias("units"),
        F.sum("sales_value").cast("double").alias("sales_value"),
        F.countDistinct("household_key").alias("n_households"),
        F.count(F.lit(1)).alias("n_lines"),
        F.try_divide(
            F.sum(F.col("sales_value") - F.col("retail_disc") - F.col("coupon_match_disc")).cast(
                "double"
            ),
            F.sum("quantity").cast("double"),
        ).alias("avg_shelf_price"),
        F.try_divide(
            F.sum(-F.col("retail_disc")).cast("double"),
            F.sum(F.col("sales_value") - F.col("retail_disc")).cast("double"),
        ).alias("avg_discount_depth"),
    )

    in_scope_products = (
        sales.groupBy("product_id")
        .agg(F.countDistinct("week_no").alias("_weeks"))
        .filter(F.col("_weeks") >= min_weeks)
    ).select("product_id")

    promo = (
        causal.filter(F.col("week_no").between(first_week, last_week))
        .join(in_scope_products, "product_id")
        .join(weights, ["week_no", "store_id"], "left")
        .groupBy("product_id", "week_no")
        .agg(
            F.sum(F.when(F.col("on_display"), F.col("store_weight")).otherwise(F.lit(0.0))).alias(
                "display_share"
            ),
            F.sum(F.when(F.col("on_mailer"), F.col("store_weight")).otherwise(F.lit(0.0))).alias(
                "mailer_share"
            ),
            F.sum(F.col("on_display").cast("int")).alias("n_stores_display"),
            F.sum(F.col("on_mailer").cast("int")).alias("n_stores_mailer"),
        )
    )

    weeks = transactions.sparkSession.range(first_week, last_week + 1).select(
        F.col("id").cast("int").alias("week_no")
    )
    grid = in_scope_products.crossJoin(F.broadcast(weeks))
    panel = (
        grid.join(sales, ["product_id", "week_no"], "left")
        .join(promo, ["product_id", "week_no"], "left")
        .join(
            products.select(
                "product_id", "department", "commodity_desc", "sub_commodity_desc", "brand"
            ),
            "product_id",
        )
    )
    for col, fill in [
        ("units", 0), ("sales_value", 0.0), ("n_households", 0), ("n_lines", 0), ("display_share", 0.0),
        ("mailer_share", 0.0), ("n_stores_display", 0), ("n_stores_mailer", 0),
    ]:  # fmt: skip
        panel = panel.withColumn(col, F.coalesce(F.col(col), F.lit(fill)))
    return panel


def dq_summary(dq_log: DataFrame) -> DataFrame:
    """Row counts per run, table, rule and action, for the dashboard."""
    return dq_log.groupBy("run_id", "table_name", "rule_id", "action").agg(
        F.count(F.lit(1)).alias("n_rows"), F.max("logged_at").alias("logged_at")
    )
