"""Run Phase 3 end to end and return tidy tables (shared by the notebook and the local run)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from retail_promo.analysis import segments as sg
from retail_promo.io import TableIO

DEMOGRAPHICS = [
    "classification_1_ord",
    "classification_3_ord",
    "classification_4_ord",
    "classification_5_ord",
]


@dataclass
class Phase3Results:
    k_summary: pd.DataFrame  # silhouette, shuffled baseline, gap, stability per k
    k_segments: pd.DataFrame  # per-segment bootstrap Jaccard per k
    k: int
    centroids: pd.DataFrame  # standardised centroid and name per segment
    assigned: pd.DataFrame  # one row per household with segment and behaviour measures
    profiles: pd.DataFrame
    demographics: pd.DataFrame
    representativeness: pd.DataFrame
    coupon_check: pd.DataFrame  # coupon use with and without campaign-redemption days


def coupon_check(io: TableIO, assigned: pd.DataFrame) -> pd.DataFrame:
    """Is the coupon segment just campaign redemptions? Coupon line share with and without the
    days a household redeemed a campaign coupon, by segment (medians)."""
    t = io.read("silver", "transactions").filter(
        ~F.col("is_non_merchandise") & (F.col("quantity") > 0)
    )
    days = io.read("silver", "coupon_redemptions").select("household_key", "day").distinct()
    days = days.withColumn("redemption_day", F.lit(True))
    lines = t.join(days, ["household_key", "day"], "left").withColumn(
        "rday", F.col("redemption_day").isNotNull()
    )
    coupon = F.col("has_coupon").cast("int")
    hh = (
        lines.groupBy("household_key")
        .agg(
            F.avg(coupon).alias("coupon_share_all_days"),
            F.avg(F.when(~F.col("rday"), coupon)).alias("coupon_share_excl_redemption_days"),
            F.sum(F.when(F.col("rday"), coupon).otherwise(0)).alias("_on_rday"),
            F.sum(coupon).alias("_all"),
        )
        .toPandas()
    )
    fit = assigned.loc[~assigned["low_activity"], ["household_key", "segment"]].merge(
        hh, on="household_key"
    )
    out = fit.groupby("segment")[
        ["coupon_share_all_days", "coupon_share_excl_redemption_days"]
    ].median()
    out["share_of_coupon_lines_on_redemption_days"] = (
        fit.groupby("segment")["_on_rday"].sum() / fit.groupby("segment")["_all"].sum()
    )
    return out.reset_index()


def run(spark: SparkSession, io: TableIO, n_boot: int = 20) -> Phase3Results:
    hf = io.read("gold", "household_features")
    behaviour = sg.household_behaviour(
        io.read("silver", "transactions"), io.read("silver", "products"), hf
    ).cache()
    k_summary, k_segments = sg.choose_k(
        spark, behaviour.filter(~F.col("low_activity")), n_boot=n_boot
    )
    k = sg.pick_k(k_summary)
    assigned_sdf, centroids = sg.fit_segments(behaviour, k)
    assigned = assigned_sdf.toPandas()

    # Per-segment stability for the chosen k uses k-means' internal numbering; map it to the final
    # segment numbers by size (both come from the same best-of-10 fit, so sizes match exactly).
    sizes = assigned[~assigned["low_activity"]].groupby("segment").size()
    if sizes.duplicated().any():
        raise ValueError("Segment sizes are not unique; cannot map stability results to segments.")
    by_size = {int(n): int(s) for s, n in sizes.items()}
    chosen = k_segments["k"] == k
    k_segments.loc[chosen, "segment"] = k_segments.loc[chosen, "size"].map(by_size)
    if k_segments.loc[chosen, "segment"].isna().any():
        raise ValueError("Stability results do not match the final segment sizes.")

    hf_pd = hf.toPandas()
    redemption = (
        io.read("gold", "campaign_household").select("household_key", "redeemed").toPandas()
    )
    prof = sg.profiles(assigned, hf_pd, redemption).merge(
        centroids[["segment", "segment_label", "segment_name"]], on="segment"
    )

    fit = assigned.loc[~assigned["low_activity"], ["household_key", "segment"]].merge(
        hf_pd, on="household_key"
    )
    demo = fit[fit["has_demographics"]].groupby("segment")[DEMOGRAPHICS].mean()
    demo.insert(
        0, "households_with_demographics", fit[fit["has_demographics"]].groupby("segment").size()
    )
    rep = hf_pd.groupby("has_demographics")[
        ["baskets_per_week", "avg_basket_spend", "total_spend"]
    ].median()
    rep.insert(0, "households", hf_pd.groupby("has_demographics").size())
    return Phase3Results(
        k_summary,
        k_segments,
        k,
        centroids,
        assigned,
        prof,
        demo.reset_index(),
        rep.reset_index(),
        coupon_check(io, assigned),
    )


def write_gold(spark: SparkSession, io: TableIO, res: Phase3Results, run_id: str) -> None:
    cols = ["household_key", "segment", "segment_label", "segment_name", "deal_index", "pl_adjusted", "log_coupon",
            "promo_adjusted", "disc_adjusted", "low_activity", "promo_imputed"]  # fmt: skip
    out = res.assigned[cols].copy()
    out["run_id"] = run_id
    io.write(spark.createDataFrame(out), "gold", "household_segments")
