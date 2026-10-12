"""Phase 3: segment households by price-and-promotion behaviour (docs/phase3_plan.md).

Each behaviour measure is the household's actual value minus the value expected from
category x store benchmark rates, computed leave-one-out (the household's own purchases are
removed from the benchmark). Spark MLlib k-means on three standardised inputs:
deal-proneness index, adjusted private-label share, log coupon use.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans, KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

CELL = ["commodity_desc", "store_id"]
INPUTS = ["deal_index", "pl_adjusted", "log_coupon"]
MIN_BASKETS = 10

# measure -> (numerator column, denominator column, line filter)
MEASURES = {
    "promo": ("promo_spend", "sales_value", "promo_observed"),
    "disc": ("disc", "shelf_value", None),
    "pl": ("pl_spend", "sales_value", None),
}


# ---------------------------------------------------------------- behaviour features (Spark)


def _lines(transactions: DataFrame, products: DataFrame) -> DataFrame:
    """Merchandise lines with a positive quantity, with the numerators each measure needs."""
    return (
        transactions.filter(~F.col("is_non_merchandise") & (F.col("quantity") > 0))
        .join(products.select("product_id", "commodity_desc", "is_private_label"), "product_id")
        .select(
            "household_key",
            *CELL,
            "promo_observed",
            F.col("sales_value").cast("double").alias("sales_value"),
            (F.col("sales_value") - F.col("retail_disc")).cast("double").alias("shelf_value"),
            (-F.col("retail_disc")).cast("double").alias("disc"),
            F.when(F.col("is_private_label"), F.col("sales_value"))
            .otherwise(0)
            .cast("double")
            .alias("pl_spend"),
            F.when(
                F.coalesce(F.col("on_display") | F.col("on_mailer"), F.lit(False)),
                F.col("sales_value"),
            )
            .otherwise(0)
            .cast("double")
            .alias("promo_spend"),
        )
    )


def _actual_expected(
    lines: DataFrame, name: str, num: str, den: str, line_filter: str | None
) -> DataFrame:
    """Per household: actual = sum(num) / sum(den); expected = spend-weighted leave-one-out rate.

    Cells where the household is the only buyer have no benchmark and are left out of both.
    """
    d = lines.filter(F.col(line_filter)) if line_filter else lines
    # Spark column names are case-insensitive, so household and cell totals get distinct names.
    hh_cell = d.groupBy("household_key", *CELL).agg(
        F.sum(num).alias("hh_num"), F.sum(den).alias("hh_den")
    )
    cell = hh_cell.groupBy(*CELL).agg(
        F.sum("hh_num").alias("cell_num"), F.sum("hh_den").alias("cell_den")
    )
    loo_rate = F.try_divide(
        F.col("cell_num") - F.col("hh_num"), F.col("cell_den") - F.col("hh_den")
    )
    usable = hh_cell.join(cell, CELL).withColumn("rate", loo_rate).filter(F.col("rate").isNotNull())
    return usable.groupBy("household_key").agg(
        F.try_divide(F.sum("hh_num"), F.sum("hh_den")).alias(f"{name}_actual"),
        F.try_divide(F.sum(F.col("hh_den") * F.col("rate")), F.sum("hh_den")).alias(
            f"{name}_expected"
        ),
    )


def household_behaviour(
    transactions: DataFrame, products: DataFrame, features: DataFrame
) -> DataFrame:
    """One row per household: actual, expected and adjusted measures plus the clustering inputs."""
    lines = _lines(transactions, products)
    out = features.select("household_key", "n_baskets", "coupon_line_share")
    for name, (num, den, flt) in MEASURES.items():
        out = out.join(_actual_expected(lines, name, num, den, flt), "household_key", "left")
        out = out.withColumn(
            f"{name}_adjusted", F.col(f"{name}_actual") - F.col(f"{name}_expected")
        )
    out = (
        out.withColumn("promo_imputed", F.col("promo_adjusted").isNull())
        .withColumn("promo_adjusted", F.coalesce("promo_adjusted", F.lit(0.0)))
        .withColumn("disc_adjusted", F.coalesce("disc_adjusted", F.lit(0.0)))
        .withColumn("pl_adjusted", F.coalesce("pl_adjusted", F.lit(0.0)))
        .withColumn("log_coupon", F.log1p(F.col("coupon_line_share") * 100))
        .withColumn("low_activity", F.col("n_baskets") < MIN_BASKETS)
    )
    # Deal-proneness index: mean of the standardised promotion and discount measures, with the
    # mean and SD taken over the households used for fitting.
    stats = (
        out.filter(~F.col("low_activity"))
        .agg(
            *[F.mean(f"{m}_adjusted").alias(f"{m}_mu") for m in ("promo", "disc")],
            *[F.stddev(f"{m}_adjusted").alias(f"{m}_sd") for m in ("promo", "disc")],
        )
        .first()
    )
    z = [(F.col(f"{m}_adjusted") - stats[f"{m}_mu"]) / stats[f"{m}_sd"] for m in ("promo", "disc")]
    return out.withColumn("deal_index", (z[0] + z[1]) / 2)


# ---------------------------------------------------------------- clustering (MLlib)


def scaler(df: DataFrame, inputs: list[str] = INPUTS) -> PipelineModel:
    return Pipeline(
        stages=[
            VectorAssembler(inputCols=inputs, outputCol="raw"),
            StandardScaler(inputCol="raw", outputCol="features", withMean=True, withStd=True),
        ]
    ).fit(df)


def fit_best(X: DataFrame, k: int, starts: int = 10, seed0: int = 0) -> KMeansModel:
    """Lowest-cost of several random starts (single starts land in different local optima)."""
    models = [KMeans(k=k, seed=seed0 + s, maxIter=100).fit(X) for s in range(starts)]
    return min(models, key=lambda m: m.summary.trainingCost)


def _labels(model: KMeansModel, X: DataFrame) -> pd.Series:
    p = model.transform(X).select("household_key", "prediction").toPandas()
    return p.set_index("household_key")["prediction"]


def jaccard_stability(full: pd.Series, boot: pd.Series) -> dict[int, float]:
    """Hennig (2007): for each original cluster, the best Jaccard match among bootstrap clusters,
    restricted to the households that appear in the bootstrap sample."""
    common = boot.index.intersection(full.index)
    f, b = full.loc[common], boot.loc[common]
    out = {}
    for c in sorted(full.unique()):
        members = set(f.index[f == c])
        best = 0.0
        for d in b.unique():
            other = set(b.index[b == d])
            union = members | other
            best = max(best, len(members & other) / len(union) if union else 0.0)
        out[int(c)] = best
    return out


def shuffled(pdf: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """Each column permuted independently: same distributions, no structure between them."""
    rng = np.random.default_rng(seed)
    out = pdf.copy()
    for c in cols:
        out[c] = rng.permutation(out[c].to_numpy())
    return out


def choose_k(
    spark: SparkSession,
    fit_df: DataFrame,
    ks: range = range(2, 7),
    n_boot: int = 20,
    boot_starts: int = 5,
    n_null: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Silhouette, shuffled-baseline silhouette and per-segment bootstrap Jaccard for each k."""
    # No .cache(): serverless does not allow it; fit_df is a small in-memory table (see phase3.run).
    X = scaler(fit_df).transform(fit_df).select("household_key", "features", *INPUTS)
    pdf = X.select("household_key", *INPUTS).toPandas()
    ev = ClusteringEvaluator()
    summary, per_segment = [], []
    for k in ks:
        model = fit_best(X, k)
        full = _labels(model, X)
        null_sil = []
        for r in range(n_null):
            sdf = spark.createDataFrame(shuffled(pdf, INPUTS, seed=r))
            XS = scaler(sdf).transform(sdf)
            null_sil.append(ev.evaluate(fit_best(XS, k).transform(XS)))
        jac = {c: [] for c in sorted(full.unique())}
        for b in range(n_boot):
            sample = X.sample(withReplacement=True, fraction=1.0, seed=1000 + b)
            boot = _labels(
                fit_best(sample, k, starts=boot_starts, seed0=100),
                sample.dropDuplicates(["household_key"]),
            )
            for c, v in jaccard_stability(full, boot).items():
                jac[c].append(v)
        sil = ev.evaluate(model.transform(X))
        mean_jac = {c: float(np.mean(v)) for c, v in jac.items()}
        summary.append({
            "k": k, "silhouette": sil, "shuffled_silhouette": float(np.mean(null_sil)),
            "gap": sil - float(np.mean(null_sil)), "min_segment_jaccard": min(mean_jac.values()),
            "all_stable": min(mean_jac.values()) >= 0.75, "sizes": sorted(full.value_counts().tolist(), reverse=True),
        })  # fmt: skip
        per_segment += [
            {"k": k, "segment": c, "size": int((full == c).sum()), "jaccard": v}
            for c, v in mean_jac.items()
        ]
    return pd.DataFrame(summary), pd.DataFrame(per_segment)


def pick_k(summary: pd.DataFrame) -> int:
    """Among k whose segments are all stable, the one with the largest silhouette gap over the
    shuffled baseline (the idea of the gap statistic, Tibshirani, Walther & Hastie 2001)."""
    ok = summary[summary["all_stable"] & (summary["gap"] > 0)]
    if ok.empty:
        raise ValueError("No k passes the stability rule; report that instead of forcing a choice.")
    return int(ok.loc[ok["gap"].idxmax(), "k"])


# ---------------------------------------------------------------- final model, names, profiles

LEVELS = [(-0.5, "below average"), (0.5, "average")]  # standardised centroid cut-offs


def _level(z: float) -> str:
    for cut, word in LEVELS:
        if z < cut:
            return word
    return "above average"


def describe_centroid(center: np.ndarray) -> str:
    """Behaviour-only name from the standardised centroid, e.g. 'Promotion above average, ...'."""
    deal, pl, coupon = center
    return f"Promotion {_level(deal)} · Own label {_level(pl)} · Coupons {_level(coupon)}"


def short_label(center: np.ndarray) -> str:
    """Short chart label naming only the inputs that are above or below average."""
    words = []
    for z, what in zip(center, ["promotion", "own label", "coupons"], strict=True):
        level = _level(z)
        if level != "average":
            words.append(f"{'more' if level == 'above average' else 'less'} {what}")
    return ", ".join(words).capitalize() if words else "Average on all three"


def fit_segments(behaviour: DataFrame, k: int) -> tuple[DataFrame, pd.DataFrame]:
    """Fit on households with >= MIN_BASKETS baskets, assign everyone, order segments by deal index."""
    fit_df = behaviour.filter(~F.col("low_activity"))
    prep = scaler(fit_df)
    model = fit_best(prep.transform(fit_df), k)
    centers = np.array(model.clusterCenters())
    order = np.argsort(-centers[:, 0])  # segment 1 = highest deal-proneness
    relabel = {int(old): new + 1 for new, old in enumerate(order)}
    centroids = pd.DataFrame(centers[order], columns=INPUTS)
    centroids.insert(0, "segment", range(1, k + 1))
    centroids["segment_name"] = [describe_centroid(c) for c in centers[order]]
    centroids["segment_label"] = [short_label(c) for c in centers[order]]

    assigned = model.transform(prep.transform(behaviour))
    mapping = F.create_map(*[x for old, new in relabel.items() for x in (F.lit(old), F.lit(new))])
    names = F.create_map(
        *[
            x
            for s, n in zip(centroids["segment"], centroids["segment_name"])
            for x in (F.lit(int(s)), F.lit(n))
        ]
    )
    labels = F.create_map(
        *[
            x
            for s, n in zip(centroids["segment"], centroids["segment_label"])
            for x in (F.lit(int(s)), F.lit(n))
        ]
    )
    out = (
        assigned.withColumn("segment", mapping[F.col("prediction")])
        .withColumn("segment_name", names[F.col("segment")])
        .withColumn("segment_label", labels[F.col("segment")])
    )
    return out.drop("raw", "features", "prediction"), centroids


def profiles(
    assigned: pd.DataFrame, household_features: pd.DataFrame, redemption: pd.DataFrame
) -> pd.DataFrame:
    """Median behaviour and descriptors per segment (households used for fitting only)."""
    d = assigned[~assigned["low_activity"]].merge(
        household_features, on="household_key", suffixes=("", "_hf")
    )
    cols = ["promo_actual", "promo_expected", "disc_actual", "disc_expected", "pl_actual", "pl_expected",
            "coupon_line_share", "baskets_per_week", "avg_basket_spend", "total_spend",
            "share_grocery", "share_drug_gm", "share_produce", "share_meat", "share_deli"]  # fmt: skip
    prof = d.groupby("segment")[cols].median()
    prof.insert(0, "households", d.groupby("segment").size())
    prof.insert(
        1, "share_of_spend", d.groupby("segment")["total_spend"].sum() / d["total_spend"].sum()
    )
    red = redemption.merge(d[["household_key", "segment"]], on="household_key")
    prof["campaign_pairs"] = red.groupby("segment").size()
    prof["redemption_rate"] = red.groupby("segment")["redeemed"].mean()
    prof["with_demographics"] = d.groupby("segment")["has_demographics"].sum()
    return prof.reset_index()
