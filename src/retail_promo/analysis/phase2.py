"""Run every Phase 2 model and return tidy tables (shared by the notebook and the local run)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from pyspark.sql import SparkSession

from retail_promo.analysis import promo_lift as pl
from retail_promo.config import Settings
from retail_promo.gold import tables as gold
from retail_promo.io import TableIO


@dataclass
class Phase2Results:
    categories: pd.DataFrame  # selected categories and selection stats
    raw_contrast: pd.DataFrame  # mean weekly units by promotion status
    results: pd.DataFrame  # every estimate (main, category, timing, depth, category net)
    sensitivity: pd.DataFrame  # pooled main-model estimates under each variant


def raw_contrast(df: pd.DataFrame) -> pd.DataFrame:
    status = pd.Series("none", index=df.index)
    status[(df["display_share"] > 0) & (df["mailer_share"] == 0)] = "display only"
    status[(df["display_share"] == 0) & (df["mailer_share"] > 0)] = "mailer only"
    status[(df["display_share"] > 0) & (df["mailer_share"] > 0)] = "both"
    out = df.groupby(status).agg(product_weeks=("units", "size"), mean_units=("units", "mean"))
    out["ratio_to_none"] = out["mean_units"] / out.loc["none", "mean_units"]
    return (
        out.reindex(["none", "display only", "mailer only", "both"])
        .rename_axis("status")
        .reset_index()
    )


def _products_bought_in(df: pd.DataFrame, min_weeks: int) -> pd.Index:
    weeks = df[df["units"] > 0].groupby("product_id")["week_no"].nunique()
    return weeks[weeks >= min_weeks].index


def run(
    spark: SparkSession, io: TableIO, settings: Settings, rebuild_min13: bool = True
) -> Phase2Results:
    panel = io.read("gold", "product_week_panel")
    cats = pl.select_categories(panel)
    first, last = settings.panel_first_week, settings.panel_last_week
    df = pl.to_pandas(pl.model_frame(panel, first, last))

    rows = pl.product_lift(df, model="product", scope="pooled", category="ALL")
    rows += pl.category_lifts(df, cats)
    rows += pl.timing_checks(df)
    rows += pl.discount_depth(df)
    selected = cats["commodity_desc"].tolist()
    # Same 20 categories as the category model, so product and category lift compare like for like.
    rows += pl.product_lift(
        df[df["commodity_desc"].isin(selected)],
        model="product",
        scope="selected",
        category="SELECTED",
    )
    rows += pl.category_net_lift(pl.category_frame(df, selected))
    results = pl.results_frame(rows)

    variants = [
        ("main", df, pl.MAIN_FE, pl.TWO_WAY),
        ("weeks 14-101", df[df["week_no"] >= 14], pl.MAIN_FE, pl.TWO_WAY),
        ("products bought in >= 52 weeks", df[df["product_id"].isin(_products_bought_in(df, 52))], pl.MAIN_FE, pl.TWO_WAY),
        ("active weeks only", df[df["active"]], pl.MAIN_FE, pl.TWO_WAY),
        ("product + week fixed effects", df, pl.WEEK_FE, pl.TWO_WAY),
        ("clustered by product only", df, pl.MAIN_FE, pl.PRODUCT_ONLY),
    ]  # fmt: skip
    if rebuild_min13:
        s = {t: io.read("silver", t) for t in ["transactions", "causal", "products"]}
        panel13 = gold.product_week_panel(
            s["transactions"], s["causal"], s["products"], first, last, 13
        )
        variants.insert(3, ("products bought in >= 13 weeks", pl.to_pandas(pl.model_frame(panel13, first, last)),
                            pl.MAIN_FE, pl.TWO_WAY))  # fmt: skip
    sens_rows = []
    for name, d, fe, vcov in variants:
        for r in pl.product_lift(
            d, fe=fe, vcov=vcov, model="product", scope="pooled", category="ALL"
        ):
            sens_rows.append({**r, "variant": name, "n_products": int(d["product_id"].nunique())})
    sensitivity = pd.DataFrame(sens_rows)

    return Phase2Results(cats, raw_contrast(df), results, sensitivity)


def write_gold(spark: SparkSession, io: TableIO, res: Phase2Results, run_id: str) -> None:
    """Store estimates for the dashboard: main results plus the sensitivity variants."""
    sens = res.sensitivity[res.sensitivity["variant"] != "main"]  # main is already in results
    table = pd.concat([res.results, sens[res.results.columns]], ignore_index=True)
    table["run_id"] = run_id
    io.write(spark.createDataFrame(table), "gold", "promo_lift_by_category")


def _fmt(row: pd.Series) -> str:
    return f"{row['lift']:+.1%} ({row['lift_low']:+.1%} to {row['lift_high']:+.1%})"


def category_table(res: Phase2Results, term: str) -> pd.DataFrame:
    """Per-category lift with 95% CI for one term, largest first, plus the pooled row."""
    r = res.results
    cat = r[(r["model"] == "product") & (r["scope"] == "category") & (r["term"] == term)]
    pooled = r[(r["model"] == "product") & (r["scope"] == "pooled") & (r["term"] == term)]
    out = pd.concat(
        [cat.sort_values("lift", ascending=False), pooled.assign(category="ALL PRODUCTS")]
    )
    return pd.DataFrame({"category": out["category"], "lift (95% CI)": out.apply(_fmt, axis=1),
                         "product-weeks": out["n_obs"]}).reset_index(drop=True)  # fmt: skip


def estimate_table(res: Phase2Results, model: str, scope: str | None = None) -> pd.DataFrame:
    r = res.results[res.results["model"] == model]
    if scope:
        r = r[r["scope"] == scope]
    return pd.DataFrame({"term": r["term"], "lift (95% CI)": r.apply(_fmt, axis=1),
                         "definition": r["lift_definition"], "observations": r["n_obs"]}).reset_index(drop=True)  # fmt: skip


def sensitivity_table(res: Phase2Results) -> pd.DataFrame:
    s = res.sensitivity.assign(estimate=lambda d: d.apply(_fmt, axis=1))
    wide = s.pivot(index="variant", columns="term", values="estimate")
    n = s.drop_duplicates("variant").set_index("variant")[["n_products", "n_obs"]]
    order = s["variant"].drop_duplicates().tolist()
    out = wide.join(n).loc[order].reset_index()
    return out.rename(
        columns={"mailer_share": "mailer-week lift", "display_share": "in-store display lift"}
    )
