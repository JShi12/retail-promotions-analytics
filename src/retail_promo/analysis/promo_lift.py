"""Phase 2: promotion lift from the product-week panel.

Spark prepares the model tables; pyfixest fits fixed-effects models on pandas.
All estimates are associations (see docs/phase2_plan.md).

Lift definitions
- Mailer-week lift (Poisson model): exp(b) - 1, the % change in weekly units in weeks the
  product is in the mailer (mailer coverage is ~100% of store traffic when present).
- In-store display lift: b itself. Units add up over stores, so for small display shares
  log(1 + L * share) ~ L * share and the coefficient approximates L, the relative lift in the
  stores that have the display. Larger shares make b somewhat smaller than L.
- Discount depth (OLS on log units, weeks with sales): exp(0.1 * b) - 1 per +10 points.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pyfixest as pf
from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window

MAIN_FE = "product_id + commodity_desc^week_no"
WEEK_FE = "product_id + week_no"
TWO_WAY = {"CRV1": "product_id+week_no"}
PRODUCT_ONLY = {"CRV1": "product_id"}

FRAME_COLUMNS = [
    "product_id", "commodity_desc", "week_no", "units", "sales_value", "display_share",
    "mailer_share", "avg_discount_depth", "mailer_lag", "mailer_lead", "active",
]  # fmt: skip


# ---------------------------------------------------------------- data preparation (Spark)


def select_categories(
    panel: DataFrame,
    n: int = 20,
    min_products: int = 20,
    min_mailer_weeks: int = 200,
    min_display_weeks: int = 1000,
) -> pd.DataFrame:
    """Top-n commodities by sales among those with enough products and mailer variation."""
    stats = (
        panel.groupBy("commodity_desc")
        .agg(
            F.sum("sales_value").alias("sales"),
            F.countDistinct("product_id").alias("products"),
            F.sum((F.col("mailer_share") > 0).cast("int")).alias("mailer_product_weeks"),
            F.sum((F.col("display_share") > 0).cast("int")).alias("display_product_weeks"),
        )
        .toPandas()
        .sort_values("sales", ascending=False, ignore_index=True)
    )
    stats["sales_rank"] = np.arange(1, len(stats) + 1)
    stats["sales_share"] = stats["sales"] / stats["sales"].sum()
    eligible = stats[
        (stats["products"] >= min_products) & (stats["mailer_product_weeks"] >= min_mailer_weeks)
    ]
    chosen = eligible.head(n).copy()
    chosen["display_estimable"] = chosen["display_product_weeks"] >= min_display_weeks
    return chosen.reset_index(drop=True)


def model_frame(panel: DataFrame, first_week: int, last_week: int) -> DataFrame:
    """Panel rows in the window, with last/next week's mailer and an active-span flag.

    Lags and leads are taken before the window filter, so the first week of the window still
    has its previous week's mailer when the panel has it.
    """
    by_product = Window.partitionBy("product_id").orderBy("week_no")
    span = Window.partitionBy("product_id")
    week_if_bought = F.when(F.col("units") > 0, F.col("week_no"))
    return (
        panel.withColumn("mailer_lag", F.lag("mailer_share").over(by_product))
        .withColumn("mailer_lead", F.lead("mailer_share").over(by_product))
        .withColumn(
            "active",
            F.col("week_no").between(
                F.min(week_if_bought).over(span), F.max(week_if_bought).over(span)
            ),
        )
        .filter(F.col("week_no").between(first_week, last_week))
        .select(*FRAME_COLUMNS)
    )


def to_pandas(frame: DataFrame) -> pd.DataFrame:
    df = frame.toPandas()
    df["active"] = df["active"].fillna(False).astype(bool)
    return df


def category_frame(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    """Category x week units and volume-weighted promotion shares.

    A product's weight is its share of the category's units over the window.
    """
    d = df[df["commodity_desc"].isin(categories)].copy()
    totals = d.groupby("product_id")["units"].transform("sum")
    cat_totals = d.groupby("commodity_desc")["units"].transform("sum")
    d["weight"] = totals / cat_totals
    d["w_mailer"] = d["weight"] * d["mailer_share"]
    d["w_display"] = d["weight"] * d["display_share"]
    out = d.groupby(["commodity_desc", "week_no"], as_index=False).agg(
        units=("units", "sum"),
        w_mailer=("w_mailer", "sum"),
        w_display=("w_display", "sum"),
        w=("weight", "sum"),
    )
    # Normalise by the weight present in the week so shares stay in [0, 1].
    out["cat_mailer_share"] = out["w_mailer"] / out["w"]
    out["cat_display_share"] = out["w_display"] / out["w"]
    return out.drop(columns=["w_mailer", "w_display", "w"])


# ---------------------------------------------------------------- estimation (pyfixest)


def _fit(kind: str, formula: str, data: pd.DataFrame, vcov):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = pf.fepois if kind == "poisson" else pf.feols
        return fit(formula, data=data, vcov=vcov)


def _rows(fitted, terms: dict[str, str], **labels) -> list[dict]:
    """One result row per term. terms maps term -> lift definition."""
    tidy = fitted.tidy()
    rows = []
    for term, definition in terms.items():
        b, se, lo, hi = (
            float(tidy.loc[term, c]) for c in ["Estimate", "Std. Error", "2.5%", "97.5%"]
        )
        if definition == "exp(b)-1":
            lift, lift_lo, lift_hi = np.expm1(b), np.expm1(lo), np.expm1(hi)
        elif definition == "b":
            lift, lift_lo, lift_hi = b, lo, hi
        elif definition == "exp(0.1b)-1":
            lift, lift_lo, lift_hi = np.expm1(0.1 * b), np.expm1(0.1 * lo), np.expm1(0.1 * hi)
        else:
            raise ValueError(definition)
        rows.append(
            {
                **labels,
                "term": term,
                "coef": b,
                "std_error": se,
                "coef_low": lo,
                "coef_high": hi,
                "lift": lift,
                "lift_low": lift_lo,
                "lift_high": lift_hi,
                "lift_definition": definition,
                "n_obs": int(fitted._N),
            }
        )
    return rows


MAIN_TERMS = {"mailer_share": "exp(b)-1", "display_share": "b"}


def product_lift(df: pd.DataFrame, fe: str = MAIN_FE, vcov=TWO_WAY, **labels) -> list[dict]:
    model = _fit("poisson", f"units ~ display_share + mailer_share | {fe}", df, vcov)
    return _rows(model, MAIN_TERMS, **labels)


def category_lifts(df: pd.DataFrame, cats: pd.DataFrame) -> list[dict]:
    """Main model per selected category (one category, so commodity x week = week)."""
    rows = []
    for c in cats.itertuples():
        d = df[df["commodity_desc"] == c.commodity_desc]
        terms = dict(MAIN_TERMS) if c.display_estimable else {"mailer_share": "exp(b)-1"}
        model = _fit("poisson", f"units ~ display_share + mailer_share | {WEEK_FE}", d, TWO_WAY)
        rows += _rows(model, terms, model="product", scope="category", category=c.commodity_desc)
    return rows


def timing_checks(df: pd.DataFrame) -> list[dict]:
    """Lead placebo and post-promotion dip, pooled, main fixed effects."""
    d = df.dropna(subset=["mailer_lag", "mailer_lead"])
    model = _fit(
        "poisson",
        f"units ~ display_share + mailer_share + mailer_lag + mailer_lead | {MAIN_FE}",
        d,
        TWO_WAY,
    )
    terms = {"mailer_share": "exp(b)-1", "mailer_lag": "exp(b)-1", "mailer_lead": "exp(b)-1"}
    return _rows(model, terms, model="timing", scope="pooled", category="ALL")


def discount_depth(df: pd.DataFrame) -> list[dict]:
    """Descriptive: weeks with sales only, log units."""
    d = df[df["units"] > 0].assign(log_units=lambda x: np.log(x["units"]))
    model = _fit(
        "ols",
        f"log_units ~ display_share + mailer_share + avg_discount_depth | {MAIN_FE}",
        d,
        TWO_WAY,
    )
    return _rows(
        model,
        {"avg_discount_depth": "exp(0.1b)-1"},
        model="depth_descriptive",
        scope="pooled",
        category="ALL",
    )


def category_net_lift(cat_df: pd.DataFrame) -> list[dict]:
    """Category units on volume-weighted promoted share; category and week fixed effects.

    Without switching between products, the mailer coefficient would be close to the
    product-level mailer-week lift (exp(b)-1); a smaller value indicates switching.
    """
    model = _fit(
        "poisson",
        "units ~ cat_display_share + cat_mailer_share | commodity_desc + week_no",
        cat_df,
        {"CRV1": "commodity_desc"},
    )
    return _rows(
        model,
        {"cat_mailer_share": "b", "cat_display_share": "b"},
        model="category_net",
        scope="pooled",
        category="SELECTED",
    )


def results_frame(rows: list[dict]) -> pd.DataFrame:
    cols = ["model", "scope", "category", "variant", "term", "coef", "std_error", "coef_low", "coef_high",
            "lift", "lift_low", "lift_high", "lift_definition", "n_obs"]  # fmt: skip
    out = pd.DataFrame(rows)
    if "variant" not in out:
        out["variant"] = "main"
    out["variant"] = out["variant"].fillna("main")
    return out[cols]
