"""Phase 4: coupon campaign response model (docs/phase4_plan.md).

Predicts whether a household redeems at least one coupon from a campaign it receives, from
pre-campaign information only. Primary metric: ROC AUC within each campaign (the targeting
decision is who receives a given campaign), averaged with weights equal to the campaign's
redemptions. Tuning is a nested, household-grouped loop that optimises the same within-campaign
AUC.

The models are scikit-learn, fitted on the driver: the table has 7,208 rows, and Spark MLlib's
gradient-boosted trees run one Spark job per tree, which on serverless compute turned a
15-minute local run into hours. Spark still reads and prepares the data.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

KEY = ["household_key", "campaign"]
LABEL = "redeemed"
TIME_CUTOFF = (
    587  # train on campaigns ended before this day, test on campaigns starting on/after it
)

# Feature groups: correlated features are permuted together for importance.
FEATURE_GROUPS: dict[str, list[str]] = {
    "coupon use": ["coupon_line_share", "total_coupon_discount"],
    "past redemptions": ["prior_redemptions", "has_prior_redemption_f", "prior_redeemed_campaigns"],
    "past campaigns received": ["prior_campaigns_completed"],
    "relevance to this campaign": ["pre_spend_on_coupon_products", "pre_coupon_product_spend_share", "pre_n_coupon_products_bought"],
    "shopping volume": ["total_spend", "n_products", "n_baskets", "baskets_per_week", "n_active_weeks",
                        "spend_last_4w", "spend_last_12w", "spend_prev_12w"],
    "basket size": ["avg_basket_spend", "avg_units_per_basket"],
    "recency and trend": ["recency_days", "tenure_days", "spend_trend_12w"],
    "price and promotion": ["private_label_share", "promo_spend_share", "avg_discount_depth", "loyalty_discount_line_share"],
    "category mix": ["share_grocery", "share_drug_gm", "share_produce", "share_meat", "share_deli",
                     "share_pastry", "share_nutrition", "non_merch_spend_share"],
    "stores used": ["n_stores"],
    "campaign design": ["is_type_a", "is_type_b", "duration_days", "n_coupons", "n_coupon_products"],
}  # fmt: skip
FEATURES = [f for group in FEATURE_GROUPS.values() for f in group]
# Never model inputs: the label's own count, identifiers and absolute days.
EXCLUDED = [
    "n_redemptions",
    "campaign",
    "household_key",
    "start_day",
    "end_day",
    "first_day",
    "last_day",
]

GRIDS = {
    # C is the inverse L1 penalty strength (smaller C = more features shrunk to zero).
    "logistic_l1": [{"C": c} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)],
    # Depth 1-3 and 50-400 trees at learning rate 0.1. The earlier Spark MLlib version kept choosing
    # the edge of a smaller grid (finally depth 1, 200 trees), so this grid spans past it.
    "gbt": [{"max_depth": d, "max_iter": n} for d in (1, 2, 3) for n in (50, 100, 200, 400)],
}


# ---------------------------------------------------------------- data


def model_table(campaign_household: pd.DataFrame) -> pd.DataFrame:
    """Model-ready pandas table: numeric features, label, keys, campaign timing (for splitting only)."""
    d = campaign_household.copy()
    d["is_type_a"] = (d["campaign_type"] == "TypeA").astype(float)
    d["is_type_b"] = (d["campaign_type"] == "TypeB").astype(float)
    d["has_prior_redemption_f"] = d["has_prior_redemption"].astype(float)
    for col in FEATURES:
        d[col] = d[col].astype(float)
    d[LABEL] = d[LABEL].astype(int)
    keep = [*KEY, "campaign_type", "start_day", "end_day", LABEL, *FEATURES]
    return d[keep].sort_values(KEY, ignore_index=True)


def grouped_folds(df: pd.DataFrame, n_folds: int, seed: int, col: str = "fold") -> pd.Series:
    """Assign households to folds so each household's campaigns share a fold and folds have
    similar numbers of redemptions: shuffle, sort by redemptions, deal out round-robin."""
    hh = df.groupby("household_key")[LABEL].agg(["sum", "size"]).reset_index()
    hh = hh.sample(frac=1.0, random_state=seed).sort_values(
        ["sum", "size"], ascending=False, kind="stable"
    )
    hh[col] = np.arange(len(hh)) % n_folds
    return df["household_key"].map(hh.set_index("household_key")[col]).astype(int)


# ---------------------------------------------------------------- metrics


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """ROC AUC via ranks (ties get the average rank)."""
    y = np.asarray(y)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank().to_numpy()
    return (ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def within_campaign_auc(df: pd.DataFrame, score: str) -> float:
    """AUC inside each campaign, averaged with weights = the campaign's redemptions."""
    rows = []
    for _, g in df.groupby("campaign"):
        a = auc(g[LABEL].to_numpy(), g[score].to_numpy())
        if not math.isnan(a):
            rows.append((a, g[LABEL].sum()))
    if not rows:
        return float("nan")
    a, w = np.array(rows).T
    return float(np.average(a, weights=w))


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """PR AUC as average precision (ties broken by order after sorting on score)."""
    order = np.argsort(-np.asarray(score), kind="stable")
    y = np.asarray(y)[order]
    if y.sum() == 0:
        return float("nan")
    hits = np.cumsum(y)
    precision = hits / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / y.sum())


def capture(df: pd.DataFrame, score: str, share: float) -> float:
    """Share of all redemptions captured by sending each campaign to its top `share` of
    households by score (top-k per campaign, k = ceil(share * campaign size))."""
    captured = 0
    for _, g in df.groupby("campaign"):
        k = math.ceil(share * len(g))
        captured += g.sort_values(score, ascending=False, kind="stable").head(k)[LABEL].sum()
    return captured / df[LABEL].sum()


def within_campaign_lift(df: pd.DataFrame, score: str, share: float = 0.10) -> float:
    """Redemption rate among each campaign's top `share` by score ÷ the overall rate."""
    top_n = top_pos = 0
    for _, g in df.groupby("campaign"):
        k = math.ceil(share * len(g))
        top_n += k
        top_pos += g.sort_values(score, ascending=False, kind="stable").head(k)[LABEL].sum()
    return (top_pos / top_n) / df[LABEL].mean()


def metrics(df: pd.DataFrame, score: str) -> dict[str, float]:
    y, s = df[LABEL].to_numpy(), df[score].to_numpy()
    return {
        "auc_within": within_campaign_auc(df, score),
        "auc_pooled": auc(y, s),
        "pr_auc": average_precision(y, s),
        "lift_top10_within": within_campaign_lift(df, score),
    }


# ---------------------------------------------------------------- models (scikit-learn)


def _pipeline(kind: str, params: dict, seed: int = 7) -> Pipeline:
    """Imputation and scaling are steps of the pipeline, so they are fitted on training folds only."""
    if kind == "logistic_l1":
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=1000, random_state=seed, **params
        )
        steps = [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
            ("model", clf),
        ]
    else:
        clf = HistGradientBoostingClassifier(
            learning_rate=0.1, early_stopping=False, random_state=seed, **params
        )
        steps = [("impute", SimpleImputer(strategy="median")), ("model", clf)]
    return Pipeline(steps)


def fit(kind: str, params: dict, data: pd.DataFrame) -> Pipeline:
    return _pipeline(kind, params).fit(data[FEATURES], data[LABEL])


def predict(model: Pipeline, data: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(data[FEATURES])[:, 1]


def tune_and_fit(train: pd.DataFrame, kind: str, inner_folds: int = 3, seed: int = 11):
    """Choose parameters by inner household-grouped CV on within-campaign AUC, then refit on all of train."""
    inner = grouped_folds(train, inner_folds, seed, col="inner")
    scores = []
    for params in GRIDS[kind]:
        fold_auc = []
        for f in range(inner_folds):
            fit_part, val_part = train[inner != f], train[inner == f]
            model = fit(kind, params, fit_part)
            fold_auc.append(
                within_campaign_auc(val_part.assign(score=predict(model, val_part)), "score")
            )
        scores.append((float(np.mean(fold_auc)), params))
    best_auc, best = max(scores, key=lambda s: s[0])
    grid = pd.DataFrame([{**p, "inner_auc_within": a} for a, p in scores])
    return fit(kind, best, train), best, best_auc, grid


# ---------------------------------------------------------------- importance


def permuted_within_campaign(df: pd.DataFrame, cols: list[str], seed: int) -> pd.DataFrame:
    """Shuffle a group of columns together, within each campaign (campaign-level columns are
    constant inside a campaign, so they are left unchanged by design)."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for idx in df.groupby("campaign").groups.values():
        idx = np.asarray(idx)
        out.loc[idx, cols] = df.loc[rng.permutation(idx), cols].to_numpy()
    return out


def group_importance(model, test: pd.DataFrame, repeats: int, seed: int) -> dict[str, float]:
    base = within_campaign_auc(test.assign(score=predict(model, test)), "score")
    drops = {}
    for name, cols in FEATURE_GROUPS.items():
        d = []
        for r in range(repeats):
            perm = permuted_within_campaign(test, cols, seed + r)
            d.append(base - within_campaign_auc(perm.assign(score=predict(model, perm)), "score"))
        drops[name] = float(np.mean(d))
    return drops


def l1_path(data: pd.DataFrame, cs=(0.001, 0.003, 0.01, 0.03, 0.1, 0.3)) -> pd.DataFrame:
    """Standardised logistic coefficients as the L1 penalty loosens (C grows): which features enter first."""
    rows = []
    for c in cs:
        coefs = fit("logistic_l1", {"C": c}, data).named_steps["model"].coef_[0]
        rows += [
            {"C": c, "feature": f, "coef": float(v)} for f, v in zip(FEATURES, coefs, strict=True)
        ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- run


RULES = {"rule: coupon share": "coupon_line_share", "rule: past redeemer": "has_prior_redemption_f"}


@dataclass
class Phase4Results:
    data: pd.DataFrame  # model table with outer fold and out-of-fold scores
    fold_metrics: pd.DataFrame  # one row per fold x scorer
    summary: pd.DataFrame  # mean ± SD per scorer
    paired: pd.DataFrame  # per-fold model minus coupon-share rule
    chosen: dict = field(default_factory=dict)  # tuned parameters per fold and model
    importance: pd.DataFrame = None
    path: pd.DataFrame = None
    capture_curve: pd.DataFrame = None
    time_forward: pd.DataFrame = None  # overall and per campaign
    drift: pd.DataFrame = None
    best_model: str = ""
    grid: pd.DataFrame = None  # inner-CV score of every grid setting, per fold


def _scorers(models: list[str]) -> dict[str, str]:
    return {**RULES, **{m: f"score_{m}" for m in models}}


def run(campaign_household: pd.DataFrame, n_folds: int = 5, repeats: int = 2) -> Phase4Results:
    data = model_table(campaign_household)
    data["fold"] = grouped_folds(data, n_folds, seed=2026)
    models = list(GRIDS)
    for m in models:
        data[f"score_{m}"] = np.nan
    chosen, imp_rows, grids = {m: [] for m in models}, [], []
    for f in range(n_folds):
        train, test = data[data["fold"] != f], data[data["fold"] == f]
        for m in models:
            model, params, inner_auc, grid = tune_and_fit(train, m)
            chosen[m].append({"fold": f, **params, "inner_auc_within": inner_auc})
            grids.append(grid.assign(model=m, fold=f))
            data.loc[test.index, f"score_{m}"] = predict(model, test)
            for g, drop in group_importance(model, test, repeats, seed=100 * f).items():
                imp_rows.append({"model": m, "fold": f, "group": g, "auc_drop": drop})

    scorers = _scorers(models)
    fold_rows = []
    for f in range(n_folds):
        part = data[data["fold"] == f]
        for name, col in scorers.items():
            fold_rows.append({"fold": f, "scorer": name, **metrics(part, col)})
    fold_metrics = pd.DataFrame(fold_rows)
    summary = fold_metrics.groupby("scorer").agg(["mean", "std"]).drop(columns="fold")
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index().sort_values(
        "auc_within_mean", ascending=False, ignore_index=True
    )

    rule = fold_metrics[fold_metrics["scorer"] == "rule: coupon share"].set_index("fold")
    paired = []
    for m in models:
        mod = fold_metrics[fold_metrics["scorer"] == m].set_index("fold")
        diff = mod["auc_within"] - rule["auc_within"]
        paired.append({"model": m, "diff_mean": diff.mean(), "diff_sd": diff.std(),
                       "folds_better": int((diff > 0).sum()), "folds": n_folds})  # fmt: skip
    paired = pd.DataFrame(paired)
    best = paired.sort_values("diff_mean", ascending=False).iloc[0]["model"]

    shares = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    curve = pd.DataFrame([
        {"share_targeted": s, "random": s, "rule: coupon share": capture(data, "coupon_line_share", s),
         f"model: {best}": capture(data, f"score_{best}", s)}
        for s in shares
    ])  # fmt: skip

    return Phase4Results(
        data=data, fold_metrics=fold_metrics, summary=summary, paired=paired,
        chosen={m: pd.DataFrame(v) for m, v in chosen.items()},
        importance=pd.DataFrame(imp_rows), path=l1_path(data), capture_curve=curve,
        time_forward=time_forward(data, best), drift=drift_table(data), best_model=best,
        grid=pd.concat(grids, ignore_index=True),
    )  # fmt: skip


def time_forward(data: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Train on campaigns ended before TIME_CUTOFF, test on campaigns starting on/after it."""
    train = data[data["end_day"] < TIME_CUTOFF]
    test = data[data["start_day"] >= TIME_CUTOFF].copy()
    model, *_ = tune_and_fit(train, kind)
    test["score"] = predict(model, test)
    rows = [{"campaign": "ALL", "pairs": len(test), "redeemed": int(test[LABEL].sum()),
             "model_auc": within_campaign_auc(test, "score"),
             "rule_auc": within_campaign_auc(test, "coupon_line_share")}]  # fmt: skip
    for c, g in test.groupby("campaign"):
        rows.append({"campaign": str(c), "pairs": len(g), "redeemed": int(g[LABEL].sum()),
                     "model_auc": auc(g[LABEL].to_numpy(), g["score"].to_numpy()),
                     "rule_auc": auc(g[LABEL].to_numpy(), g["coupon_line_share"].to_numpy())})  # fmt: skip
    out = pd.DataFrame(rows)
    out.insert(
        1,
        "campaign_type",
        ["ALL"]
        + [
            test.loc[test["campaign"] == int(c), "campaign_type"].iloc[0]
            for c in out["campaign"][1:]
        ],
    )
    return out


def drift_table(data: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 400, 500, TIME_CUTOFF, 10_000]
    labels = ["< 400", "400-499", f"500-{TIME_CUTOFF - 1}", f">= {TIME_CUTOFF}"]
    d = data.assign(period=pd.cut(data["start_day"], bins, right=False, labels=labels))
    rows = []
    for p, g in d.groupby("period", observed=True):
        rows.append({"campaign start day": str(p), "pairs": len(g), "redemption_rate": g[LABEL].mean(),
                     "share_with_prior_redemption": g["has_prior_redemption_f"].mean(),
                     "share_no_earlier_campaign": (g["prior_campaigns_completed"] == 0).mean(),
                     "rule_auc_within": within_campaign_auc(g, "coupon_line_share")})  # fmt: skip
    return pd.DataFrame(rows)


def write_gold(spark: SparkSession, io, res: Phase4Results, run_id: str) -> None:
    cols = [*KEY, "campaign_type", LABEL, "fold", "coupon_line_share", f"score_{res.best_model}"]
    out = res.data[cols].rename(columns={f"score_{res.best_model}": "score"})
    out["model"] = res.best_model
    out["run_id"] = run_id
    io.write(spark.createDataFrame(out), "gold", "campaign_response_scores")
