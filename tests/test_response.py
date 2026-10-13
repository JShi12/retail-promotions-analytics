import numpy as np
import pandas as pd
import pytest

from retail_promo.analysis import response as rs


def _pairs(n_households=40, campaigns=(1, 2, 3), seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_households):
        for c in campaigns:
            rows.append({"household_key": h, "campaign": c, "redeemed": int(rng.random() < 0.3)})
    return pd.DataFrame(rows)


def test_grouped_folds_keep_households_together_and_balance_redemptions():
    df = _pairs()
    folds = rs.grouped_folds(df, 4, seed=1)
    assert df.assign(f=folds).groupby("household_key")["f"].nunique().max() == 1
    pos = df.assign(f=folds).groupby("f")["redeemed"].sum()
    assert pos.max() - pos.min() <= 3


def test_excluded_columns_never_reach_the_model():
    assert not set(rs.EXCLUDED) & set(rs.FEATURES)
    assert "redeemed" not in rs.FEATURES and "n_redemptions" not in rs.FEATURES


def test_auc_and_within_campaign_auc():
    assert rs.auc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.3, 0.4])) == 1.0
    assert rs.auc(np.array([0, 1]), np.array([0.5, 0.5])) == 0.5  # ties count half
    # Campaign A is perfectly ranked (1 redemption), campaign B is inverted (3 redemptions).
    df = pd.DataFrame({
        "campaign": ["A"] * 2 + ["B"] * 4,
        "redeemed": [0, 1, 1, 1, 1, 0],
        "score": [0.1, 0.9, 0.1, 0.2, 0.3, 0.9],
    })  # fmt: skip
    assert rs.within_campaign_auc(df, "score") == pytest.approx((1.0 * 1 + 0.0 * 3) / 4)


def test_capture_and_lift_select_top_k_within_each_campaign():
    df = pd.DataFrame({
        "campaign": [1] * 10 + [2] * 10,
        "redeemed": [1, 0, 0, 0, 0, 0, 0, 0, 0, 0] + [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        "score": list(range(10, 0, -1)) + list(range(10, 0, -1)),  # campaign 1 right, campaign 2 wrong
    })  # fmt: skip
    assert rs.capture(df, "score", 0.1) == pytest.approx(0.5)  # top 1 of each: 1 of 2 redemptions
    assert rs.capture(df, "score", 1.0) == pytest.approx(1.0)
    # top 10%: 1 redemption in 2 households = 50%, overall 10% -> lift 5
    assert rs.within_campaign_lift(df, "score") == pytest.approx(5.0)


def test_average_precision():
    assert rs.average_precision(np.array([1, 0, 1]), np.array([0.9, 0.5, 0.1])) == pytest.approx(
        (1 + 2 / 3) / 2
    )


def test_permutation_stays_within_campaign():
    df = pd.DataFrame(
        {"campaign": [1, 1, 1, 2, 2, 2], "x": [1, 2, 3, 10, 20, 30], "c": [5, 5, 5, 9, 9, 9]}
    )
    out = rs.permuted_within_campaign(df, ["x", "c"], seed=3)
    assert sorted(out.loc[out.campaign == 1, "x"]) == [1, 2, 3]
    assert sorted(out.loc[out.campaign == 2, "x"]) == [10, 20, 30]
    assert (out["c"] == df["c"]).all()  # campaign-level values unchanged


def test_logistic_pipeline_learns_a_simple_signal():
    rng = np.random.default_rng(5)
    n = 400
    d = pd.DataFrame({f: rng.normal(size=n) for f in rs.FEATURES})
    d["household_key"] = np.arange(n)
    d["campaign"] = rng.integers(1, 4, n)
    d["redeemed"] = (d["coupon_line_share"] + rng.normal(0, 0.5, n) > 0.8).astype(int)
    d.loc[::7, "promo_spend_share"] = np.nan  # missing values are imputed inside the pipeline
    for kind, params in [("logistic_l1", {"C": 0.1}), ("gbt", {"max_depth": 1, "max_iter": 50})]:
        model = rs.fit(kind, params, d)
        assert rs.within_campaign_auc(d.assign(score=rs.predict(model, d)), "score") > 0.9
