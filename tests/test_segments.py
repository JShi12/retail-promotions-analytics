import numpy as np
import pandas as pd
import pytest

from retail_promo.analysis import segments as sg


def _lines(spark, rows):
    cols = "household_key int, commodity_desc string, store_id int, promo_observed boolean, "
    cols += (
        "sales_value double, shelf_value double, disc double, pl_spend double, promo_spend double"
    )
    return spark.createDataFrame(rows, cols)


def test_expected_is_leave_one_out(spark):
    # One category-store cell. Household 1 spends 100 with 60 promoted; 2 and 3 spend 100 with 20 and 10.
    rows = [
        (1, "SOUP", 1, True, 100.0, 100.0, 0.0, 0.0, 60.0),
        (2, "SOUP", 1, True, 100.0, 100.0, 0.0, 0.0, 20.0),
        (3, "SOUP", 1, True, 100.0, 100.0, 0.0, 0.0, 10.0),
    ]
    out = sg._actual_expected(
        _lines(spark, rows), "promo", "promo_spend", "sales_value", "promo_observed"
    )
    r = {x["household_key"]: x for x in out.collect()}
    assert r[1]["promo_actual"] == pytest.approx(0.60)
    assert r[1]["promo_expected"] == pytest.approx((20 + 10) / 200)  # its own spend left out
    assert r[3]["promo_expected"] == pytest.approx((60 + 20) / 200)


def test_heavily_promoted_category_alone_gives_no_excess(spark):
    # Everyone buys MEAT at a 50% promoted share: buying promoted categories is not deal-proneness.
    rows = [(h, "MEAT", 1, True, 100.0, 100.0, 0.0, 0.0, 50.0) for h in (1, 2, 3)]
    out = sg._actual_expected(
        _lines(spark, rows), "promo", "promo_spend", "sales_value", "promo_observed"
    )
    for r in out.collect():
        assert r["promo_actual"] - r["promo_expected"] == pytest.approx(0.0)


def test_cell_with_a_single_buyer_has_no_benchmark(spark):
    rows = [
        (1, "SOUP", 1, True, 100.0, 100.0, 0.0, 0.0, 60.0),
        (2, "SOUP", 1, True, 100.0, 100.0, 0.0, 0.0, 20.0),
        (1, "RARE", 9, True, 50.0, 50.0, 0.0, 0.0, 50.0),  # only household 1 buys here
    ]
    out = sg._actual_expected(
        _lines(spark, rows), "promo", "promo_spend", "sales_value", "promo_observed"
    )
    r = {x["household_key"]: x for x in out.collect()}
    assert r[1]["promo_actual"] == pytest.approx(0.60)  # the RARE cell is left out of actual too
    assert r[1]["promo_expected"] == pytest.approx(0.20)


def test_jaccard_stability():
    full = pd.Series([0, 0, 0, 1, 1, 1], index=range(6))
    same = pd.Series([5, 5, 5, 7, 7, 7], index=range(6))  # same partition, other label numbers
    assert sg.jaccard_stability(full, same) == {0: 1.0, 1: 1.0}
    boot = pd.Series([5, 5, 7, 7], index=[0, 1, 2, 3])  # households 4 and 5 not sampled
    jac = sg.jaccard_stability(full, boot)
    assert jac[0] == pytest.approx(2 / 3)  # {0,1,2} vs best match {0,1}
    assert jac[1] == pytest.approx(1 / 2)  # {3} vs best match {2,3}


def test_shuffled_keeps_each_columns_values():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    s = sg.shuffled(df, ["a", "b"], seed=1)
    assert sorted(s["a"]) == [1, 2, 3, 4] and sorted(s["b"]) == [10, 20, 30, 40]


def test_fit_best_recovers_separated_clusters(spark):
    rng = np.random.default_rng(0)
    centres = np.array([[0, 0, 0], [5, 5, 0], [0, 5, 5]])
    pts = np.vstack([c + rng.normal(0, 0.3, (40, 3)) for c in centres])
    df = spark.createDataFrame(
        pd.DataFrame(
            {"household_key": range(120), **{c: pts[:, i] for i, c in enumerate(sg.INPUTS)}}
        )
    )
    X = sg.scaler(df).transform(df)
    labels = sg._labels(sg.fit_best(X, 3, starts=3), X).sort_index().to_numpy()
    truth = np.repeat([0, 1, 2], 40)
    assert (
        pd.crosstab(truth, labels).gt(0).sum(axis=1).eq(1).all()
    )  # each true cluster maps to one label


def test_pick_k_and_labels():
    summary = pd.DataFrame(
        {
            "k": [2, 3, 4, 5],
            "all_stable": [True, True, False, True],
            "gap": [0.003, 0.08, 0.02, 0.009],
        }
    )
    assert sg.pick_k(summary) == 3  # largest gap among stable k, not the largest stable k
    with pytest.raises(ValueError):
        sg.pick_k(summary.assign(all_stable=False))
    assert sg.short_label(np.array([0.6, -0.2, 1.6])) == "More promotion, more coupons"
    assert sg.short_label(np.array([0.1, 0.0, -0.1])) == "Average on all three"
