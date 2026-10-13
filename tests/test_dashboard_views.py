"""The dashboard views must reproduce the notebooks' numbers, not just run."""

import numpy as np
import pandas as pd
import pytest

from retail_promo.analysis import response, segments
from retail_promo.dashboard import views

DBS = ("dtest_gold", "dtest_ops", "dtest_dash")


@pytest.fixture
def catalog(spark):
    for db in DBS:
        spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")
        spark.sql(f"CREATE DATABASE {db}")

    def save(pdf: pd.DataFrame, table: str):
        spark.createDataFrame(pdf).write.mode("overwrite").saveAsTable(table)

    yield save
    for db in DBS:
        spark.sql(f"DROP DATABASE IF EXISTS {db} CASCADE")


def _create(spark):
    views.create_views(spark, gold="dtest_gold", ops="dtest_ops", dashboard="dtest_dash")


def _minimal_tables(save, rng):
    """Every table the views read, so all views can be created."""
    n = 60
    save(pd.DataFrame({"household_key": range(n), "total_spend": rng.uniform(100, 1000, n),
                       "baskets_per_week": rng.uniform(0.2, 3, n), "avg_basket_spend": rng.uniform(10, 50, n)}),
         "dtest_gold.household_features")  # fmt: skip
    save(pd.DataFrame({"week_no": [1, 2, 20], "sales": [10.0, 20.0, 30.0], "sales_per_household": [1.0, 2.0, 3.0],
                       "active_households": [10, 10, 10], "baskets": [2, 4, 6], "avg_basket_spend": [5.0, 5.0, 5.0],
                       "loyalty_discount": [1.0, 1.0, 1.0], "promo_sales": [None, None, 6.0],
                       "promo_observed_sales": [None, None, 24.0], "promo_share_of_sales": [None, None, 0.25],
                       "ramp_up": [True, True, False]}), "dtest_gold.weekly_kpis")  # fmt: skip
    lift = pd.DataFrame({
        "model": ["product", "product", "product", "timing"], "scope": ["pooled", "category", "pooled", "pooled"],
        "category": ["ALL", "SOUP", "ALL", "ALL"], "variant": ["main", "main", "weeks 14-101", "main"],
        "term": ["mailer_share", "mailer_share", "mailer_share", "mailer_lag"],
        "lift": [0.8, 0.5, 0.79, 0.07], "lift_low": [0.7, 0.4, 0.69, 0.03], "lift_high": [0.9, 0.6, 0.89, 0.1],
        "n_obs": [100, 40, 90, 100],
    })  # fmt: skip
    save(lift, "dtest_gold.promo_lift_by_category")
    seg = pd.DataFrame({"household_key": range(n), "segment": rng.integers(1, 4, n), "low_activity": [False] * (n - 5) + [True] * 5,
                        **{c: rng.uniform(0, 0.4, n) for c in ["promo_actual", "promo_expected", "disc_actual",
                                                               "disc_expected", "pl_actual", "pl_expected"]}})  # fmt: skip
    seg["segment_label"] = "S" + seg["segment"].astype(str)
    save(seg, "dtest_gold.household_segments")
    camp = pd.DataFrame({"household_key": np.repeat(range(n), 2), "campaign": np.tile([1, 2], n),
                         "redeemed": rng.integers(0, 2, 2 * n)})  # fmt: skip
    save(camp, "dtest_gold.campaign_household")
    scores = camp.assign(
        score=rng.choice([0.1, 0.2, 0.3, 0.5], 2 * n),
        coupon_line_share=rng.choice([0.0, 0.01, 0.05], 2 * n),
    )
    save(scores, "dtest_gold.campaign_response_scores")
    runs = pd.DataFrame({"run_id": ["old", "new", "new"], "status": ["succeeded", "started", "succeeded"],
                         "finished_at": pd.to_datetime(["2026-01-01", "2026-02-01", "2026-02-02"]), "detail": ["", "", ""]})  # fmt: skip
    save(runs, "dtest_ops.pipeline_runs")
    checks = pd.DataFrame({"run_id": ["old", "new", "new"], "layer": ["silver"] * 3, "table_name": ["t"] * 3,
                           "check_id": ["a", "a", "b"], "severity": ["error", "error", "warning"],
                           "passed": [False, True, False], "observed": ["1", "0", "2"], "expected": ["0"] * 3})  # fmt: skip
    save(checks, "dtest_ops.dq_check_results")
    save(pd.DataFrame({"run_id": ["old", "new"], "table_name": ["t", "t"], "rule_id": ["r", "r"],
                       "action": ["flagged", "flagged"], "n_rows": [5, 7]}), "dtest_gold.dq_summary")  # fmt: skip
    return seg, camp, scores


def test_views_reproduce_notebook_calculations(spark, catalog):
    rng = np.random.default_rng(0)
    seg, camp, scores = _minimal_tables(catalog, rng)
    _create(spark)

    # Capture curve: same top-k-per-campaign rule and tie-break as notebook 30.
    data = scores.sort_values(["household_key", "campaign"], ignore_index=True)
    curve = spark.table("dtest_dash.capture_curve").toPandas().sort_values("share_targeted")
    for row in curve.itertuples():
        share = float(row.share_targeted)
        assert float(row.model) == pytest.approx(response.capture(data, "score", share))
        assert float(row.rule_coupon_share) == pytest.approx(
            response.capture(data, "coupon_line_share", share)
        )

    # Segment profile: same medians and shares as notebook 20's profiles().
    hf = spark.table("dtest_gold.household_features").toPandas().assign(
        share_grocery=0.0, share_drug_gm=0.0, share_produce=0.0, share_meat=0.0, share_deli=0.0,
        coupon_line_share=0.0, has_demographics=False,
    )  # fmt: skip
    expected = segments.profiles(seg, hf, camp).set_index("segment")
    got = spark.table("dtest_dash.segment_profile").toPandas().set_index("segment")
    for col in [
        "households",
        "share_of_spend",
        "promo_actual",
        "promo_expected",
        "pl_actual",
        "baskets_per_week",
    ]:
        assert got.loc[expected.index, col].astype(float).to_numpy() == pytest.approx(
            expected[col].astype(float).to_numpy()
        )
    red = spark.table("dtest_dash.segment_redemption").toPandas().set_index("segment")
    assert red.loc[expected.index, "redemption_rate"].to_numpy() == pytest.approx(
        expected["redemption_rate"].to_numpy()
    )


def test_overview_lift_and_dq_views(spark, catalog):
    _minimal_tables(catalog, np.random.default_rng(1))
    _create(spark)
    head = spark.table("dtest_dash.kpi_headline").first()
    assert head["total_sales"] == 60.0 and head["households"] == 60
    assert float(head["promo_share_of_sales"]) == pytest.approx(
        0.25
    )  # only weeks where it is recorded
    weekly = {r["week_no"]: r for r in spark.table("dtest_dash.weekly_kpis").collect()}
    assert weekly[1]["period"].startswith("Panel ramp-up") and weekly[20]["period"] == "Full panel"
    assert weekly[1]["days_in_week"] == 5 and not weekly[1]["full_week"]  # days 1-5
    assert weekly[20]["days_in_week"] == 7 and weekly[20]["full_week"]

    mailer = {
        r["category"]: r["lift"]
        for r in spark.table("dtest_dash.mailer_lift_by_category").collect()
    }
    assert mailer == {"ALL PRODUCTS": 0.8, "SOUP": 0.5}  # main variant only
    assert spark.table("dtest_dash.lift_sensitivity").count() == 2
    assert spark.table("dtest_dash.promo_timing").first()["week"] == "Week after the mailer"

    checks = spark.table("dtest_dash.dq_latest_checks").toPandas()
    assert set(checks["run_id"]) == {"new"} and len(checks) == 2  # latest succeeded run only
    status = spark.table("dtest_dash.dq_check_status").toPandas().set_index("severity")
    assert status.loc["error", "failed"] == 0 and status.loc["warning", "failed"] == 1
    assert spark.table("dtest_dash.dq_rule_counts").first()["n_rows"] == 7
