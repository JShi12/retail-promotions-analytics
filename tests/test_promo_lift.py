import numpy as np
import pandas as pd
import pytest

from retail_promo.analysis import promo_lift as pl
from retail_promo.analysis.phase2 import raw_contrast

TRUE_MAILER_LIFT = 0.8  # +80% units in mailer weeks
TRUE_DISPLAY_COEF = 1.5


def simulated_panel(seed: int = 7, products: int = 300, weeks: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pid = np.repeat(np.arange(products), weeks)
    week = np.tile(np.arange(1, weeks + 1), products)
    base = np.repeat(rng.lognormal(0.5, 0.8, products), weeks)
    season = 1 + 0.3 * np.sin(week / 6)
    mailer = (rng.random(products * weeks) < 0.1).astype(float)
    display = np.where(
        rng.random(products * weeks) < 0.4, rng.uniform(0, 0.2, products * weeks), 0.0
    )
    mu = base * season * np.exp(np.log1p(TRUE_MAILER_LIFT) * mailer + TRUE_DISPLAY_COEF * display)
    return pd.DataFrame(
        {
            "product_id": pid,
            "commodity_desc": np.where(pid % 2 == 0, "A", "B"),
            "week_no": week,
            "units": rng.poisson(mu),
            "mailer_share": mailer,
            "display_share": display,
        }
    )


def test_product_lift_recovers_known_effects():
    rows = {r["term"]: r for r in pl.product_lift(simulated_panel())}
    mailer, display = rows["mailer_share"], rows["display_share"]
    assert mailer["lift_low"] < TRUE_MAILER_LIFT < mailer["lift_high"]
    assert mailer["lift"] == pytest.approx(TRUE_MAILER_LIFT, abs=0.1)
    assert display["coef_low"] < TRUE_DISPLAY_COEF < display["coef_high"]
    assert display["lift_definition"] == "b" and mailer["lift_definition"] == "exp(b)-1"


def test_category_frame_shares_are_volume_weighted():
    df = pd.DataFrame(
        {
            "product_id": [1, 1, 2, 2],
            "commodity_desc": ["A"] * 4,
            "week_no": [1, 2, 1, 2],
            "units": [3, 3, 1, 1],  # product 1 has 75% of category units
            "mailer_share": [1.0, 0.0, 0.0, 1.0],
            "display_share": [0.0, 0.0, 0.0, 0.0],
        }
    )
    out = pl.category_frame(df, ["A"]).set_index("week_no")
    assert out.loc[1, "units"] == 4
    assert out.loc[1, "cat_mailer_share"] == pytest.approx(0.75)
    assert out.loc[2, "cat_mailer_share"] == pytest.approx(0.25)


def test_raw_contrast_ratios():
    df = pd.DataFrame(
        {"units": [1, 1, 4, 10], "display_share": [0, 0, 0.1, 0], "mailer_share": [0, 0, 0, 1.0]}
    )
    out = raw_contrast(df).set_index("status")
    assert out.loc["none", "ratio_to_none"] == 1.0
    assert out.loc["mailer only", "ratio_to_none"] == 10.0
    assert out.loc["display only", "mean_units"] == 4.0


def _panel_spark(spark, rows):
    cols = "product_id int, commodity_desc string, week_no int, units long, sales_value double, "
    cols += "display_share double, mailer_share double, avg_discount_depth double"
    return spark.createDataFrame(rows, cols)


def test_model_frame_lags_leads_and_active_span(spark):
    rows = [
        (1, "A", 8, 0, 0.0, 0.0, 1.0, None),
        (1, "A", 9, 2, 2.0, 0.0, 0.0, 0.1),
        (1, "A", 10, 0, 0.0, 0.0, 1.0, None),
        (1, "A", 11, 1, 1.0, 0.0, 0.0, 0.0),
        (1, "A", 12, 0, 0.0, 0.0, 0.0, None),
    ]
    out = {r["week_no"]: r for r in pl.model_frame(_panel_spark(spark, rows), 9, 11).collect()}
    assert sorted(out) == [9, 10, 11]  # window applied
    assert out[9]["mailer_lag"] == 1.0  # lag taken from week 8, before the window filter
    assert out[11]["mailer_lead"] == 0.0 and out[10]["mailer_lead"] == 0.0
    assert out[9]["active"] and out[10]["active"] and out[11]["active"]  # bought in weeks 9 and 11


def test_select_categories_requires_promotion_variation(spark):
    rows = []
    for pid in range(1, 4):  # category BIG: highest sales but never in the mailer
        rows += [(pid, "BIG", w, 5, 100.0, 0.5, 0.0, None) for w in range(1, 4)]
    for pid in range(4, 7):  # category OK: mailer in every week
        rows += [(pid, "OK", w, 1, 1.0, 0.0, 1.0, None) for w in range(1, 4)]
    cats = pl.select_categories(
        _panel_spark(spark, rows), n=5, min_products=3, min_mailer_weeks=5, min_display_weeks=5
    )
    assert cats["commodity_desc"].tolist() == ["OK"]
    assert not cats.loc[0, "display_estimable"]
    assert cats.loc[0, "sales_rank"] == 2
