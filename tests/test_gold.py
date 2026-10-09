import pytest

from retail_promo.gold import tables as gold
from retail_promo.silver import campaigns, causal, demographics, products, transactions


def _txn(hh, basket, pid, day, qty="1", sales="2.00", disc="0", store="1", coupon="0"):
    week = (int(day) + 1) // 7 + 1
    return {
        "household_key": hh, "basket_id": basket, "day": day, "product_id": pid, "quantity": qty,
        "sales_value": sales, "store_id": store, "retail_disc": disc, "trans_time": "1200",
        "week_no": str(week), "coupon_disc": coupon, "coupon_match_disc": "0",
    }  # fmt: skip


@pytest.fixture
def silver(make_bronze):
    prod, _ = products.build(
        make_bronze([
            {"product_id": "10", "manufacturer": "1", "department": "GROCERY", "brand": "Private",
             "commodity_desc": "SOUP", "sub_commodity_desc": "S", "curr_size_of_product": "1"},
            {"product_id": "11", "manufacturer": "1", "department": "PRODUCE", "brand": "National",
             "commodity_desc": "FRUIT", "sub_commodity_desc": "F", "curr_size_of_product": "1"},
            {"product_id": "99", "manufacturer": "1", "department": "KIOSK-GAS", "brand": "Private",
             "commodity_desc": "COUPON/MISC ITEMS", "sub_commodity_desc": "G", "curr_size_of_product": ""},
        ]),
        "r",
    )  # fmt: skip
    caus, _ = causal.build(
        make_bronze([
            {"product_id": "10", "store_id": "1", "week_no": "11", "display": "7", "mailer": "0"},
            {"product_id": "10", "store_id": "2", "week_no": "11", "display": "0", "mailer": "A"},
            {"product_id": "11", "store_id": "2", "week_no": "12", "display": "1", "mailer": "0"},
        ]),
        "r",
    )  # fmt: skip
    txn, _ = transactions.build(
        make_bronze([
            # household 1: day 60 (week 9), day 70 (week 11, product 10 on display in store 1), fuel, day 90
            _txn("1", "100", "10", "60", qty="1", sales="2.00"),
            _txn("1", "101", "10", "70", qty="2", sales="3.00", disc="-1.00"),
            _txn("1", "101", "99", "70", qty="40000", sales="90.00"),
            _txn("1", "102", "11", "90", qty="1", sales="5.00", coupon="-1.00"),
            # household 2: stores 1 and 2, weeks 11-12
            _txn("2", "200", "11", "70", qty="1", sales="4.00", store="2"),
            _txn("2", "201", "10", "77", qty="3", sales="6.00", store="1"),
        ]),
        prod, caus, "r", 9, 101,
    )  # fmt: skip
    demo, _ = demographics.build(
        make_bronze([
            {"classification_1": "Age Group3", "classification_2": "X", "classification_3": "Level4",
             "homeowner_desc": "Renter", "classification_5": "Group2", "classification_4": "2",
             "kid_category_desc": "1", "household_key": "1"},
        ]),
        "r",
    )  # fmt: skip
    cdesc, _ = campaigns.build_campaign_desc(
        make_bronze([
            {"description": "TypeA", "campaign": "1", "start_day": "65", "end_day": "80"},
            {"description": "TypeB", "campaign": "2", "start_day": "85", "end_day": "100"},
        ]),
        "r",
    )  # fmt: skip
    ctab, _ = campaigns.build_campaign_table(
        make_bronze([
            {"description": "TypeA", "household_key": "1", "campaign": "1"},
            {"description": "TypeB", "household_key": "1", "campaign": "2"},
            {"description": "TypeA", "household_key": "2", "campaign": "1"},
        ]),
        cdesc, "r",
    )  # fmt: skip
    coup, _ = campaigns.build_coupons(
        make_bronze([
            {"coupon_upc": "1000", "product_id": "10", "campaign": "1"},
            {"coupon_upc": "2000", "product_id": "11", "campaign": "2"},
        ]),
        "r",
    )  # fmt: skip
    red, _ = campaigns.build_redemptions(
        make_bronze([{"household_key": "1", "day": "70", "coupon_upc": "1000", "campaign": "1"}]),
        cdesc,
        "r",
    )
    return {
        "prod": prod, "caus": caus, "txn": txn, "demo": demo,
        "cdesc": cdesc, "ctab": ctab, "coup": coup, "red": red,
    }  # fmt: skip


def test_household_features(silver):
    hf = {
        r["household_key"]: r
        for r in gold.household_features(silver["txn"], silver["prod"], silver["demo"]).collect()
    }
    h1 = hf[1]
    assert h1["n_baskets"] == 3
    assert h1["total_spend"] == pytest.approx(10.0)  # fuel excluded
    assert h1["non_merch_spend_share"] == pytest.approx(90 / 100)
    assert h1["private_label_share"] == pytest.approx(5 / 10)
    assert h1["recency_days"] == 712 - 90
    # promo-observed merchandise spend: day 60 (not promoted) 2 + day 70 (display) 3 + day 90 5 = 10
    assert h1["promo_spend_share"] == pytest.approx(3 / 10)
    assert h1["avg_discount_depth"] == pytest.approx(1 / 11)
    assert h1["has_demographics"] and h1["classification_3_ord"] == 4
    assert not hf[2]["has_demographics"]


def test_campaign_household_uses_only_pre_campaign_data(silver):
    ch = gold.campaign_household(
        silver["txn"],
        silver["prod"],
        silver["ctab"],
        silver["cdesc"],
        silver["coup"],
        silver["red"],
    )
    rows = {(r["household_key"], r["campaign"]): r for r in ch.collect()}
    assert len(rows) == 3

    h1c1 = rows[(1, 1)]  # starts day 65: only the day-60 basket counts
    assert h1c1["n_baskets"] == 1 and h1c1["last_day"] == 60
    assert h1c1["total_spend"] == pytest.approx(2.0)
    assert h1c1["redeemed"] == 1 and h1c1["prior_redemptions"] == 0
    assert h1c1["pre_spend_on_coupon_products"] == pytest.approx(
        2.0
    )  # product 10 is on coupon 1000
    assert h1c1["campaign_type"] == "TypeA" and h1c1["n_coupons"] == 1

    h1c2 = rows[(1, 2)]  # starts day 85: campaign 1 (ended day 80) and its redemption are prior
    assert h1c2["last_day"] == 70 and h1c2["redeemed"] == 0
    assert h1c2["prior_campaigns_completed"] == 1 and h1c2["prior_redemptions"] == 1
    assert h1c2["has_prior_redemption"]

    h2c1 = rows[(2, 1)]  # household 2 has no purchases before day 65
    assert h2c1["n_baskets"] is None and h2c1["redeemed"] == 0

    assert all(r["last_day"] is None or r["last_day"] < r["start_day"] for r in rows.values())


def test_product_week_panel(silver):
    panel = gold.product_week_panel(
        silver["txn"], silver["caus"], silver["prod"], 9, 12, min_weeks=1
    )
    rows = {(r["product_id"], r["week_no"]): r for r in panel.collect()}
    assert {pid for pid, _ in rows} == {10, 11}  # fuel product excluded
    assert len(rows) == 8  # 2 products x weeks 9-12, zero weeks included
    assert rows[(10, 10)]["units"] == 0 and rows[(10, 10)]["avg_shelf_price"] is None
    # Week 11 in-scope merchandise lines: store 1 has 1 (hh1 product 10), store 2 has 1 (hh2 product 11).
    # Product 10 is on display in store 1 (weight 0.5) and in the mailer in store 2 (weight 0.5).
    w11 = rows[(10, 11)]
    assert w11["display_share"] == pytest.approx(0.5) and w11["mailer_share"] == pytest.approx(0.5)
    assert w11["units"] == 2 and w11["avg_shelf_price"] == pytest.approx(2.0)  # (3 + 1) / 2
    # Week 12: only store 1 has lines, so store 2 (product 11 display) carries no traffic weight.
    assert rows[(11, 12)]["display_share"] == pytest.approx(0.0)
    assert rows[(11, 12)]["n_stores_display"] == 1
