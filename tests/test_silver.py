from decimal import Decimal

from retail_promo.silver import campaigns, causal, demographics, products, transactions


def _product(pid, dept="GROCERY", commodity="SOUP", brand="National", size="10 OZ"):
    return {
        "product_id": pid, "manufacturer": "1", "department": dept, "brand": brand,
        "commodity_desc": commodity, "sub_commodity_desc": "SUB", "curr_size_of_product": size,
    }  # fmt: skip


def _causal(pid, store, week, display, mailer):
    return {
        "product_id": pid,
        "store_id": store,
        "week_no": week,
        "display": display,
        "mailer": mailer,
    }


def _txn(basket, pid, qty="1", sales="2.00", disc="0", day="70", week="11", store="1", **kw):
    row = {
        "household_key": "1", "basket_id": basket, "day": day, "product_id": pid, "quantity": qty,
        "sales_value": sales, "store_id": store, "retail_disc": disc, "trans_time": "1200", "week_no": week,
        "coupon_disc": "0", "coupon_match_disc": "0",
    }  # fmt: skip
    row.update(kw)
    return row


# ---------------------------------------------------------------- products


def test_products_blank_hierarchy_becomes_unknown_and_is_logged(make_bronze):
    bronze = make_bronze([_product("1"), _product("2", dept=" ", commodity=" ", size=" ")])
    df, log = products.build(bronze, "r")
    row = {r["product_id"]: r for r in df.collect()}[2]
    assert row["department"] == "UNKNOWN" and row["commodity_desc"] == "UNKNOWN"
    assert row["curr_size_of_product"] is None
    assert {r["rule_id"] for r in log.collect()} == {"blank_hierarchy", "blank_size_to_null"}


def test_products_non_merchandise_and_private_label(make_bronze):
    bronze = make_bronze([_product("1", dept="KIOSK-GAS"), _product("2", commodity="COUPON/MISC ITEMS"),
                          _product("3", brand="Private")])  # fmt: skip
    rows = {r["product_id"]: r for r in products.build(bronze, "r")[0].collect()}
    assert rows[1]["is_non_merchandise"] and rows[2]["is_non_merchandise"]
    assert not rows[3]["is_non_merchandise"] and rows[3]["is_private_label"]


def test_cast_failure_is_flagged_not_raised(make_bronze):
    df, _ = products.build(make_bronze([_product("abc")]), "r")
    row = df.first()
    assert row["product_id"] is None
    assert "cast_failed:product_id" in row["dq_flags"]


# ---------------------------------------------------------------- causal


def test_causal_or_collapse(make_bronze):
    bronze = make_bronze([
        _causal("10", "1", "20", "0", "A"), _causal("10", "1", "20", "7", "A"),   # display conflict
        _causal("11", "1", "20", "0", "D"), _causal("11", "1", "20", "0", "A"),   # two mailer codes
        _causal("12", "1", "20", "3", "0"),                                       # no conflict
    ])  # fmt: skip
    df, log = causal.build(bronze, "r")
    rows = {r["product_id"]: r for r in df.collect()}
    assert df.count() == 3
    assert rows[10]["on_display"] and rows[10]["display_code"] == "7" and rows[10]["on_mailer"]
    assert rows[10]["display_codes_raw"] == ["0", "7"]
    assert "causal_conflicting_duplicate" in rows[10]["dq_flags"]
    assert rows[11]["mailer_code"] == "A"  # alphabetically first of two non-zero codes
    assert not rows[11]["on_display"]
    assert rows[12]["display_codes_raw"] is None and rows[12]["dq_flags"] == []
    collapse_log = [r for r in log.collect() if r["rule_id"] == "causal_conflicting_duplicate"]
    assert len(collapse_log) == 4  # both source rows of each collapsed key
    assert {r["action"] for r in collapse_log} == {"modified"}


def test_causal_invalid_code_flagged(make_bronze):
    df, _ = causal.build(make_bronze([_causal("10", "1", "20", "8", "A")]), "r")
    assert "invalid_display_code" in df.first()["dq_flags"]


# ---------------------------------------------------------------- transactions


def _txn_inputs(make_bronze):
    prod, _ = products.build(make_bronze([_product("10"), _product("99", dept="KIOSK-GAS")]), "r")
    caus, _ = causal.build(make_bronze([_causal("10", "1", "11", "7", "0")]), "r")
    return prod, caus


def test_transactions_flags_and_prices(make_bronze):
    prod, caus = _txn_inputs(make_bronze)
    bronze = make_bronze([
        _txn("1", "10", qty="2", sales="2.00", disc="-1.34"),        # guide example: shelf price 1.67
        _txn("2", "10", qty="0", sales="0.50"),                      # quantity 0 with sales
        _txn("3", "99", qty="50000", sales="100.00"),                # fuel
        _txn("4", "10", disc="0.50"),                                # positive discount
        _txn("5", "10", sales="5.551115E-17"),                       # float noise
        _txn("6", "10", day="7", week="1"),                          # week formula broken
    ])  # fmt: skip
    df, log = transactions.build(bronze, prod, caus, "r")
    rows = {r["basket_id"]: r for r in df.collect()}
    assert rows[1]["shelf_price"] == Decimal("1.6700")
    assert rows[1]["discount_depth"] == Decimal("0.4012")  # 1.34 / 3.34
    assert rows[2]["shelf_price"] is None
    assert {"quantity_zero", "quantity_zero_sales_positive"} <= set(rows[2]["dq_flags"])
    assert rows[3]["is_non_merchandise"] and "non_merchandise" in rows[3]["dq_flags"]
    assert "retail_disc_positive" in rows[4]["dq_flags"]
    assert rows[5]["sales_value"] == Decimal("0.00") and "money_rounded" in rows[5]["dq_flags"]
    assert "week_day_mismatch" in rows[6]["dq_flags"]
    assert log.filter("rule_id = 'quantity_zero'").count() == 1


def test_transactions_promo_status_unknown_outside_causal_scope(make_bronze):
    prod, caus = _txn_inputs(make_bronze)
    bronze = make_bronze([
        _txn("1", "10", day="70", week="11", store="1"),   # promoted (display 7)
        _txn("2", "10", day="77", week="12", store="1"),   # causal store, not listed -> not promoted
        _txn("3", "10", day="70", week="11", store="2"),   # store not in causal -> unknown
        _txn("4", "10", day="20", week="3", store="1"),    # week before causal coverage -> unknown
    ])  # fmt: skip
    rows = {r["basket_id"]: r for r in transactions.build(bronze, prod, caus, "r")[0].collect()}
    assert rows[1]["on_display"] is True and rows[1]["on_mailer"] is False
    assert rows[2]["on_display"] is False and rows[2]["promo_observed"]
    assert rows[3]["on_display"] is None and not rows[3]["promo_observed"]
    assert rows[4]["on_display"] is None


# ---------------------------------------------------------------- demographics and campaigns


def test_demographics_ordinals_and_unexpected_values(make_bronze):
    base = {"classification_1": "Age Group3", "classification_2": "X", "classification_3": "Level12",
            "homeowner_desc": "Renter", "classification_5": "Group2", "classification_4": "5+",
            "kid_category_desc": "None/Unknown", "household_key": "1"}  # fmt: skip
    odd = {**base, "household_key": "2", "classification_2": "W"}
    rows = {
        r["household_key"]: r
        for r in demographics.build(make_bronze([base, odd]), "r")[0].collect()
    }
    assert rows[1]["classification_1_ord"] == 3 and rows[1]["classification_3_ord"] == 12
    assert rows[1]["classification_4_ord"] == 5 and rows[1]["kids_ord"] is None
    assert rows[1]["dq_flags"] == [] and "unexpected_category" in rows[2]["dq_flags"]


def test_campaigns(make_bronze):
    desc, _ = campaigns.build_campaign_desc(
        make_bronze([
            {"description": "TypeA", "campaign": "1", "start_day": "100", "end_day": "150"},
            {"description": "TypeB", "campaign": "2", "start_day": "650", "end_day": "719"},
        ]),
        "r",
    )  # fmt: skip
    d = {r["campaign"]: r for r in desc.collect()}
    assert "campaign_window_beyond_data" in d[2]["dq_flags"] and d[1]["duration_days"] == 51

    table, _ = campaigns.build_campaign_table(
        make_bronze([
            {"description": "TypeA", "household_key": "1", "campaign": "1"},
            {"description": "TypeB", "household_key": "2", "campaign": "1"},
        ]),
        desc, "r",
    )  # fmt: skip
    t = {r["household_key"]: r for r in table.collect()}
    assert t[1]["dq_flags"] == [] and "campaign_type_mismatch" in t[2]["dq_flags"]

    red, _ = campaigns.build_redemptions(
        make_bronze([
            {"household_key": "1", "day": "120", "coupon_upc": "555", "campaign": "1"},
            {"household_key": "1", "day": "120", "coupon_upc": "555", "campaign": "2"},
            {"household_key": "1", "day": "200", "coupon_upc": "556", "campaign": "1"},
        ]),
        desc, "r",
    )  # fmt: skip
    flags = {(r["campaign"], r["day"]): set(r["dq_flags"]) for r in red.collect()}
    assert flags[(1, 120)] == {"upc_multi_campaign_redemption"}
    assert flags[(2, 120)] == {"upc_multi_campaign_redemption", "redemption_outside_window"}
    assert flags[(1, 200)] == {"redemption_outside_window"}


def test_coupons_exact_duplicates_dropped(make_bronze):
    row = {"coupon_upc": "1000", "product_id": "10", "campaign": "1"}
    df, log = campaigns.build_coupons(make_bronze([row, row, {**row, "product_id": "11"}]), "r")
    assert df.count() == 2
    assert [r["rule_id"] for r in log.collect()] == ["exact_duplicate"]
