"""End-to-end: tiny synthetic CSVs in the raw file layout -> bronze -> silver -> gold (Parquet)."""

from pyspark.sql import functions as F

from retail_promo import pipeline
from retail_promo.config import Settings
from retail_promo.io import LocalParquetIO

RAW = {
    "transaction_data": [
        "household_key,BASKET_ID,DAY,PRODUCT_ID,QUANTITY,SALES_VALUE,STORE_ID,RETAIL_DISC,TRANS_TIME,WEEK_NO,COUPON_DISC,COUPON_MATCH_DISC",
        "1,100,60,10,1,2.00,1,0,1200,9,0,0",
        "1,101,70,10,2,3.00,1,-1.00,1300,11,0,0",
        "1,101,70,99,40000,90.00,1,0,1300,11,0,0",
        "2,200,70,11,0,0.50,1,0,1000,11,0,0",
        "2,201,77,10,1,5.551115E-17,1,0,1000,12,0,0",
    ],
    "causal_data": [
        "PRODUCT_ID,STORE_ID,WEEK_NO,display,mailer",
        "10,1,11,0,A",
        "10,1,11,7,A",
        "11,1,12,3,0",
    ],
    "product": [
        "PRODUCT_ID,MANUFACTURER,DEPARTMENT,BRAND,COMMODITY_DESC,SUB_COMMODITY_DESC,CURR_SIZE_OF_PRODUCT",
        "10,1,GROCERY,Private,SOUP,S,10 OZ",
        "11,1,PRODUCE,National,FRUIT,F, ",
        "99,1,KIOSK-GAS,Private,COUPON/MISC ITEMS,GASOLINE, ",
    ],
    "hh_demographic": [
        "classification_1,classification_2,classification_3,HOMEOWNER_DESC,classification_5,classification_4,KID_CATEGORY_DESC,household_key",
        "Age Group3,X,Level4,Renter,Group2,2,1,1",
    ],
    "campaign_desc": ["DESCRIPTION,CAMPAIGN,START_DAY,END_DAY", "TypeA,1,65,80"],
    "campaign_table": ["DESCRIPTION,household_key,CAMPAIGN", "TypeA,1,1", "TypeA,2,1"],
    "coupon": ["COUPON_UPC,PRODUCT_ID,CAMPAIGN", "1000,10,1", "1000,10,1"],
    "coupon_redempt": ["household_key,DAY,COUPON_UPC,CAMPAIGN", "1,70,1000,1"],
}  # fmt: skip


def test_pipeline_end_to_end(spark, tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    for stem, lines in RAW.items():
        (raw_dir / f"{stem}.csv").write_text("\n".join(lines) + "\n")
    io = LocalParquetIO(spark, tmp_path / "out", raw_dir)
    settings = Settings(panel_min_weeks=1)

    run_id = pipeline.run_all(spark, io, settings)

    checks = io.read("ops", "dq_check_results").filter(F.col("run_id") == run_id)
    failed = checks.filter("severity = 'error' AND NOT passed").collect()
    assert failed == [], failed
    assert checks.count() > 30

    log = io.read("ops", "dq_log").filter(F.col("run_id") == run_id)
    rules = {
        r["rule_id"]: r["n"] for r in log.groupBy("rule_id").agg(F.count("*").alias("n")).collect()
    }
    assert rules["exact_duplicate"] == 1
    assert rules["causal_conflicting_duplicate"] == 2
    assert rules["money_rounded"] == 1
    assert rules["non_merchandise"] == 1
    assert rules["blank_size_to_null"] == 2

    assert io.read("silver", "causal").count() == 2
    assert io.read("gold", "household_features").count() == 2
    assert io.read("gold", "campaign_household").count() == 2
    # product 11 was only bought with quantity 0 and product 99 is fuel, so only product 10 is in scope
    assert io.read("gold", "product_week_panel").count() == 93
    runs = {(r["run_id"], r["status"]) for r in io.read("ops", "pipeline_runs").collect()}
    assert runs == {(run_id, "started"), (run_id, "succeeded")}
    assert pipeline.latest_started_run(io) == run_id
