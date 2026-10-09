"""Bronze -> silver -> gold, with checks after each layer.

Shared by the Databricks notebooks (UnityCatalogIO) and the local dry run (LocalParquetIO).
Data tables are overwritten each run; ops tables (dq_log, dq_check_results, pipeline_runs) are
appended and keyed by run_id.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.bronze import count_raw_records, read_raw_csv, to_bronze
from retail_promo.config import DISPLAY_CODES, MAILER_CODES, RAW_FILES, Settings
from retail_promo.gold import tables as gold
from retail_promo.io import TableIO
from retail_promo.silver import campaigns, causal, demographics, products, transactions


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:6]


def _write_checks(
    spark: SparkSession, io: TableIO, results: list[dq.CheckResult], run_id: str
) -> None:
    if results:
        io.write(dq.checks_to_df(spark, results, run_id), "ops", "dq_check_results", mode="append")


# ---------------------------------------------------------------- bronze


def run_bronze(spark: SparkSession, io: TableIO, run_id: str) -> list[dq.CheckResult]:
    results = []
    for stem, table in RAW_FILES.items():
        path = io.raw_path(stem)
        io.write(to_bronze(read_raw_csv(spark, path), f"{stem}.csv", run_id), "bronze", table)
        loaded = io.read("bronze", table).count()
        results.append(
            dq.check_equal(
                "bronze", table, "rows_equal_raw_lines", loaded, count_raw_records(spark, path)
            )
        )
    _write_checks(spark, io, results, run_id)
    dq.raise_on_errors(results)
    return results


# ---------------------------------------------------------------- silver


def run_silver(
    spark: SparkSession, io: TableIO, run_id: str, settings: Settings
) -> list[dq.CheckResult]:
    b = {t: io.read("bronze", t) for t in RAW_FILES.values()}
    logs: list[DataFrame] = []

    def save(name: str, built: tuple[DataFrame, DataFrame]) -> DataFrame:
        df, log = built
        io.write(df, "silver", name)
        logs.append(log)
        return io.read("silver", name)

    prod = save("products", products.build(b["products"], run_id))
    caus = save("causal", causal.build(b["causal"], run_id))
    txn = save(
        "transactions",
        transactions.build(
            b["transactions"],
            prod,
            caus,
            run_id,
            settings.panel_first_week,
            settings.panel_last_week,
        ),
    )
    demo = save("demographics", demographics.build(b["demographics"], run_id))
    cdesc = save("campaign_desc", campaigns.build_campaign_desc(b["campaign_desc"], run_id))
    ctab = save(
        "campaign_table", campaigns.build_campaign_table(b["campaign_table"], cdesc, run_id)
    )
    coup = save("coupons", campaigns.build_coupons(b["coupons"], run_id))
    red = save(
        "coupon_redemptions", campaigns.build_redemptions(b["coupon_redemptions"], cdesc, run_id)
    )

    io.write(dq.union_logs(spark, logs), "ops", "dq_log", mode="append")
    log = io.read("ops", "dq_log").filter(F.col("run_id") == run_id)

    results = silver_checks(b, log, prod, caus, txn, demo, cdesc, ctab, coup, red)
    _write_checks(spark, io, results, run_id)
    dq.raise_on_errors(results)
    return results


def silver_checks(b, log, prod, caus, txn, demo, cdesc, ctab, coup, red) -> list[dq.CheckResult]:
    L = "silver"
    dropped = {
        r["table_name"]: r["n"]
        for r in log.filter(F.col("action") == "dropped")
        .groupBy("table_name")
        .agg(F.count("*").alias("n"))
        .collect()
    }
    results = []
    # Reconciliation: silver rows + logged drops = bronze rows. The causal collapse merges rows,
    # so it reconciles on collapsed rows: bronze rows - (logged collapsed rows - collapsed keys).
    for name, df in [("products", prod), ("transactions", txn), ("demographics", demo),
                     ("campaign_desc", cdesc), ("campaign_table", ctab), ("coupons", coup),
                     ("coupon_redemptions", red)]:  # fmt: skip
        results.append(
            dq.check_equal(
                L,
                name,
                "rows_plus_drops_equal_bronze",
                df.count() + dropped.get(name, 0),
                b[name].count(),
            )
        )
    collapsed_rows = log.filter(F.col("rule_id") == "causal_conflicting_duplicate").count()
    collapsed_keys = caus.filter(dq.has_flag("causal_conflicting_duplicate")).count()
    results.append(
        dq.check_equal(
            L,
            "causal",
            "rows_plus_merged_equal_bronze",
            caus.count() + collapsed_rows - collapsed_keys,
            b["causal"].count(),
        )
    )

    results += [
        dq.check_unique(prod, ["product_id"], L, "products"),
        dq.check_unique(demo, ["household_key"], L, "demographics"),
        dq.check_unique(caus, ["product_id", "store_id", "week_no"], L, "causal"),
        dq.check_unique(txn, ["basket_id", "product_id"], L, "transactions"),
        dq.check_unique(cdesc, ["campaign"], L, "campaign_desc"),
        dq.check_unique(ctab, ["household_key", "campaign"], L, "campaign_table"),
        dq.check_fk(txn, prod, ["product_id"], L, "transactions", "products"),
        dq.check_fk(caus, prod, ["product_id"], L, "causal", "products"),
        dq.check_fk(coup, prod, ["product_id"], L, "coupons", "products"),
        dq.check_fk(demo, txn, ["household_key"], L, "demographics", "transactions"),
        dq.check_fk(ctab, txn, ["household_key"], L, "campaign_table", "transactions"),
        dq.check_fk(ctab, cdesc, ["campaign"], L, "campaign_table", "campaign_desc"),
        dq.check_fk(
            red, ctab, ["household_key", "campaign"], L, "coupon_redemptions", "campaign_table"
        ),
        dq.check_fk(red, coup, ["coupon_upc", "campaign"], L, "coupon_redemptions", "coupons"),
        dq.check_in_set(caus, "display_code", DISPLAY_CODES, L, "causal"),
        dq.check_in_set(caus, "mailer_code", MAILER_CODES, L, "causal"),
        dq.check_no_violations(
            txn, dq.has_flag("week_day_mismatch"), "week_formula", L, "transactions"
        ),
        dq.check_no_violations(
            red,
            dq.has_flag("redemption_outside_window"),
            "redemption_in_window",
            L,
            "coupon_redemptions",
            "warning",
        ),
    ]
    for name, df in [("products", prod), ("causal", caus), ("transactions", txn), ("demographics", demo),
                     ("campaign_desc", cdesc), ("campaign_table", ctab), ("coupons", coup),
                     ("coupon_redemptions", red)]:  # fmt: skip
        cast_failed = F.exists("dq_flags", lambda f: f.startswith("cast_failed:"))
        results.append(dq.check_no_violations(df, cast_failed, "no_cast_failures", L, name))
    return results


# ---------------------------------------------------------------- gold


def run_gold(
    spark: SparkSession, io: TableIO, run_id: str, settings: Settings
) -> list[dq.CheckResult]:
    s = {t: io.read("silver", t) for t in RAW_FILES.values()}
    io.write(
        gold.household_features(s["transactions"], s["products"], s["demographics"]),
        "gold",
        "household_features",
    )
    io.write(
        gold.product_week_panel(
            s["transactions"],
            s["causal"],
            s["products"],
            settings.panel_first_week,
            settings.panel_last_week,
            settings.panel_min_weeks,
        ),
        "gold",
        "product_week_panel",
    )
    io.write(
        gold.campaign_household(
            s["transactions"],
            s["products"],
            s["campaign_table"],
            s["campaign_desc"],
            s["coupons"],
            s["coupon_redemptions"],
        ),
        "gold",
        "campaign_household",
    )
    io.write(gold.dq_summary(io.read("ops", "dq_log")), "gold", "dq_summary")

    hf = io.read("gold", "household_features")
    panel = io.read("gold", "product_week_panel")
    ch = io.read("gold", "campaign_household")
    L = "gold"
    results = [
        dq.check_unique(hf, ["household_key"], L, "household_features"),
        dq.check_equal(
            L,
            "household_features",
            "one_row_per_household",
            hf.count(),
            s["transactions"].select("household_key").distinct().count(),
        ),
        dq.check_unique(panel, ["product_id", "week_no"], L, "product_week_panel"),
        dq.check_no_violations(
            panel,
            (F.col("display_share") > 1.0 + 1e-9) | (F.col("mailer_share") > 1.0 + 1e-9),
            "promo_share_at_most_1",
            L,
            "product_week_panel",
        ),
        dq.check_unique(ch, ["household_key", "campaign"], L, "campaign_household"),
        dq.check_equal(
            L, "campaign_household", "one_row_per_pair", ch.count(), s["campaign_table"].count()
        ),
        dq.check_no_violations(
            ch,
            F.col("last_day") >= F.col("start_day"),
            "features_before_campaign_start",
            L,
            "campaign_household",
        ),
    ]
    _write_checks(spark, io, results, run_id)
    dq.raise_on_errors(results)
    return results


def record_run(
    spark: SparkSession, io: TableIO, run_id: str, status: str, detail: str = ""
) -> None:
    row = [(run_id, status, detail)]
    df = spark.createDataFrame(row, "run_id string, status string, detail string").withColumn(
        "finished_at", F.current_timestamp()
    )
    io.write(df, "ops", "pipeline_runs", mode="append")


def latest_started_run(io: TableIO) -> str:
    """run_id of the most recent bronze load, so notebooks run one by one share one id."""
    row = (
        io.read("ops", "pipeline_runs")
        .filter(F.col("status") == "started")
        .orderBy(F.col("finished_at").desc())
        .first()
    )
    if row is None:
        raise RuntimeError("No started run found; run 01_bronze first or pass a run_id.")
    return row["run_id"]


def run_all(spark: SparkSession, io: TableIO, settings: Settings, run_id: str | None = None) -> str:
    run_id = run_id or new_run_id()
    record_run(spark, io, run_id, "started")
    try:
        run_bronze(spark, io, run_id)
        run_silver(spark, io, run_id, settings)
        run_gold(spark, io, run_id, settings)
    except Exception as exc:
        record_run(spark, io, run_id, "failed", str(exc)[:2000])
        raise
    record_run(spark, io, run_id, "succeeded")
    return run_id
