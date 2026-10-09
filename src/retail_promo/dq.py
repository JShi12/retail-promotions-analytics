"""Data-quality building blocks.

Two mechanisms:
- Row flags: each silver row carries a `dq_flags` array; every flagged, modified or dropped
  row is written to `ops.dq_log` with the rule that caught it.
- Table checks: pass/fail assertions on a whole table, written to `ops.dq_check_results`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql import types as T

# rule_id -> (action, description). `cast_failed:<column>` rules are generated.
RULES: dict[str, tuple[str, str]] = {
    "exact_duplicate": ("dropped", "Exact duplicate row; one copy kept"),
    "causal_conflicting_duplicate": (
        "modified",
        "Same product/store/week listed twice with different codes; collapsed by OR rule",
    ),
    "invalid_display_code": ("flagged", "display code not in the documented list"),
    "invalid_mailer_code": ("flagged", "mailer code not in the documented list"),
    "non_merchandise": ("flagged", "Fuel or non-merchandise department; quantity not in units"),
    "quantity_zero": ("flagged", "quantity = 0; price undefined"),
    "sales_zero": ("flagged", "sales_value = 0"),
    "quantity_zero_sales_positive": ("flagged", "quantity = 0 but sales_value > 0"),
    "retail_disc_positive": ("flagged", "retail_disc > 0; discounts should be <= 0"),
    "money_rounded": ("modified", "Floating-point noise (e.g. 5.55E-17) rounded to 2 decimals"),
    "week_day_mismatch": ("flagged", "week_no != floor((day + 1) / 7) + 1"),
    "blank_hierarchy": ("modified", "Blank department/commodity set to UNKNOWN"),
    "blank_size_to_null": ("modified", "Blank curr_size_of_product set to null"),
    "unexpected_category": ("flagged", "Demographic value outside the known list"),
    "campaign_window_beyond_data": ("flagged", "Campaign ends after the last transaction day"),
    "campaign_window_invalid": ("flagged", "start_day > end_day"),
    "campaign_type_mismatch": ("flagged", "description differs from campaign_desc"),
    "upc_multi_campaign_redemption": (
        "flagged",
        "Same household, day and coupon redeemed under more than one campaign",
    ),
    "redemption_outside_window": ("flagged", "Redemption day outside the campaign window"),
}

DQ_LOG_SCHEMA = T.StructType(
    [
        T.StructField("run_id", T.StringType(), False),
        T.StructField("table_name", T.StringType(), False),
        T.StructField("rule_id", T.StringType(), False),
        T.StructField("action", T.StringType(), False),
        T.StructField("record_key", T.StringType(), True),
        T.StructField("detail", T.StringType(), True),
        T.StructField("logged_at", T.TimestampType(), False),
    ]
)


def rule_action(rule_id: str) -> str:
    if rule_id.startswith("cast_failed:"):
        return "flagged"
    return RULES[rule_id][0]


def rule_detail(rule_id: str) -> str:
    if rule_id.startswith("cast_failed:"):
        return f"Value in {rule_id.split(':', 1)[1]} could not be converted; set to null"
    return RULES[rule_id][1]


# ---------------------------------------------------------------- row flags


def init_flags(df: DataFrame) -> DataFrame:
    return df.withColumn("dq_flags", F.array().cast("array<string>"))


def add_flag(df: DataFrame, condition: Column, rule_id: str) -> DataFrame:
    """Append `rule_id` to dq_flags where condition is true (null counts as false)."""
    if not rule_id.startswith("cast_failed:") and rule_id not in RULES:
        raise KeyError(f"Unknown DQ rule: {rule_id}")
    return df.withColumn(
        "dq_flags",
        F.when(
            F.coalesce(condition, F.lit(False)), F.array_union("dq_flags", F.array(F.lit(rule_id)))
        ).otherwise(F.col("dq_flags")),
    )


def has_flag(rule_id: str) -> Column:
    return F.array_contains("dq_flags", rule_id)


def _record_key(key_cols: list[str]) -> Column:
    return F.to_json(F.struct(*[F.col(c) for c in key_cols]))


def _detail_map() -> Column:
    pairs = []
    for rule_id in RULES:
        pairs += [F.lit(rule_id), F.lit(rule_detail(rule_id))]
    return F.create_map(*pairs)


def _action_map() -> Column:
    pairs = []
    for rule_id, (action, _) in RULES.items():
        pairs += [F.lit(rule_id), F.lit(action)]
    return F.create_map(*pairs)


def flags_to_log(df: DataFrame, table_name: str, key_cols: list[str], run_id: str) -> DataFrame:
    """One dq_log row per (row, flag) for every flagged row."""
    exploded = df.select(
        _record_key(key_cols).alias("record_key"), F.explode("dq_flags").alias("rule_id")
    )
    is_cast = F.col("rule_id").startswith("cast_failed:")
    return exploded.select(
        F.lit(run_id).alias("run_id"),
        F.lit(table_name).alias("table_name"),
        "rule_id",
        F.when(is_cast, F.lit("flagged"))
        .otherwise(_action_map()[F.col("rule_id")])
        .alias("action"),
        "record_key",
        F.when(
            is_cast,
            F.concat(
                F.lit("Value in "),
                F.substring_index("rule_id", ":", -1),
                F.lit(" could not be converted; set to null"),
            ),
        )
        .otherwise(_detail_map()[F.col("rule_id")])
        .alias("detail"),
        F.current_timestamp().alias("logged_at"),
    )


def log_rows(
    df: DataFrame,
    table_name: str,
    rule_id: str,
    key_cols: list[str],
    run_id: str,
    detail: Column | None = None,
) -> DataFrame:
    """dq_log rows for every row of `df` under one rule (used for dropped/collapsed rows)."""
    return df.select(
        F.lit(run_id).alias("run_id"),
        F.lit(table_name).alias("table_name"),
        F.lit(rule_id).alias("rule_id"),
        F.lit(rule_action(rule_id)).alias("action"),
        _record_key(key_cols).alias("record_key"),
        (detail if detail is not None else F.lit(rule_detail(rule_id))).alias("detail"),
        F.current_timestamp().alias("logged_at"),
    )


def empty_log(spark: SparkSession) -> DataFrame:
    return spark.createDataFrame([], DQ_LOG_SCHEMA)


def union_logs(spark: SparkSession, logs: list[DataFrame]) -> DataFrame:
    out = empty_log(spark)
    for log in logs:
        out = out.unionByName(log)
    return out


def drop_exact_duplicates(
    df: DataFrame, table_name: str, run_id: str, cols: list[str] | None = None
) -> tuple[DataFrame, DataFrame]:
    """Keep one copy of each exact duplicate; log one dq_log row per dropped copy."""
    cols = cols or df.columns
    counted = df.groupBy(*cols).agg(F.count(F.lit(1)).alias("_n"))
    dropped = counted.filter("_n > 1").withColumn(
        "_copy", F.explode(F.sequence(F.lit(2), F.col("_n")))
    )
    log = log_rows(
        dropped,
        table_name,
        "exact_duplicate",
        cols,
        run_id,
        detail=F.concat(
            F.lit("Copy "), F.col("_copy"), F.lit(" of "), F.col("_n"), F.lit(" dropped")
        ),
    )
    return counted.drop("_n"), log


# ---------------------------------------------------------------- table checks


@dataclass(frozen=True)
class CheckResult:
    layer: str
    table_name: str
    check_id: str
    severity: str  # "error" stops the run, "warning" is recorded
    passed: bool
    observed: str
    expected: str
    detail: str = ""


CHECK_SCHEMA = T.StructType(
    [
        T.StructField("run_id", T.StringType(), False),
        T.StructField("layer", T.StringType(), False),
        T.StructField("table_name", T.StringType(), False),
        T.StructField("check_id", T.StringType(), False),
        T.StructField("severity", T.StringType(), False),
        T.StructField("passed", T.BooleanType(), False),
        T.StructField("observed", T.StringType(), True),
        T.StructField("expected", T.StringType(), True),
        T.StructField("detail", T.StringType(), True),
        T.StructField("checked_at", T.TimestampType(), False),
    ]
)


def check_equal(
    layer: str,
    table: str,
    check_id: str,
    observed: int,
    expected: int,
    severity: str = "error",
    detail: str = "",
) -> CheckResult:
    return CheckResult(
        layer, table, check_id, severity, observed == expected, str(observed), str(expected), detail
    )


def check_unique(
    df: DataFrame, cols: list[str], layer: str, table: str, severity: str = "error"
) -> CheckResult:
    dup_keys = df.groupBy(*cols).count().filter("count > 1").count()
    return CheckResult(
        layer,
        table,
        f"unique({','.join(cols)})",
        severity,
        dup_keys == 0,
        f"{dup_keys} duplicated keys",
        "0",
    )


def check_no_violations(
    df: DataFrame, condition: Column, check_id: str, layer: str, table: str, severity: str = "error"
) -> CheckResult:
    """`condition` describes a violating row."""
    bad = df.filter(F.coalesce(condition, F.lit(False))).count()
    return CheckResult(layer, table, check_id, severity, bad == 0, f"{bad} rows", "0")


def check_fk(
    child: DataFrame, parent: DataFrame, cols: list[str], layer: str, table: str, parent_name: str
) -> CheckResult:
    orphans = (
        child.select(*cols)
        .distinct()
        .join(parent.select(*cols).distinct(), cols, "left_anti")
        .count()
    )
    return CheckResult(
        layer,
        table,
        f"fk({','.join(cols)})->{parent_name}",
        "error",
        orphans == 0,
        f"{orphans} orphan keys",
        "0",
    )


def check_in_set(
    df: DataFrame, col: str, allowed: frozenset[str], layer: str, table: str
) -> CheckResult:
    bad = df.filter(~F.col(col).isin(*sorted(allowed))).select(col).distinct().collect()
    values = sorted(str(r[0]) for r in bad)
    return CheckResult(
        layer,
        table,
        f"in_set({col})",
        "error",
        not values,
        ",".join(values) or "none",
        "documented codes",
    )


def checks_to_df(spark: SparkSession, results: list[CheckResult], run_id: str) -> DataFrame:
    rows = [{"run_id": run_id, **asdict(r)} for r in results]
    without_ts = T.StructType([f for f in CHECK_SCHEMA.fields if f.name != "checked_at"])
    return spark.createDataFrame(rows, without_ts).withColumn("checked_at", F.current_timestamp())


class DataQualityError(RuntimeError):
    pass


def raise_on_errors(results: list[CheckResult]) -> None:
    failed = [r for r in results if r.severity == "error" and not r.passed]
    if failed:
        lines = [
            f"{r.layer}.{r.table_name} {r.check_id}: observed {r.observed}, expected {r.expected}"
            for r in failed
        ]
        raise DataQualityError("Error-level DQ checks failed:\n" + "\n".join(lines))
