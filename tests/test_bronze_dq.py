import pytest
from pyspark.sql import functions as F

from retail_promo import dq
from retail_promo.bronze import normalize_column_name, to_bronze


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("PRODUCT_ID", "product_id"),
        ("household_key", "household_key"),
        (" Curr Size ", "curr_size"),
    ],
)
def test_normalize_column_name(raw, expected):
    assert normalize_column_name(raw) == expected


def test_to_bronze_keeps_strings_and_adds_metadata(spark):
    raw = spark.createDataFrame([("1", "2.50")], "PRODUCT_ID string, SALES_VALUE string")
    out = to_bronze(raw, "x.csv", "run1")
    assert out.columns == ["product_id", "sales_value", "_source_file", "_ingested_at", "_run_id"]
    assert dict(out.dtypes)["sales_value"] == "string"
    assert out.first()["_run_id"] == "run1"


def test_add_flag_appends_once_and_treats_null_as_false(spark):
    df = dq.init_flags(spark.createDataFrame([(1,), (0,), (None,)], "q int"))
    df = dq.add_flag(df, F.col("q") == 0, "quantity_zero")
    df = dq.add_flag(df, F.col("q") == 0, "quantity_zero")
    flags = {r["q"]: r["dq_flags"] for r in df.collect()}
    assert flags == {1: [], 0: ["quantity_zero"], None: []}


def test_add_flag_rejects_unknown_rule(spark):
    df = dq.init_flags(spark.createDataFrame([(1,)], "q int"))
    with pytest.raises(KeyError):
        dq.add_flag(df, F.lit(True), "not_a_rule")


def test_flags_to_log_one_row_per_flag(spark):
    df = dq.init_flags(spark.createDataFrame([(1, 0)], "k int, q int"))
    df = dq.add_flag(df, F.lit(True), "quantity_zero")
    df = dq.add_flag(df, F.lit(True), "cast_failed:q")
    log = dq.flags_to_log(df, "t", ["k"], "run1").collect()
    by_rule = {r["rule_id"]: r for r in log}
    assert set(by_rule) == {"quantity_zero", "cast_failed:q"}
    assert by_rule["quantity_zero"]["action"] == "flagged"
    assert by_rule["cast_failed:q"]["detail"].startswith("Value in q")
    assert by_rule["quantity_zero"]["record_key"] == '{"k":1}'


def test_drop_exact_duplicates_logs_each_dropped_copy(spark):
    df = spark.createDataFrame(
        [("a", "1"), ("a", "1"), ("a", "1"), ("b", "2")], "x string, y string"
    )
    kept, log = dq.drop_exact_duplicates(df, "t", "run1")
    assert kept.count() == 2
    assert log.count() == 2  # three copies of ("a","1") -> two dropped
    assert {r["action"] for r in log.collect()} == {"dropped"}


def test_checks(spark):
    df = spark.createDataFrame([(1, "A"), (1, "B"), (2, "Q")], "k int, code string")
    parent = spark.createDataFrame([(1,)], "k int")
    assert not dq.check_unique(df, ["k"], "silver", "t").passed
    assert dq.check_unique(df, ["k", "code"], "silver", "t").passed
    fk = dq.check_fk(df, parent, ["k"], "silver", "t", "p")
    assert not fk.passed and fk.observed == "1 orphan keys"
    in_set = dq.check_in_set(df, "code", frozenset({"A", "B"}), "silver", "t")
    assert not in_set.passed and in_set.observed == "Q"
    with pytest.raises(dq.DataQualityError):
        dq.raise_on_errors([fk])
    dq.raise_on_errors(
        [dq.check_equal("s", "t", "c", 1, 2, severity="warning")]
    )  # warnings don't raise
