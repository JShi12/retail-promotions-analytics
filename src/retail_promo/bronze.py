"""Bronze: raw CSVs loaded as-is (all strings) plus load metadata."""

from __future__ import annotations

import re

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def normalize_column_name(name: str) -> str:
    """'PRODUCT_ID' -> 'product_id', 'Curr Size' -> 'curr_size'."""
    return re.sub(r"[^0-9a-z]+", "_", name.strip().lower()).strip("_")


def read_raw_csv(spark: SparkSession, path: str) -> DataFrame:
    """Every column as a string, so nothing is converted silently."""
    return (
        spark.read.option("header", True)
        .option("inferSchema", False)
        .option("mode", "PERMISSIVE")
        .csv(path)
    )


def count_raw_records(spark: SparkSession, path: str) -> int:
    """Physical data lines in the file (header excluded), for reconciliation."""
    return spark.read.text(path).filter(F.length("value") > 0).count() - 1


def to_bronze(df: DataFrame, source_file: str, run_id: str) -> DataFrame:
    renamed = df.toDF(*[normalize_column_name(c) for c in df.columns])
    return renamed.select(
        *[F.col(c).cast("string") for c in renamed.columns],
        F.lit(source_file).alias("_source_file"),
        F.current_timestamp().alias("_ingested_at"),
        F.lit(run_id).alias("_run_id"),
    )
