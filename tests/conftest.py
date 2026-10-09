import os
from pathlib import Path

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Local convenience: Homebrew's keg-only openjdk@17 is not on the default path. CI sets JAVA_HOME.
_BREW_JDK = Path("/opt/homebrew/opt/openjdk@17")
if "JAVA_HOME" not in os.environ and _BREW_JDK.exists():
    os.environ["JAVA_HOME"] = str(_BREW_JDK)


@pytest.fixture(scope="session")
def spark():
    session = (
        SparkSession.builder.master("local[2]")
        .appName("retail-promo-tests")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


def bronze_df(spark, rows: list[dict]):
    """Bronze-shaped DataFrame: all strings plus load metadata."""
    cols = list(rows[0])
    data = [tuple(None if r[c] is None else str(r[c]) for c in cols) for r in rows]
    schema = ", ".join(f"`{c}` string" for c in cols)
    return (
        spark.createDataFrame(data, schema)
        .withColumn("_source_file", F.lit("test.csv"))
        .withColumn("_ingested_at", F.current_timestamp())
        .withColumn("_run_id", F.lit("test"))
    )


@pytest.fixture
def make_bronze(spark):
    return lambda rows: bronze_df(spark, rows)
