"""Run the whole pipeline locally on the raw CSVs, writing Parquet instead of Delta.

    JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run python -m retail_promo.run_local \
        --raw-dir "data/raw/complete_journey/.../dunnhumby_The-Complete-Journey CSV" \
        --out-dir data/processed
"""

from __future__ import annotations

import argparse
import time

from pyspark.sql import SparkSession

from retail_promo import pipeline
from retail_promo.config import Settings
from retail_promo.io import LocalParquetIO


def local_spark(app: str = "retail-promo", memory: str = "6g") -> SparkSession:
    return (
        SparkSession.builder.master("local[*]")
        .appName(app)
        .config("spark.driver.memory", memory)
        .config("spark.sql.shuffle.partitions", "32")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", required=True)
    parser.add_argument("--out-dir", default="data/processed")
    parser.add_argument("--min-weeks", type=int, default=Settings.panel_min_weeks)
    args = parser.parse_args()

    spark = local_spark()
    spark.sparkContext.setLogLevel("ERROR")
    io = LocalParquetIO(spark, args.out_dir, args.raw_dir)
    settings = Settings(panel_min_weeks=args.min_weeks)

    run_id = pipeline.new_run_id()
    pipeline.record_run(spark, io, run_id, "started")
    started = time.time()
    for layer, step in [
        ("bronze", lambda: pipeline.run_bronze(spark, io, run_id)),
        ("silver", lambda: pipeline.run_silver(spark, io, run_id, settings)),
        ("gold", lambda: pipeline.run_gold(spark, io, run_id, settings)),
    ]:
        t0 = time.time()
        results = step()
        failed = [r for r in results if not r.passed]
        print(f"{layer}: {len(results)} checks, {len(failed)} not passed, {time.time() - t0:.0f}s")
        for r in failed:
            print(
                f"  [{r.severity}] {r.table_name} {r.check_id}: observed {r.observed}, expected {r.expected}"
            )
    pipeline.record_run(spark, io, run_id, "succeeded")
    print(f"run {run_id} finished in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
