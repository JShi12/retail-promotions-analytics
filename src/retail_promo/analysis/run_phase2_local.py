"""Run Phase 2 locally on the Parquet output of the local pipeline run; save figures and print tables.

JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run python -m retail_promo.analysis.run_phase2_local
"""

from __future__ import annotations

import argparse

import pandas as pd

from retail_promo.analysis import phase2, plots
from retail_promo.config import Settings
from retail_promo.io import LocalParquetIO
from retail_promo.run_local import local_spark

FIGURES = {
    "phase2_raw_vs_model": plots.raw_vs_model,
    "phase2_mailer_by_category": plots.mailer_by_category,
    "phase2_display_by_category": plots.display_by_category,
    "phase2_product_vs_category": plots.product_vs_category,
    "phase2_timing": plots.timing,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    spark = local_spark()
    spark.sparkContext.setLogLevel("ERROR")
    io = LocalParquetIO(spark, args.data_dir, "unused")
    res = phase2.run(spark, io, Settings())
    phase2.write_gold(spark, io, res, run_id="local")

    for name, make in FIGURES.items():
        fig = (
            make(res.raw_contrast, res.results)
            if name == "phase2_raw_vs_model"
            else make(res.results)
        )
        print("saved", plots.save(fig, args.figures_dir, name))

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(res.categories.round(4).to_string())
    print(res.raw_contrast.round(3).to_string())
    print(res.results.drop(columns=["lift_definition"]).round(3).to_string())
    print(
        res.sensitivity[["variant", "term", "lift", "lift_low", "lift_high", "n_obs", "n_products"]]
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
