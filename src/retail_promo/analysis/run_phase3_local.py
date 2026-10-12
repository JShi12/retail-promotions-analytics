"""Run Phase 3 locally on the Parquet output of the local pipeline run; save figures and print tables.

JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run python -m retail_promo.analysis.run_phase3_local
"""

from __future__ import annotations

import argparse

import pandas as pd

from retail_promo.analysis import phase3, plots
from retail_promo.io import LocalParquetIO
from retail_promo.run_local import local_spark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="reports/figures")
    args = parser.parse_args()

    spark = local_spark()
    spark.sparkContext.setLogLevel("ERROR")
    io = LocalParquetIO(spark, args.data_dir, "unused")
    res = phase3.run(spark, io)
    phase3.write_gold(spark, io, res, run_id="local")

    figures = {
        "phase3_k_selection": plots.k_selection(res.k_summary, res.k),
        "phase3_centroids": plots.segment_centroids(res.centroids),
        "phase3_actual_vs_expected": plots.actual_vs_expected(res.profiles),
        "phase3_size_redemption": plots.size_and_redemption(res.profiles),
    }
    for name, fig in figures.items():
        print("saved", plots.save(fig, args.figures_dir, name))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("k =", res.k)
    for name in ["k_summary", "centroids", "demographics", "representativeness", "coupon_check"]:
        print(f"\n{name}\n", getattr(res, name).round(4).to_string())
    print(
        "\nk_segments (chosen k)\n",
        res.k_segments[res.k_segments["k"] == res.k].round(3).to_string(),
    )
    print("\nprofiles\n", res.profiles.round(4).T.to_string())
    print(
        "\nlow activity by segment:",
        res.assigned[res.assigned["low_activity"]]["segment"].value_counts().to_dict(),
    )


if __name__ == "__main__":
    main()
