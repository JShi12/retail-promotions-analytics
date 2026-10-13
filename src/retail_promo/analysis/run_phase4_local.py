"""Run Phase 4 locally on the Parquet output of the local pipeline run; save figures and print tables.

JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run python -m retail_promo.analysis.run_phase4_local
"""

from __future__ import annotations

import argparse
import pickle

import pandas as pd

from retail_promo.analysis import plots, response
from retail_promo.io import LocalParquetIO
from retail_promo.run_local import local_spark


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--pickle", default=None, help="optionally save the results object here")
    args = parser.parse_args()

    spark = local_spark()
    spark.sparkContext.setLogLevel("ERROR")
    io = LocalParquetIO(spark, args.data_dir, "unused")
    res = response.run(io.read("gold", "campaign_household").toPandas())
    response.write_gold(spark, io, res, run_id="local")
    if args.pickle:
        with open(args.pickle, "wb") as fh:
            pickle.dump(res, fh)

    for name, fig in plots.phase4_figures(res).items():
        print("saved", plots.save(fig, args.figures_dir, name))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    print("best model:", res.best_model)
    for name in ["summary", "paired", "time_forward", "drift", "capture_curve"]:
        print(f"\n{name}\n", getattr(res, name).round(4).to_string())
    for m, t in res.chosen.items():
        print(f"\nchosen parameters: {m}\n", t.round(4).to_string())
    g = res.grid.groupby(
        ["model", *[c for c in res.grid.columns if c not in ("model", "fold", "inner_auc_within")]],
        dropna=False,
    )
    print(
        "\ngrid (mean inner AUC over folds)\n",
        g["inner_auc_within"].agg(["mean", "std"]).round(4).to_string(),
    )
    imp = res.importance.groupby(["model", "group"])["auc_drop"].agg(["mean", "std"]).round(4)
    print("\nimportance\n", imp.to_string())
    path = res.path.pivot(index="feature", columns="C", values="coef").round(3)
    print("\nL1 path (nonzero at any penalty)\n", path[(path != 0).any(axis=1)].to_string())


if __name__ == "__main__":
    main()
