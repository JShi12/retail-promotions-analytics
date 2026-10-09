# Retail promotions analytics on Databricks

Promotion effectiveness, customer segmentation and coupon-response targeting for a grocery
retailer, built as a Databricks medallion pipeline (Delta + Unity Catalog) in PySpark on
dunnhumby's *The Complete Journey* (2,500 households, ~2 years, 2.6M transaction lines).

> Status: Phase 0 complete. Phase 1 pipeline built and verified locally (all 49 DQ checks pass on the
> full data); the Databricks run is pending. Analysis results will be added as phases land.

## Questions

1. Which promotion levers (display, mailer, discount depth) are associated with higher unit sales, by category?
2. Which household segments exist, and how do they differ in basket, promo sensitivity and demographics?
3. Which households should receive a coupon campaign, and how much better is a model than a simple rule?

## Plan

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Setup, data download and inspection, column dictionary ([data/README.md](data/README.md)) | done |
| 1 | Medallion pipeline: bronze → silver (typed, deduplicated, flag-don't-drop, `dq_log`) → gold; DQ checks; pytest; CI | local run passed; Databricks run pending |
| 2 | Promotion effectiveness: log weekly units on display / mailer / discount depth with product and week effects; % lift with CIs by category (associational) | planned |
| 3 | Segmentation: Spark MLlib k-means on household features; k by silhouette and seed stability | planned |
| 4 | Campaign response classifier: pre-campaign features, household-grouped folds, baselines, tuned GBT, importance, targeting curve | planned |
| 5 | Dashboard on gold tables, scheduled Databricks job, commercial README, REPORT.md | planned |
| Stretch | Re-run on dunnhumby "Let's Get Sort-of-Real" (~300M rows) for scale | optional |

## Stack

PySpark 4.2 / Delta Lake on Databricks Free Edition (serverless), Unity Catalog, MLlib, pytest,
GitHub Actions. Local development with `uv` and Python 3.12, matching Databricks serverless
environment version 6.

## Repo layout

```
src/retail_promo/   PySpark transforms (bronze/silver/gold, dq, features), unit-tested
notebooks/          Thin Databricks notebooks that call src/ (via a Databricks Git folder)
tests/              pytest on local Spark with synthetic data
data/README.md      Column dictionary and data findings (raw data is gitignored)
docs/               Phase plans (docs/phase1_plan.md)
```

## Local setup

```bash
brew install openjdk@17            # local Spark needs a JDK
uv sync
JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run pytest
```

Raw data: download `dunnhumby_The-Complete-Journey.zip` from
<https://www.dunnhumby.com/source-files/> and unzip into `data/raw/complete_journey/`.
Do not commit or redistribute it; see the terms note in [data/README.md](data/README.md).

Local dry run of the whole pipeline (Parquet instead of Delta, same code as Databricks):

```bash
JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run python -m retail_promo.run_local \
  --raw-dir "data/raw/complete_journey/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV" \
  --out-dir data/processed
```

## Running on Databricks (Free Edition)

1. Push this repo to GitHub, then in Databricks: *Workspace → Create → Git folder* with the repo URL.
2. Run `notebooks/00_setup` once. It creates the catalog (`retail`, or set the widget to `workspace`),
   the `raw/bronze/silver/gold/ops` schemas and the `raw.landing` volume.
3. Upload the 8 CSVs to the volume (*Catalog → raw → landing → Upload to this volume*), about 848 MB.
4. Re-run the last cell of `00_setup` to confirm all files are present, then run `01_bronze`,
   `02_silver`, `03_gold` and `04_dq_report` in order. `01_bronze` starts a run; the next two pick
   up its `run_id` automatically (or pass one in the `run_id` widget, as a job does).

## Pipeline (Phase 1)

| Layer | What it holds | Key rules |
|---|---|---|
| bronze | 8 raw tables, all strings, `_source_file` / `_ingested_at` / `_run_id` | row counts must equal raw file lines |
| silver | typed tables with a `dq_flags` array per row | `try_cast` (bad values flagged), exact duplicates dropped, everything else flagged; `causal_data` conflicts collapsed by OR rule |
| gold | `household_features`, `product_week_panel`, `campaign_household`, `dq_summary` | campaign features use only data before each campaign start |
| ops | `dq_log` (one row per flagged / modified / dropped record), `dq_check_results`, `pipeline_runs` | appended per `run_id` |

Rules and checks are listed in [`src/retail_promo/dq.py`](src/retail_promo/dq.py) and
[`src/retail_promo/pipeline.py`](src/retail_promo/pipeline.py).
