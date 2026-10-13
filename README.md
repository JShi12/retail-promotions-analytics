# Retail promotions analytics on Databricks

Which promotions lift sales, which customers respond to price and coupons, and who should receive the
next coupon campaign: an end-to-end analysis for a grocery retailer's pricing and promotions team,
built on Databricks (PySpark, Delta Lake, Unity Catalog) with dunnhumby's *The Complete Journey*
(2,500 households, two years, 2.6M purchase lines).

**Read the findings:** [REPORT.md](REPORT.md) (for a pricing and promotions manager).

## What we found

- **Mailer weeks sell about 84% more units** of the featured product (+84.4%, 95% CI +75.3% to +94.1%),
  not the five-fold lift a raw comparison suggests. Fresh meat and milk respond most (pork +745%,
  chicken +356%); snacks, bread and frozen pizza least (+19% to +26%); beer shows no detectable lift.
- **An in-store display goes with 2.8 to 4.6 times normal sales** in the stores that have it.
- **Three stable customer segments** by price-and-promotion behaviour. One in five households, the
  coupon users, redeem campaign coupons in 30.4% of campaigns against 6.3–6.4% for everyone else.
- **A model targets coupon campaigns better than the best simple rule**: sending each campaign to the
  top 20% of eligible households would have captured 58.5% of redemptions, against 51.4% using past
  coupon use and 20% at random. It also beat the rule on later campaigns it was not trained on.

Promotion results are associations, not proven causal effects, and the targeting model predicts
redemption, not extra sales. The recommended next step is a **randomised holdout** in future
campaigns to measure what coupons actually add (see [REPORT.md](REPORT.md)).

## Dashboard

A Databricks AI/BI dashboard on the gold tables, with five pages: overview, promotion levers,
customer segments, a targeting back-test and data quality. Every chart reads a tested SQL view, so it
shows the same numbers as the notebooks. Build guide: [docs/dashboard_guide.md](docs/dashboard_guide.md).

## How it is built

```
raw CSVs ──► bronze ──► silver ──► gold ──► analysis notebooks ──► dashboard views ──► dashboard
(volume)    as received  typed,     business   10 promotions         SQL, tested
                         checked,   tables     20 segments
                         flagged               30 coupon targeting
                └──────────── data-quality log and checks (ops) ────────────┘
```

| Stage | What it does |
|---|---|
| Data pipeline (notebooks `00`–`04`) | Medallion layers in Delta tables on Unity Catalog. Problem rows are **flagged, not deleted**; every changed, flagged or dropped row is logged with its reason, and error-level checks stop a run. Column dictionary and data findings: [data/README.md](data/README.md). |
| Promotion effectiveness (`10`) | Fixed-effects Poisson regression of weekly product units on mailer and display exposure, by category, with timing checks and seven sensitivity variants (pyfixest). |
| Customer segments (`20`) | Spark MLlib k-means on behaviour measured against category × store benchmarks; number of segments chosen by bootstrap stability and a shuffled-data baseline. |
| Coupon targeting (`30`) | Logistic regression and gradient-boosted trees (scikit-learn) with nested, household-grouped cross-validation, judged within each campaign against a simple rule, plus a time-forward test. |
| Dashboard (`40`) | SQL views in a `dashboard` schema; the dashboard itself is built from the guide. |
| Orchestration | A scheduled Databricks job runs everything in order ([jobs/retail_pipeline.yml](jobs/retail_pipeline.yml), [setup](docs/job_setup.md)). |
| Quality | Logic lives in `src/` and is unit-tested with pytest on local Spark; GitHub Actions runs lint and tests on every push. |

**Tools:** PySpark 4.2, Delta Lake, Unity Catalog, Databricks Free Edition (serverless), Spark MLlib
(clustering), scikit-learn (small-data models), pyfixest (fixed-effects regression), pytest, GitHub
Actions, `uv`, Python 3.12 (matching Databricks serverless environment version 6).

**Why both Spark and scikit-learn:** Spark handles the 36.8M-row promotion table and the pipeline;
the coupon model trains on 7,208 household-campaign rows, where single-machine scikit-learn is the
right tool (Spark MLlib's boosted trees ran for over an hour on serverless for the same work).

## Project status

| Phase | Deliverable | Status |
|---|---|---|
| 0 | Data download, inspection and column dictionary | done |
| 1 | Medallion pipeline with data-quality logging, tests, CI ([plan](docs/phase1_plan.md)) | done, run on Databricks |
| 2 | Promotion effectiveness ([plan](docs/phase2_plan.md)) | done, run on Databricks |
| 3 | Customer segmentation ([plan](docs/phase3_plan.md)) | done, run on Databricks |
| 4 | Coupon campaign response model ([plan](docs/phase4_plan.md)) | done, run on Databricks |
| 5 | Dashboard, scheduled job, README and report ([plan](docs/phase5_plan.md)) | views, job definition and report built; dashboard and job run pending |
| Stretch | Re-run on dunnhumby "Let's Get Sort-of-Real" (~300M rows) for scale | optional |

Every Databricks result so far matches the local run of the same code exactly.

## Repo layout

```
src/retail_promo/   pipeline (bronze/silver/gold, data-quality rules and checks), analysis and dashboard SQL
notebooks/          thin Databricks notebooks that call src/ (run from a Databricks Git folder)
tests/              pytest on local Spark with synthetic data (never real rows)
jobs/               Databricks job definition
docs/               phase plans, dashboard and job guides
data/README.md      column dictionary and data findings (raw data is gitignored)
reports/figures/    figures (aggregates only)
REPORT.md           findings for a pricing and promotions manager
```

Notebook numbering: `00`–`04` are the pipeline and run in order; analysis notebooks are numbered in
tens by phase (`10`, `20`, `30`, `40`), leaving room for follow-ups (e.g. `11_...`) without renumbering.

## Running it

### Locally

```bash
brew install openjdk@17            # local Spark needs a JDK
uv sync
JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run pytest
```

Download `dunnhumby_The-Complete-Journey.zip` from <https://www.dunnhumby.com/source-files/> and unzip
into `data/raw/complete_journey/`. The data is about 850 MB of CSV, small enough for local Spark, so
every phase also runs locally with the same `src/` code, writing Parquet instead of Delta. These runs
are the development loop and the source of the figures in `reports/figures/`.

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
uv run python -m retail_promo.run_local \
  --raw-dir "data/raw/complete_journey/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV" \
  --out-dir data/processed                                  # pipeline, ~3 min
uv run python -m retail_promo.analysis.run_phase2_local     # promotions, ~30 s
uv run python -m retail_promo.analysis.run_phase3_local     # segments, ~10 min
uv run python -m retail_promo.analysis.run_phase4_local     # coupon targeting, ~1 min
```

### On Databricks (Free Edition)

1. Add this repo as a **Git folder** (*Workspace → Create → Git folder*).
2. Run `notebooks/00_setup` once: it creates the `retail` catalog, the `raw`, `bronze`, `silver`,
   `gold` and `ops` (operations: logs and run records) schemas and the `raw.landing` volume.
3. Upload the 8 CSVs (about 848 MB) to *Catalog → raw → landing*.
4. Create the job from [docs/job_setup.md](docs/job_setup.md) and run it once, or run the notebooks
   in order: `01`–`04`, then `10`, `20`, `30`, then `40`.
5. Build the dashboard from [docs/dashboard_guide.md](docs/dashboard_guide.md).

Things that matter on Free Edition:

- Attach notebooks to plain **Serverless**, not **Git Folder Serverless**: the latter installs this
  repo's `pyproject.toml`, whose open-source `pyspark` replaces Databricks' own and stops the kernel.
- After pulling new commits, **detach and reattach** a notebook before running: a live session keeps
  the code it first imported. Notebooks print a `code fingerprint` and stop if the session is stale
  (compare with `uv run python -c "import retail_promo; print(retail_promo.CODE_FINGERPRINT)"`).
- Notebooks save figures to the Unity Catalog volume `ops/figures`, never into the Git folder.

## Data terms

dunnhumby's site terms allow research, personal and non-commercial use and prohibit redistributing the
data without permission, so this repo contains no raw or row-level data: tests use synthetic rows and
published outputs are aggregates. Details in [data/README.md](data/README.md).
