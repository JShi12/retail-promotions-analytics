# Phase 5 plan: dashboard, scheduled job, README and REPORT

2026-10-13 · approved plan

Phase 5 turns Phases 1–4 into something a pricing and promotions team would use and read: a Databricks AI/BI dashboard on the gold tables, a scheduled job that refreshes everything, a README for a commercial reader, and a `REPORT.md` whose every number comes from the notebooks.

## 1. Dashboard-ready data

- New gold table **`gold.weekly_kpis`**: one row per week with sales, units, baskets, active households, **sales per active household**, average basket spend and the promotion share of sales. Weeks 1–13 are flagged `ramp_up` (active households grow from 88 to 972 as the panel fills, so raw totals rise for reasons unrelated to demand). Promotion share is only defined where promotion status is recorded (115 stores, weeks 9–101) and left empty elsewhere, not 0. Built in `03_gold`, with a test and a unique-week check.
- `gold.household_segments` also stores each household's actual and expected promotion share, discount depth and own-label share, so the dashboard can show actual vs expected by segment.
- A **`dashboard`** schema of SQL views, one per chart group, created by notebook `40_dashboard_views` from templates in `src/retail_promo/dashboard/views.py` and tested locally:
    - Overview: headline totals, weekly KPIs.
    - Promotion levers: mailer-week and in-store display lift by category with 95% CIs; the sensitivity table.
    - Segments: size and share of spend, actual vs expected behaviour, campaign redemption rate.
    - Targeting: capture curve from `gold.campaign_response_scores` (model vs coupon-share rule vs random), computed in SQL with the same top-k-per-campaign rule as notebook 30. This is a **back-test on past campaigns**, not a list of whom to target next.
    - Data quality: latest run's checks, rows flagged / modified / dropped by rule, run history.

## 2. Dashboard

Five pages (Overview, Promotion levers, Segments, Targeting: "how well the model would have targeted past campaigns", Data quality), each with a short text panel stating its caveats (associations not causal effects; mailer weeks include the price cut; segments describe behaviour and demographics cover heavier shoppers; redemption is not extra sales; what "flagged" means). Built by hand in the Databricks UI from a chart-by-chart guide (`docs/dashboard_guide.md`), on the Free Edition SQL warehouse. Saved inside the Git folder it is stored as a `.lvdash.json` file, which is committed so the dashboard is versioned with the code. Colours follow the project palette where the dashboard allows. After committing the dashboard from the Databricks Git dialog, pull locally before the next local commit so the two copies do not diverge. Screenshots of each page (aggregates only) are exported to `reports/figures/` for the README.

## 3. Scheduled job

- Tasks: `00_setup` → `01_bronze` → `02_silver` → `03_gold` → (`04_dq_report`, `10_promo_effectiveness`, `20_customer_segments`, `30_coupon_response`) in parallel → `40_dashboard_views`, which waits for 10, 20 and 30 because its views read the tables they write. At most 4 tasks run at once (Free Edition limit: 5).
- The job passes one `run_id` to every task, so the `ops` tables tie a whole run together.
- Recorded as `jobs/retail_pipeline.yml`; created in the UI from a step list, then compared with the UI's "Switch to code version (YAML)".
- The first job run is timed and its duration recorded in the README; run it at a quiet time (four tasks run in parallel, and notebook 10 reinstalls pyfixest each run).
- Schedule: weekly, **paused after one successful run** to protect the Free Edition daily compute quota (the data is static, so the schedule demonstrates the setup rather than refreshing new data).

## 4. README

Rewritten for a commercial reader: the business question, headline findings with numbers, recommendations, then how it is built and how to run it. The build history moves to the phase plans.

## 5. REPORT.md

Two to three pages for a **pricing and promotions manager**: context, data and its quality, promotion effectiveness, customer segments, coupon targeting, limitations, recommendations and next steps (randomised holdout, mailer placement, promotion ROI proxy). Every number cites the notebook section it comes from.

## Done when

- [x] `gold.weekly_kpis` and the dashboard views pass their tests locally; the views reproduce the notebook tables (lifts, segment profiles, capture curve)
- [x] Job created and one full run succeeds on Databricks (46 min 10 s, 2026-10-13); schedule to be paused
- [x] Dashboard built and published; screenshots in `reports/figures/` (the `.lvdash.json` file still to be committed from Databricks)
- [x] README and REPORT.md written; every REPORT number traced to a notebook

## Review changes and future work

Added after review: per-household weekly trend with the ramp-up flagged; targeting framed as a back-test; caveat panels on every page; tests that views match the notebook tables; job duration recorded; pull after committing the dashboard in Databricks; dashboard screenshots for the README.

Future work: a live scoring job that scores eligible households for an upcoming campaign (features as of today plus that campaign's coupon list); alerts on data-quality failures; publishing the dashboard on a schedule.
