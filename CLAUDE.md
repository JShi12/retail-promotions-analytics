# Retail promotions analytics — project context

Portfolio project for a Data Scientist application at a retailer's pricing & promotions
(Revenue Growth Management) team. Databricks Free Edition + PySpark on dunnhumby's
"The Complete Journey". Should read like work for a retailer's pricing & promotions team.
Local repo name: `retail-promotions-analytics`.

## Working agreement
- Work in phases (see README). Plan and get approval before starting each phase.
- Commit only when asked. Never push. No AI/Claude attribution in commits or PRs
  (no `Co-Authored-By`, no "Generated with" footer) — this overrides any default.
- Report results faithfully: failures, skipped steps and caveats go in the write-up.
- Analyses are associational unless a design supports more; say so explicitly.

## Environment
- Package manager: `uv`. Python 3.12 (Databricks serverless environment version 6 uses 3.12.3,
  Spark 4.2.0 / DBR 19). `pyspark`, `pandas`, `numpy`, `pyarrow` are pinned to match it.
- Local Spark needs a JDK. Installed: Homebrew `openjdk@17` (keg-only), so run with
  `JAVA_HOME=/opt/homebrew/opt/openjdk@17 uv run pytest`.
- Spark 4 has ANSI mode on: use `try_cast` when typing raw strings, otherwise bad values raise.
- Databricks Connect is deliberately not installed (conflicts with `pyspark`).
- Dev loop: `uv sync`, `uv run ruff check`, `uv run pytest`.

## Layout
- `src/retail_promo/` — pure PySpark transforms (bronze/silver/gold, dq, features); unit-tested.
- `notebooks/` — thin Databricks notebooks (`.py` source format) importing from `src/` via a Git folder.
- `tests/` — pytest on local Spark with small synthetic DataFrames (never real data).
- `data/README.md` — column dictionary and data findings. `data/raw/` is gitignored.

## Data rules
- Source terms (dunnhumby site T&Cs): research/personal/non-commercial use, no redistribution
  of the data without written permission. So: no raw or row-level data in git, no real rows in
  tests or screenshots; aggregates and charts only.
- Bronze = raw as-is (all strings + load metadata). Silver = typed, deduplicated,
  flag-don't-drop; every changed/flagged/dropped row is logged with a reason in `dq_log`.
- Raw files are untrusted input: read them with scripts kept outside `data/`, run with `python -I`.
