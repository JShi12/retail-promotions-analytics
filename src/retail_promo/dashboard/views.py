"""Dashboard views: one per chart group, all reading gold and ops tables.

Each view is a SQL template with `{gold}` and `{ops}` placeholders (e.g. `retail.gold`), so the
same SQL runs on Databricks and in local tests. The dashboard only does `SELECT * FROM view`.
"""

from __future__ import annotations

from pyspark.sql import SparkSession

from retail_promo.config import LAST_DAY

SHARES = "0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00"

VIEWS: dict[str, str] = {
    # ---------------------------------------------------------------- overview
    "kpi_headline": """
        SELECT
          SUM(sales) AS total_sales,
          (SELECT COUNT(*) FROM {gold}.household_features) AS households,
          SUM(baskets) AS baskets,
          SUM(sales) / SUM(baskets) AS avg_basket_spend,
          SUM(promo_sales) / SUM(promo_observed_sales) AS promo_share_of_sales,
          SUM(loyalty_discount) AS loyalty_discount,
          COUNT(*) AS weeks
        FROM {gold}.weekly_kpis""",
    # Days in a week follow from week_no = floor((day + 1) / 7) + 1 over days 1..LAST_DAY: week w
    # covers days 7w-8 .. 7w-2, so week 1 has 5 days and week 102 has 6. Partial weeks are flagged
    # so trend charts can leave them out.
    "weekly_kpis": """
        SELECT week_no, sales, sales_per_household, active_households, baskets, avg_basket_spend,
               promo_share_of_sales, ramp_up,
               LEAST({last_day}, 7 * week_no - 2) - GREATEST(1, 7 * week_no - 8) + 1 AS days_in_week,
               LEAST({last_day}, 7 * week_no - 2) - GREATEST(1, 7 * week_no - 8) + 1 = 7 AS full_week,
               CASE WHEN ramp_up THEN 'Panel ramp-up (weeks 1-13)' ELSE 'Full panel' END AS period
        FROM {gold}.weekly_kpis""",
    # ---------------------------------------------------------------- promotion levers
    "mailer_lift_by_category": """
        SELECT CASE WHEN scope = 'pooled' THEN 'ALL PRODUCTS' ELSE category END AS category,
               lift, lift_low, lift_high, n_obs AS product_weeks
        FROM {gold}.promo_lift_by_category
        WHERE model = 'product' AND term = 'mailer_share' AND variant = 'main'
          AND scope IN ('category', 'pooled')""",
    "display_lift_by_category": """
        SELECT CASE WHEN scope = 'pooled' THEN 'ALL PRODUCTS' ELSE category END AS category,
               lift, lift_low, lift_high, n_obs AS product_weeks
        FROM {gold}.promo_lift_by_category
        WHERE model = 'product' AND term = 'display_share' AND variant = 'main'
          AND scope IN ('category', 'pooled')""",
    "lift_sensitivity": """
        SELECT variant, term, lift, lift_low, lift_high, n_obs
        FROM {gold}.promo_lift_by_category
        WHERE model = 'product' AND scope = 'pooled' AND category = 'ALL'""",
    "promo_timing": """
        SELECT CASE term WHEN 'mailer_lead' THEN 'Week before the mailer'
                         WHEN 'mailer_share' THEN 'Mailer week'
                         WHEN 'mailer_lag' THEN 'Week after the mailer' END AS week,
               lift, lift_low, lift_high
        FROM {gold}.promo_lift_by_category
        WHERE model = 'timing'""",
    # ---------------------------------------------------------------- segments
    "segment_profile": """
        WITH s AS (
          SELECT s.*, f.total_spend, f.baskets_per_week, f.avg_basket_spend
          FROM {gold}.household_segments s JOIN {gold}.household_features f USING (household_key)
          WHERE NOT s.low_activity)
        SELECT segment, segment_label,
               COUNT(*) AS households,
               COUNT(*) / SUM(COUNT(*)) OVER () AS share_of_households,
               SUM(total_spend) / SUM(SUM(total_spend)) OVER () AS share_of_spend,
               PERCENTILE(promo_actual, 0.5) AS promo_actual, PERCENTILE(promo_expected, 0.5) AS promo_expected,
               PERCENTILE(disc_actual, 0.5) AS disc_actual, PERCENTILE(disc_expected, 0.5) AS disc_expected,
               PERCENTILE(pl_actual, 0.5) AS pl_actual, PERCENTILE(pl_expected, 0.5) AS pl_expected,
               PERCENTILE(baskets_per_week, 0.5) AS baskets_per_week,
               PERCENTILE(avg_basket_spend, 0.5) AS avg_basket_spend
        FROM s GROUP BY segment, segment_label""",
    "segment_redemption": """
        SELECT s.segment, s.segment_label, COUNT(*) AS campaign_pairs, AVG(c.redeemed) AS redemption_rate
        FROM {gold}.campaign_household c
        JOIN {gold}.household_segments s USING (household_key)
        WHERE NOT s.low_activity
        GROUP BY s.segment, s.segment_label""",
    # ---------------------------------------------------------------- targeting (back-test)
    "capture_curve": """
        WITH r AS (
          SELECT campaign, redeemed,
                 COUNT(*) OVER (PARTITION BY campaign) AS n,
                 ROW_NUMBER() OVER (PARTITION BY campaign ORDER BY score DESC, household_key) AS rank_model,
                 ROW_NUMBER() OVER (PARTITION BY campaign ORDER BY coupon_line_share DESC, household_key)
                   AS rank_rule
          FROM {gold}.campaign_response_scores),
        shares AS (SELECT EXPLODE(ARRAY("""
    + SHARES
    + """)) AS share),
        total AS (SELECT SUM(redeemed) AS t FROM {gold}.campaign_response_scores)
        SELECT share AS share_targeted, share AS random,
               SUM(CASE WHEN rank_rule <= CEIL(share * n) THEN redeemed ELSE 0 END) / MAX(t) AS rule_coupon_share,
               SUM(CASE WHEN rank_model <= CEIL(share * n) THEN redeemed ELSE 0 END) / MAX(t) AS model
        FROM r CROSS JOIN shares CROSS JOIN total
        GROUP BY share""",
    # ---------------------------------------------------------------- data quality
    "dq_latest_checks": """
        WITH latest AS (
          SELECT run_id FROM {ops}.pipeline_runs WHERE status = 'succeeded'
          ORDER BY finished_at DESC LIMIT 1)
        SELECT c.layer, c.table_name, c.check_id, c.severity, c.passed, c.observed, c.expected, c.run_id
        FROM {ops}.dq_check_results c JOIN latest USING (run_id)""",
    "dq_check_status": """
        WITH latest AS (
          SELECT run_id FROM {ops}.pipeline_runs WHERE status = 'succeeded'
          ORDER BY finished_at DESC LIMIT 1)
        SELECT c.layer, c.severity, SUM(CASE WHEN c.passed THEN 1 ELSE 0 END) AS passed,
               SUM(CASE WHEN c.passed THEN 0 ELSE 1 END) AS failed
        FROM {ops}.dq_check_results c JOIN latest USING (run_id)
        GROUP BY c.layer, c.severity""",
    "dq_rule_counts": """
        WITH latest AS (
          SELECT run_id FROM {ops}.pipeline_runs WHERE status = 'succeeded'
          ORDER BY finished_at DESC LIMIT 1)
        SELECT d.table_name, d.rule_id, d.action, d.n_rows
        FROM {gold}.dq_summary d JOIN latest USING (run_id)""",
    "pipeline_runs": """
        SELECT run_id, status, finished_at, detail FROM {ops}.pipeline_runs""",
}


def render(name: str, gold: str, ops: str) -> str:
    return VIEWS[name].format(gold=gold, ops=ops, last_day=LAST_DAY)


def create_views(spark: SparkSession, gold: str, ops: str, dashboard: str) -> list[str]:
    """Create or replace every view in the `dashboard` schema; returns the view names."""
    for name in VIEWS:
        spark.sql(f"CREATE OR REPLACE VIEW {dashboard}.{name} AS {render(name, gold, ops)}")
    return list(VIEWS)
