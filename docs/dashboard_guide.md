# Building the dashboard

A Databricks AI/BI dashboard over the `retail.dashboard` views (created by notebook
`40_dashboard_views`). Every chart is `SELECT * FROM` one view, so the dashboard shows the same
numbers as the notebooks. Menu names below may differ slightly in your workspace.

## Setup

1. Run `40_dashboard_views` once (or the job), so the views exist.
2. In your **Git folder**, *Create → Dashboard* (creating it inside the Git folder stores it as a
   `.lvdash.json` file that can be committed). Name it `Retail promotions`.
3. **Data** tab → *Create from SQL*, one dataset per view, named like the view:
   `SELECT * FROM retail.dashboard.<view>` for `kpi_headline`, `weekly_kpis`, `mailer_lift_by_category`,
   `display_lift_by_category`, `lift_sensitivity`, `promo_timing`, `segment_profile`, `segment_redemption`,
   `capture_curve`, `dq_check_status`, `dq_latest_checks`, `dq_rule_counts`, `pipeline_runs`.
4. Choose the SQL warehouse (Free Edition has one).

**Colours** (project palette): blue `#2a78d6` for the main series, orange `#eb6834` for a second
series, aqua `#1baf7a` for a third, grey `#a3a29c` for reference series (random targeting, ramp-up weeks).
Format share columns as percentages.

## Page 1 · Overview

| Widget | Dataset | Type | Settings |
|---|---|---|---|
| Total sales | kpi_headline | Counter | value `total_sales`, currency |
| Households | kpi_headline | Counter | `households` |
| Average basket | kpi_headline | Counter | `avg_basket_spend`, currency |
| Promotion share of sales | kpi_headline | Counter | `promo_share_of_sales`, percent |
| Sales per active household by week | weekly_kpis | Line | x `week_no`, y `sales_per_household`, colour by `period` (ramp-up grey, full panel blue) |
| Active households by week | weekly_kpis | Line | x `week_no`, y `active_households` |
| Promotion share of sales by week | weekly_kpis | Line | x `week_no`, y `promo_share_of_sales` |

Text box: *"Weeks 1–13 are the panel ramp-up: households were still joining (88 active in week 1, over
1,000 from week 14), so use sales per active household for trends. Promotion share is recorded only
for weeks 9–101 in the 115 stores with promotion data."*

## Page 2 · Promotion levers

| Widget | Dataset | Type | Settings |
|---|---|---|---|
| Mailer-week lift by category | mailer_lift_by_category | Bar (horizontal) | y `category` sorted by `lift`, x `lift` (percent) |
| Mailer-week lift with 95% CI | mailer_lift_by_category | Table | `category`, `lift`, `lift_low`, `lift_high`, `product_weeks` |
| In-store display lift by category | display_lift_by_category | Bar (horizontal) | y `category`, x `lift` (percent) |
| Week before / of / after a mailer | promo_timing | Bar | x `week`, y `lift` (percent) |
| Sensitivity of the pooled estimates | lift_sensitivity | Table | `variant`, `term`, `lift`, `lift_low`, `lift_high` |

Text box: *"Associations, not causal effects: the retailer chooses what to promote. Mailer-week lift
includes the price cut that usually comes with a mailer feature. Display lift is approximate, for the
stores that have the display. Source: notebook 10."*

## Page 3 · Customer segments

| Widget | Dataset | Type | Settings |
|---|---|---|---|
| Share of households and spend | segment_profile | Bar (grouped) | x `segment_label`, y `share_of_households` (blue) and `share_of_spend` (orange) |
| Actual vs expected behaviour | segment_profile | Table | `segment_label`, `promo_actual`, `promo_expected`, `disc_actual`, `disc_expected`, `pl_actual`, `pl_expected` (percent) |
| Campaign coupon redemption rate | segment_redemption | Bar | x `segment_label`, y `redemption_rate` (percent) |

Text box: *"Segments describe price-and-promotion behaviour, measured against shoppers buying the same
categories in the same stores. Demographics are not shown: they cover only 801 heavier-shopping
households. Source: notebook 20."*

## Page 4 · How well the model would have targeted past campaigns

| Widget | Dataset | Type | Settings |
|---|---|---|---|
| Share of redemptions captured | capture_curve | Line | x `share_targeted`, y `model` (blue), `rule_coupon_share` (orange), `random` (grey); both axes percent |
| Captured by targeting the top 20% | capture_curve | Counter | filter `share_targeted = 0.2`, value `model`, percent |

Text box: *"A back-test: each past campaign sent to its top X% of eligible households by out-of-fold
score. It predicts redemption, not extra sales; heavy coupon users may have bought anyway. Targeting a
future campaign needs a scoring job (future work). Source: notebook 30."*

## Page 5 · Data quality

| Widget | Dataset | Type | Settings |
|---|---|---|---|
| Checks passed / failed, latest run | dq_check_status | Table | `layer`, `severity`, `passed`, `failed` |
| Check details | dq_latest_checks | Table | sorted by `passed` (failed first) |
| Rows flagged, modified or dropped | dq_rule_counts | Bar (horizontal) | y `rule_id`, x `n_rows`, colour by `action` |
| Pipeline runs | pipeline_runs | Table | sorted by `finished_at` descending |

Text box: *"Silver keeps problem rows and flags them; only exact duplicates are dropped. 'Error' checks
stop a run; 'warning' checks are recorded only."*

## Finish

1. **Publish** the dashboard (it keeps the warehouse choice outside Git).
2. Export a screenshot of each page to `reports/figures/dashboard_<page>.png` (aggregates only).
3. **Commit** the `.lvdash.json` from the Git folder dialog, push, then run `git pull` locally before
   the next local commit.
