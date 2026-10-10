# Databricks notebook source
# MAGIC %md
# MAGIC # 10 · Promotion effectiveness
# MAGIC **Goal:** estimate how much weekly unit sales rise when a product is in the weekly mailer or on an in-store display, for each of the largest categories, with 95% confidence intervals (CIs). These are **associations, not causal effects**: the retailer chooses what to promote and when, and the models cannot rule out that this choice tracks demand they do not see.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Product-week panel* (`gold.product_week_panel`): one row per product per week, weeks 9–101, products bought in at least 26 of those weeks, with units bought by the 2,500 panel households and promotion exposure. Weeks with no sales are kept as zeros.
# MAGIC - *Mailer-week lift*: the % change in a product's weekly units in weeks it is in the mailer. Mailer weeks usually also carry a price cut and sometimes a display, and price cannot be controlled for in weeks without sales, so this is the lift of the whole promotion event that comes with a mailer feature.
# MAGIC - *In-store display lift*: approximately the % lift in the stores that have the display. The panel records the traffic-weighted share of stores with a display; because units add up over stores, the model coefficient on that share approximates the in-store lift.
# MAGIC - *Fixed effects*: a separate baseline for each product (its normal sales level) and for each category in each week (seasonality and the household ramp-up), so a promoted week is compared with the same product's other weeks.
# MAGIC - *Poisson pseudo-maximum likelihood (PPML)*: the count regression used for weekly units; it keeps zero-sales weeks. Standard errors are clustered by product and by week.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Setup and model run
# MAGIC 2. Categories analysed
# MAGIC 3. Raw averages vs the fixed-effects model
# MAGIC 4. Mailer-week lift by category
# MAGIC 5. In-store display lift by category
# MAGIC 6. Product-level vs category-level lift
# MAGIC 7. Timing checks: the weeks before and after a mailer
# MAGIC 8. Discount depth (descriptive)
# MAGIC 9. Sensitivity checks
# MAGIC 10. Save results to gold
# MAGIC 11. Key findings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and model run
# MAGIC Free Edition restricts outbound internet; if this install fails, see "Future work" in `docs/phase2_plan.md`.

# COMMAND ----------

# MAGIC %pip install pyfixest

# COMMAND ----------

import os
import sys

REPO = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
FIG_DIR = os.path.join(REPO, "reports", "figures")

from retail_promo import pipeline
from retail_promo.analysis import phase2, plots
from retail_promo.config import Settings
from retail_promo.io import UnityCatalogIO

dbutils.widgets.text("catalog", "retail")
settings = Settings(catalog=dbutils.widgets.get("catalog"))
io = UnityCatalogIO(spark, settings.catalog, settings.landing_dir)

res = phase2.run(spark, io, settings)  # every model and sensitivity variant, a few minutes


def show(fig, name):
    plots.save(fig, FIG_DIR, name)
    display(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Categories analysed
# MAGIC Rule: the 20 largest commodities by panel sales among those with ≥ 20 products and ≥ 200 mailer product-weeks. A category's display lift is estimated only with ≥ 1,000 display product-weeks.

# COMMAND ----------

cats = res.categories
print(f"Share of panel sales in the selected categories: {cats['sales_share'].sum():.1%}")
print(f"Categories with a display estimate: {int(cats['display_estimable'].sum())} of {len(cats)}")
display(cats[["commodity_desc", "sales_rank", "sales_share", "products", "mailer_product_weeks",
              "display_product_weeks", "display_estimable"]])  # fmt: skip

# COMMAND ----------

# MAGIC %md
# MAGIC - The 20 selected categories hold 43.6% of panel sales. Sales ranks 15 and 21 are skipped because they have too few mailer product-weeks.
# MAGIC - 16 of the 20 have a display estimate. Beef (272 display product-weeks), pork (155), deli meats (81) and chicken (5) do not.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Raw averages vs the fixed-effects model

# COMMAND ----------

display(res.raw_contrast)
display(phase2.estimate_table(res, "product", "pooled"))

# COMMAND ----------

show(plots.raw_vs_model(res.raw_contrast, res.results), "phase2_raw_vs_model")

# COMMAND ----------

# MAGIC %md
# MAGIC - Raw averages: mailer-only product-weeks sell 8.15 units vs 1.58 in unpromoted weeks, a ratio of 5.16, i.e. +416%.
# MAGIC - With product and category-week fixed effects the mailer-week lift is +84.4% (95% CI +75.3% to +94.1%). The raw contrast is about five times larger (416 / 84.4 = 4.9) because the products that get promoted sell more anyway.
# MAGIC - Every later estimate uses the fixed-effects model.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Mailer-week lift by category

# COMMAND ----------

display(phase2.category_table(res, "mailer_share"))

# COMMAND ----------

show(plots.mailer_by_category(res.results), "phase2_mailer_by_category")

# COMMAND ----------

# MAGIC %md
# MAGIC - 19 of the 20 categories have a mailer-week lift whose CI is above zero. Beers/ales is the exception: +4.3% (−9.5% to +20.2%).
# MAGIC - Fresh meat and milk stand out: pork +745.4% (+524.8% to +1043.9%), chicken +356.4%, beef +317.5%, fluid milk +281.4%. These events may carry deeper price cuts than other categories' mailer weeks; this notebook cannot separate the two.
# MAGIC - Staple and snack categories respond least: bag snacks +19.0% (+10.8% to +27.8%), bread +20.2%, frozen pizza +26.2%.
# MAGIC - With 20 intervals, about one may miss its true value by chance.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. In-store display lift by category

# COMMAND ----------

display(phase2.category_table(res, "display_share"))

# COMMAND ----------

show(plots.display_by_category(res.results), "phase2_display_by_category")

# COMMAND ----------

# MAGIC %md
# MAGIC - In the stores that have it, a display goes with roughly +175% to +365% units: yogurt +175.4% (2.8× normal sales) to lunchmeat +364.9% (4.6×). Pooled over all products: +201.4% (+190.1% to +212.7%).
# MAGIC - Fluid milk's interval is the widest, +76.8% to +434.0%, so its ranking is uncertain.
# MAGIC - The in-store reading is an approximation; for larger display shares the coefficient understates the in-store lift.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Product-level vs category-level lift
# MAGIC Product-level lift counts sales a promoted product takes from other products in its category. The category model uses weekly category totals instead, so switching cancels out. Both are on the same footing: the approximate % lift if all of the category's sales were promoted.

# COMMAND ----------

display(phase2.estimate_table(res, "product", "selected"))
display(phase2.estimate_table(res, "category_net"))

# COMMAND ----------

show(plots.product_vs_category(res.results), "phase2_product_vs_category")

# COMMAND ----------

# MAGIC %md
# MAGIC - Mailer: category totals give +61.2% (+34.8% to +87.7%) against +97.6% (+82.3% to +114.1%) at product level on the same 20 categories. The category figure is 63% of the product figure (61.2 / 97.6), consistent with roughly a third of product-level lift being switching, but the intervals overlap.
# MAGIC - Display: +167.3% at category level vs +176.9% at product level; the category interval (+68.6% to +265.9%) is too wide to separate them.
# MAGIC - The category model has only 20 clusters, so its CIs may be too narrow rather than too wide.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Timing checks: the weeks before and after a mailer
# MAGIC One model with this week's, last week's and next week's mailer. A clear association with *next* week's mailer would mean promotions are timed to demand the model does not see (a placebo check). A negative association with *last* week's mailer would mean customers stocked up.

# COMMAND ----------

display(phase2.estimate_table(res, "timing"))

# COMMAND ----------

show(plots.timing(res.results), "phase2_timing")

# COMMAND ----------

# MAGIC %md
# MAGIC - In the mailer week: +85.4% (+75.9% to +95.4%), close to the main estimate.
# MAGIC - The week after: +7.1% (+3.3% to +10.9%). There is no post-promotion dip on average, so stockpiling does not visibly offset the lift; some promotions may run on into the next week.
# MAGIC - The week before: +3.9% (+1.0% to +6.9%). The placebo is not zero, so promotion timing tracks demand a little, but it is small next to the in-week lift (3.9% vs 85.4%).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Discount depth (descriptive)
# MAGIC Ordinary least squares (OLS) of log units on display, mailer and discount depth, weeks with sales only. Descriptive: it conditions on a purchase having happened, and depth is computed from the same transactions as units, so it is not a price elasticity.

# COMMAND ----------

display(phase2.estimate_table(res, "depth_descriptive"))

# COMMAND ----------

# MAGIC %md
# MAGIC - In weeks with sales, +10 percentage points of discount depth goes with +11.5% (+10.9% to +12.2%) more units, holding mailer and display fixed.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Sensitivity checks
# MAGIC The pooled main model re-estimated under each alternative setting.

# COMMAND ----------

display(phase2.sensitivity_table(res))

# COMMAND ----------

# MAGIC %md
# MAGIC - The mailer-week lift stays between +79.6% (product + week fixed effects) and +86.0% (products bought in ≥ 13 weeks) across all seven variants; the main estimate is +84.4%.
# MAGIC - The in-store display lift stays between +185.1% (products bought in ≥ 52 weeks) and +215.3% (≥ 13 weeks).
# MAGIC - Dropping weeks 9–13 (the household ramp-up) changes neither estimate by more than 0.3 points.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Save results to gold
# MAGIC Writes every estimate, including the sensitivity variants, to `gold.promo_lift_by_category` for the dashboard.

# COMMAND ----------

run_id = pipeline.latest_started_run(io)
phase2.write_gold(spark, io, res, run_id)
display(
    spark.table(f"{settings.catalog}.gold.promo_lift_by_category")
    .groupBy("model", "variant")
    .count()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Key findings
# MAGIC - **Mailer weeks:** products sell +84.4% (95% CI +75.3% to +94.1%) more units in weeks they are in the mailer, against +416% from a raw comparison (section 3). This is the lift of the whole promotion event, which usually includes a price cut.
# MAGIC - **Where it works:** fresh meat and milk show several-fold mailer-week lifts (pork +745.4%, chicken +356.4%, beef +317.5%, fluid milk +281.4%); snacks, bread and frozen pizza +19% to +26%; beer shows no detectable lift (section 4).
# MAGIC - **Displays:** in the stores that have one, a display goes with +175% to +365% units depending on category (section 5).
# MAGIC - **Net of switching:** category totals suggest a smaller mailer lift (+61.2% vs +97.6% at product level), but the intervals overlap (section 6).
# MAGIC - **Robustness:** no post-promotion dip (+7.1% the week after), a small non-zero placebo (+3.9% the week before), and the mailer-week lift stays within +79.6% to +86.0% across seven sensitivity variants (sections 7 and 9). All results describe the 2,500 panel households' purchases and are associations, not causal effects.
