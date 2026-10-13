# Databricks notebook source
# MAGIC %md
# MAGIC # 30 · Coupon campaign response
# MAGIC **Goal:** predict which households will redeem a coupon campaign they receive, using only what was known before the campaign started, and turn that into a targeting recommendation. The model predicts **redemption, not extra sales**: there is no randomised control group, so it cannot show whether a coupon changed what a household bought.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Coupon campaign*: one of 30 direct-marketing campaigns sent to selected households, each carrying coupons for specific products valid between its start and end days. *Redeemed* = the household used at least one of that campaign's coupons. One row per household per campaign received (`gold.campaign_household`, 7,208 rows, 889 redeemed).
# MAGIC - *Pre-campaign features*: household behaviour, campaign history and campaign design, all computed from data before the campaign's start day.
# MAGIC - *ROC AUC (area under the receiver operating characteristic curve)*: the chance that a randomly chosen redeemer scores higher than a randomly chosen non-redeemer (0.5 = random, 1 = perfect).
# MAGIC - *Within-campaign AUC*: AUC computed inside each campaign, averaged with weights equal to its redemptions. It measures the targeting decision (who receives a given campaign); pooled AUC also rewards telling campaigns apart. **Primary metric.**
# MAGIC - *PR AUC (area under the precision-recall curve)*: average precision; random scores give the redemption rate, 12.3%.
# MAGIC - *Top-decile lift*: redemption rate among each campaign's top 10% of scores ÷ the overall rate.
# MAGIC - *Household-grouped folds*: 5 cross-validation folds with every household's campaigns in one fold, so no household is in both training and test. Models are tuned in an inner 3-fold household-grouped loop on within-campaign AUC.
# MAGIC - *Coupon-share rule*: rank households by the share of their pre-campaign purchase lines that used a manufacturer coupon. The bar the models must beat, fold by fold.
# MAGIC - *GBT (gradient-boosted trees)* and *logistic regression with an L1 penalty* (shrinks weak features to zero): the two models, fitted with scikit-learn on the notebook's driver. Spark reads and prepares the data; the table has only 7,208 rows, and Spark MLlib's boosted trees run one Spark job per tree, which made the first Databricks run take hours.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Setup and model run
# MAGIC 2. Drift over time
# MAGIC 3. Models vs rules
# MAGIC 4. Tuning
# MAGIC 5. Targeting: capture curve
# MAGIC 6. What drives the score
# MAGIC 7. Time-forward check
# MAGIC 8. Save scores to gold
# MAGIC 9. Key findings and recommendation

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and model run
# MAGIC Runs both models through nested household-grouped cross-validation, permutation importance, the L1 path and the time-forward check (under a minute locally).

# COMMAND ----------

import os
import sys

REPO = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))

from retail_promo import pipeline
from retail_promo._version import check_fresh
from retail_promo.analysis import plots, response
from retail_promo.config import Settings
from retail_promo.io import UnityCatalogIO

dbutils.widgets.text("catalog", "retail")
settings = Settings(catalog=dbutils.widgets.get("catalog"))
check_fresh()  # stops if this session is running code older than the Git folder
# Figures go to a Unity Catalog volume, not the Git folder (the copies in reports/figures/
# come from the local run of the same code).
spark.sql(f"CREATE VOLUME IF NOT EXISTS {settings.catalog}.ops.figures")
FIG_DIR = f"/Volumes/{settings.catalog}/ops/figures"
io = UnityCatalogIO(spark, settings.catalog, settings.landing_dir)

res = response.run(io.read("gold", "campaign_household").toPandas())


def show(fig, name):
    plots.save(fig, FIG_DIR, name)
    display(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Drift over time
# MAGIC Pairs grouped by campaign start day.

# COMMAND ----------

display(res.drift)

# COMMAND ----------

# MAGIC %md
# MAGIC - The redemption rate rises from 7.5% for campaigns starting before day 400 to 14.8% (days 500–586) and 14.0% (day 587 on).
# MAGIC - History features carry little early on: 65.1% of pairs before day 400 have no earlier campaign, and only 6.0% have a prior redemption, against 7.0% and 35.4% from day 587.
# MAGIC - Even the coupon-share rule moves between periods (within-campaign AUC 0.740 to 0.817), so the time-forward check in section 7 matters.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Models vs rules
# MAGIC Out-of-fold scores, mean ± SD over the 5 household-grouped folds. The second table compares each model with the coupon-share rule fold by fold (model AUC − rule AUC on the same fold).

# COMMAND ----------

display(res.summary)
display(res.paired)

# COMMAND ----------

show(plots.model_comparison(res.summary, res.best_model), "phase4_model_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC - Gradient-boosted trees reach a within-campaign AUC of 0.819 ± 0.010, logistic regression 0.810 ± 0.015, the coupon-share rule 0.784 ± 0.005 and the past-redeemer rule 0.673 ± 0.010.
# MAGIC - Fold by fold, GBT beats the coupon-share rule by +0.035 ± 0.009 and logistic regression by +0.027 ± 0.014; both win in 5 of 5 folds. That clears the success bar set in the plan.
# MAGIC - The gain shows most at the top of the ranking: PR AUC 0.474 ± 0.016 for GBT vs 0.324 ± 0.012 for the rule (random = 0.123), and top-decile lift 3.65 ± 0.25 vs 3.02 ± 0.28.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Tuning
# MAGIC Parameters chosen in each outer fold, and the inner-CV within-campaign AUC of every grid setting (mean over the 5 outer folds).

# COMMAND ----------

for name, table in res.chosen.items():
    print(name)
    display(table)
grid_cols = [c for c in res.grid.columns if c not in ("model", "fold", "inner_auc_within")]
display(
    res.grid.groupby(["model", *grid_cols], dropna=False)["inner_auc_within"]
    .agg(["mean", "std"])
    .reset_index()
)

# COMMAND ----------

# MAGIC %md
# MAGIC - GBT chooses depth 1 with 100 trees in 3 of 5 folds (otherwise depth 1 with 200, or depth 2 with 50), all inside the grid. Depth-1 trees use one feature each, so the model adds up single-feature effects without interactions; deeper trees score lower (depth 3 with 400 trees: 0.780).
# MAGIC - At depth 1 the score peaks at 100 trees (0.813 with 50, 0.818 with 100, 0.817 with 200, 0.812 with 400), so more trees start to overfit.
# MAGIC - Logistic regression chooses C = 0.03 in 3 folds and 0.1 in 2, inside its grid; its settings score 0.790–0.808.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Targeting: capture curve
# MAGIC Each campaign is sent to its top X% of eligible households by score; the curve shows the share of all redemptions captured.

# COMMAND ----------

display(res.capture_curve)

# COMMAND ----------

show(plots.capture_curve(res.capture_curve, res.best_model), "phase4_capture_curve")

# COMMAND ----------

# MAGIC %md
# MAGIC - Sending each campaign to its top 20% by GBT score captures 58.5% of redemptions, against 51.4% with the coupon-share rule and 20% at random.
# MAGIC - The advantage is largest for small target lists: top 10% captures 38.2% vs 31.9%; by the top 50% it narrows to 86.2% vs 83.7%.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. What drives the score
# MAGIC Grouped permutation importance: the drop in within-campaign AUC when a group of correlated features is shuffled together inside each campaign, on held-out folds (mean ± SD over 5 folds). Campaign-design features are constant within a campaign, so shuffling them inside a campaign changes nothing by construction. The second table is the L1 path: standardised logistic coefficients as the penalty loosens (only features that are ever non-zero).

# COMMAND ----------

imp = res.importance.groupby(["model", "group"])["auc_drop"].agg(["mean", "std"]).reset_index()
display(imp.sort_values(["model", "mean"], ascending=[True, False]))
path = res.path.pivot(index="feature", columns="C", values="coef")
display(path[(path != 0).any(axis=1)].reset_index())

# COMMAND ----------

show(plots.importance(res.importance, res.best_model), "phase4_importance")

# COMMAND ----------

# MAGIC %md
# MAGIC - For GBT, shuffling coupon use cuts within-campaign AUC by 0.178 ± 0.013; the next groups are past redemptions (0.023 ± 0.003), shopping volume (0.013 ± 0.006), relevance to this campaign (0.008 ± 0.006) and category mix (0.007 ± 0.003).
# MAGIC - Logistic regression leans on coupon use (0.067 ± 0.007) and past redemptions (0.074 ± 0.010) about equally.
# MAGIC - On the L1 path, the first features to enter (at the strongest penalty with any, C = 0.003) are coupon share (0.237), spend on the campaign's coupon products (0.150) and the past-redemption features. The number of coupons in the campaign enters at C = 0.01, but it is constant within a campaign, so it cannot change who is targeted.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Time-forward check
# MAGIC Trained on campaigns that ended before day 587 and tested on campaigns starting on or after it (campaigns 18–25), as the model would be used in practice.

# COMMAND ----------

display(res.time_forward)

# COMMAND ----------

show(plots.time_forward(res.time_forward), "phase4_time_forward")

# COMMAND ----------

# MAGIC %md
# MAGIC - On the later campaigns the model reaches a within-campaign AUC of 0.835 against 0.799 for the coupon-share rule, a similar gap to cross-validation.
# MAGIC - Campaign 18 holds 214 of the 324 redemptions: 0.848 vs 0.805. The model also scores higher in campaigns 19, 20, 21, 23 and 24, and lower in 22 (0.817 vs 0.824) and 25 (0.806 vs 0.821).
# MAGIC - Small campaigns are noisy: campaign 21 has 4 redemptions (0.650 vs 0.574).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Save scores to gold
# MAGIC `gold.campaign_response_scores`: one out-of-fold score per household-campaign pair from the chosen model, for the dashboard.

# COMMAND ----------

response.write_gold(spark, io, res, pipeline.latest_started_run(io))
display(
    spark.table(f"{settings.catalog}.gold.campaign_response_scores")
    .groupBy("model", "fold")
    .count()
    .orderBy("fold")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Key findings and recommendation
# MAGIC - **The model beats the best simple rule:** gradient-boosted trees rank redeemers within a campaign at AUC 0.819 ± 0.010 against 0.784 ± 0.005 for the coupon-share rule, +0.035 ± 0.009 fold by fold, winning in all 5 folds (section 3), and 0.835 vs 0.799 on later campaigns (section 7).
# MAGIC - **Recommendation:** choose who receives each campaign by model score. Sending a campaign to the top 20% of eligible households captures 58.5% of its redemptions, against 51.4% by coupon share and 20% at random (section 5).
# MAGIC - **Coupon use is the main signal**, then past redemptions (section 6). If a model cannot be deployed, ranking by pre-campaign coupon share keeps most of the gain over random.
# MAGIC - **Limits:** the model predicts redemption, not extra sales. Frequent coupon users may have bought the products anyway, so targeting them maximises redemptions, and coupon cost, not necessarily incremental revenue. Results also apply only to households the retailer already selects for campaigns.
# MAGIC - **Next step:** hold back a random share of eligible households in future campaigns (a randomised holdout). Comparing their purchases with households that received the coupons measures the true effect of a campaign, and would allow an uplift model that targets households whose buying a coupon actually changes.
