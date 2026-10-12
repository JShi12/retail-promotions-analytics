# Databricks notebook source
# MAGIC %md
# MAGIC # 20 · Customer segments
# MAGIC **Goal:** group households by **how they buy on price and promotion**, so a promotions team can decide who to target with which kind of promotion. Spend, frequency, category mix, coupon redemption and demographics describe the segments; they do not define them. Method: Spark MLlib k-means.
# MAGIC
# MAGIC **Key terms**
# MAGIC - *Actual vs expected*: for each household, a behaviour (for example the share of its spend on promoted products) is compared with what other households show when buying **the same categories in the same stores**. Expected = the household's spend-weighted average of the category × store rates, computed from all other households (leave-one-out, so a household is not compared with itself). *Adjusted* = actual − expected; positive means more than expected.
# MAGIC - *Promotion share*: share of spend on products that were on display or in the mailer, in that store, that week (only where promotion status is recorded: 115 stores, weeks 9–101).
# MAGIC - *Discount depth*: loyalty-card discount as a % of the shelf price (the price before the discount).
# MAGIC - *Own label*: the retailer's private-label brand.
# MAGIC - *Coupon campaign*: one of 30 direct-marketing campaigns sent to selected households (1,584 of 2,500 received at least one), each carrying coupons for specific products valid between its start and end days. Different from the weekly **mailer**, the chain-wide advertising flyer analysed in notebook 10. *Redemption* = the household used at least one of that campaign's coupons.
# MAGIC - *Clustering inputs* (standardised): **deal-proneness index** = average of the standardised adjusted promotion share and adjusted discount depth; **adjusted own-label share**; **coupon use** = log(1 + 100 × share of purchase lines with a manufacturer coupon).
# MAGIC - *Silhouette*: how much closer households are to their own segment than to the next one (−1 to 1). *Shuffled baseline*: the silhouette of the same data with each input shuffled independently, which keeps each input's distribution but removes any structure between them.
# MAGIC - *Bootstrap Jaccard*: segment stability (Hennig 2007, *Computational Statistics & Data Analysis* 52): refit on resampled households and measure the overlap of each segment with its best match (1 = identical). A segment is stable if its mean is ≥ 0.75.
# MAGIC
# MAGIC **Contents**
# MAGIC 1. Setup and model run
# MAGIC 2. Behaviour measures
# MAGIC 3. Choosing the number of segments
# MAGIC 4. Segment centres
# MAGIC 5. Actual vs expected by segment
# MAGIC 6. Size, spend and coupon redemption
# MAGIC 7. Shopping profile
# MAGIC 8. Demographics
# MAGIC 9. Save segments to gold
# MAGIC 10. Key findings

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Setup and model run
# MAGIC Fits on the 2,402 households with at least 10 baskets; the rest are assigned to the nearest segment and flagged `low_activity`. Choosing k refits k-means many times (bootstrap and shuffled baselines), so this cell takes several minutes.

# COMMAND ----------

import os
import sys

REPO = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.insert(0, os.path.join(REPO, "src"))
FIG_DIR = os.path.join(REPO, "reports", "figures")

from retail_promo import pipeline
from retail_promo.analysis import phase3, plots
from retail_promo.config import Settings
from retail_promo.io import UnityCatalogIO

dbutils.widgets.text("catalog", "retail")
settings = Settings(catalog=dbutils.widgets.get("catalog"))
io = UnityCatalogIO(spark, settings.catalog, settings.landing_dir)

res = phase3.run(spark, io)
fit = res.assigned[~res.assigned["low_activity"]]


def show(fig, name):
    plots.save(fig, FIG_DIR, name)
    display(fig)


# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Behaviour measures
# MAGIC Spearman correlations between the adjusted measures, households used for fitting.

# COMMAND ----------

display(
    fit[["promo_adjusted", "disc_adjusted", "pl_adjusted", "log_coupon"]]
    .corr(method="spearman")
    .round(2)
    .reset_index()
)

# COMMAND ----------

# MAGIC %md
# MAGIC - Adjusted promotion share and adjusted discount depth correlate 0.77, so they measure one behaviour and are combined into the deal-proneness index rather than counted twice.
# MAGIC - Own label (0.34 and 0.38 with the two deal measures) and coupon use (0.24 and 0.29) carry different information, so they stay separate inputs.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Choosing the number of segments
# MAGIC Rule: the largest k whose segments are all stable (bootstrap Jaccard ≥ 0.75) and whose silhouette beats the shuffled baseline.

# COMMAND ----------

display(res.k_summary)
display(res.k_segments[res.k_segments["k"] == res.k])

# COMMAND ----------

show(plots.k_selection(res.k_summary, res.k), "phase3_k_selection")

# COMMAND ----------

# MAGIC %md
# MAGIC - k = 3 is chosen: silhouette 0.463 against 0.385 for shuffled data (gap 0.079, the largest), and every segment is stable (Jaccard 0.922, 0.921 and 0.949 for segments 1–3).
# MAGIC - k = 2 is stable but no better than shuffled data (0.479 vs 0.476). From k = 4 at least one segment falls below 0.75 (lowest 0.667 at k = 4).
# MAGIC - A silhouette near 0.46 means real but modest structure: these are useful groups along a continuum, not sharply separated types.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Segment centres
# MAGIC Segments are numbered by deal-proneness (1 = highest). Labels name only the inputs more than half a standard deviation from average.

# COMMAND ----------

display(res.centroids)

# COMMAND ----------

show(plots.segment_centroids(res.centroids), "phase3_centroids")

# COMMAND ----------

# MAGIC %md
# MAGIC - **Segment 1, more promotion, more coupons:** deal-proneness +0.64 standard deviations and coupon use +1.65; own label about average (−0.18).
# MAGIC - **Segment 2, more promotion, more own label:** deal-proneness +0.52 and own label +0.93; coupon use −0.35.
# MAGIC - **Segment 3, less promotion, less own label:** deal-proneness −0.60, own label −0.56, coupon use −0.40.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Actual vs expected by segment
# MAGIC Medians of the households used for fitting.

# COMMAND ----------

display(res.profiles[["segment", "segment_label", "households", "promo_actual", "promo_expected", "disc_actual",
                      "disc_expected", "pl_actual", "pl_expected"]])  # fmt: skip

# COMMAND ----------

show(plots.actual_vs_expected(res.profiles), "phase3_actual_vs_expected")

# COMMAND ----------

# MAGIC %md
# MAGIC - Segment 1 puts 26.7% of spend on promotion against 22.8% expected (+3.9 points) and takes 18.8% off the shelf price against 16.2% expected.
# MAGIC - Segment 2 is the own-label buyer: 29.9% of spend against 22.4% expected (+7.5 points), plus promotion 26.4% vs 23.1% (+3.3 points).
# MAGIC - Segment 3 buys less on promotion than its categories and stores predict: 17.8% vs 21.8% (−4.0 points), 13.2% vs 15.5% off shelf price, and own label 17.4% vs 21.5% (−4.1 points).
# MAGIC - The expected values are close across segments (21.8%–23.1% for promotion), so the differences come from behaviour, not from where or what the segments buy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Size, spend and coupon redemption
# MAGIC Redemption is descriptive: the share of household-campaign pairs with at least one campaign coupon redeemed (`gold.campaign_household`). It is not a clustering input. Because coupon use (an input) could include those same redemptions, the second table shows coupon use with and without the days a household redeemed a campaign coupon.

# COMMAND ----------

display(
    res.profiles[
        [
            "segment",
            "segment_label",
            "households",
            "share_of_spend",
            "campaign_pairs",
            "redemption_rate",
        ]
    ]
)
display(res.coupon_check)

# COMMAND ----------

show(plots.size_and_redemption(res.profiles), "phase3_size_redemption")

# COMMAND ----------

# MAGIC %md
# MAGIC - Segment 3 is the core: 1,160 of 2,402 households (48%) and 49.0% of spend. Segment 1 is 454 households (19%) but 22.9% of spend; segment 2 is 788 households (33%) and 28.1% of spend.
# MAGIC - Segment 1 redeems campaign coupons in 30.4% of its campaign pairs against 6.4% for segments 2 and 3 (30.4 / 6.4 = 4.8×).
# MAGIC - That is not built in: excluding redemption days, segment 1's median coupon share barely moves (2.81% → 2.64%) and stays far above segments 2 and 3 (0.21% and 0.16%). Only 15.3% of segment 1's coupon lines fall on redemption days.
# MAGIC - Segment 2 buys more on promotion than expected yet redeems no more than segment 3 (6.4% each): buying on in-store promotion is not the same as responding to coupons.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Shopping profile
# MAGIC Descriptors, not clustering inputs. Medians of the households used for fitting.

# COMMAND ----------

display(res.profiles[["segment", "segment_label", "baskets_per_week", "avg_basket_spend", "total_spend", "share_grocery",
                      "share_drug_gm", "share_produce", "share_meat", "share_deli"]])  # fmt: skip

# COMMAND ----------

# MAGIC %md
# MAGIC - Segment 1 shops most often (1.17 baskets a week vs 0.85 and 0.83) and spends the most over two years ($2,868 vs $1,869 and $2,137), so the coupon users are also the most frequent shoppers.
# MAGIC - Segment 2 has the smallest baskets ($22.25 vs $27.33 and $26.72) and the highest meat share (14.0% vs 11.7% and 11.0%).
# MAGIC - Category mixes are otherwise similar (grocery 55.0%–57.2% of spend), consistent with segments defined by behaviour rather than what households buy.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Demographics
# MAGIC Only 801 of 2,500 households have demographics, and they are not typical: the first table compares them with the rest. The `classification_*_ord` columns are dunnhumby's anonymised ordinal groups (meanings not documented); the second table shows their mean by segment.

# COMMAND ----------

display(res.representativeness)
display(res.demographics)

# COMMAND ----------

# MAGIC %md
# MAGIC - Households with demographics shop 1.52 baskets a week against 0.57 for the rest (×2.7) and spend $4,212 against $1,318 (×3.2), so any demographic profile describes heavier shoppers.
# MAGIC - Mean demographic levels differ little between segments (for example `classification_3_ord` 4.67, 4.31 and 4.97), so the segments are not explained by these demographics.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Save segments to gold
# MAGIC `gold.household_segments`: one row per household with its segment, label, clustering inputs and flags. It uses two years of purchases, including weeks after every campaign, so **the Phase 4 response model must not use it as a feature**.

# COMMAND ----------

phase3.write_gold(spark, io, res, pipeline.latest_started_run(io))
display(
    spark.table(f"{settings.catalog}.gold.household_segments")
    .groupBy("segment", "segment_label", "low_activity")
    .count()
    .orderBy("segment", "low_activity")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Key findings
# MAGIC - **Three stable segments** on price-and-promotion behaviour (k = 3: silhouette 0.463 vs 0.385 shuffled; every segment's Jaccard ≥ 0.92). The structure is real but modest: segments are useful groups along a continuum (section 3).
# MAGIC - **More promotion, more coupons (segment 1):** 19% of households, 22.9% of spend, the most frequent shoppers, and they redeem campaign coupons in 30.4% of campaigns against 6.4% for the others (sections 6 and 7). This is the natural target for coupon campaigns.
# MAGIC - **More promotion, more own label (segment 2):** 33% of households, 28.1% of spend, own label 29.9% of spend vs 22.4% expected (section 5). They buy more on promotion and own label than expected but redeem campaign coupons no more than segment 3 (6.4% each), which points to in-store price promotions and own-label offers rather than coupons; testing that needs a response measure by segment.
# MAGIC - **Less promotion, less own label (segment 3):** 48% of households and 49.0% of spend, buying 4.0 points less on promotion than their categories and stores predict (section 5), so promotions appear to matter less to their buying. Whether a promotion changes what they buy, or mainly discounts purchases they would make anyway, needs a response measure by segment (future work in `docs/phase3_plan.md`).
# MAGIC - **Caveats:** segments are descriptive and built from two years of data; demographics cover only heavier shoppers (section 8); and segment membership must not be used as a Phase 4 feature.
