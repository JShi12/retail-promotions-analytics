# Phase 4 plan: coupon campaign response model

2026-10-12 · approved plan (revised after review)

Phase 4 predicts whether a household will redeem at least one coupon from a campaign it receives, using only information available before the campaign starts, and turns the scores into a targeting recommendation. It predicts **redemption, not incremental sales**: the data has no randomised control group, so the model cannot say whether a coupon changed what a household bought.

Data: `gold.campaign_household`, one row per household per campaign received (7,208 pairs, 889 redeemed, 12.3%).

## What the data supports

| Fact | Number | Consequence |
| --- | --- | --- |
| Households | 1,584, 4.55 campaigns each on average (up to 17) | cross-validation folds grouped by household |
| Strong single-feature rule | pre-campaign coupon share of lines: AUC 0.769 pooled, **0.783 within campaign** | the bar to beat |
| Campaign identity alone | each campaign's own redemption rate as the score: pooled AUC 0.637 | pooled AUC partly measures which campaign, not which household → judge within campaign |
| "Relevance" (pre-campaign spend share on the campaign's coupon products) | pooled AUC 0.642, within-campaign 0.551 | mostly a campaign-level signal |
| Redemption by type × prior redemption | TypeA 47.7% vs 10.5%; TypeB 17.4% vs 4.1%; TypeC 16.2% vs 3.2% | interactions → a tree model may add value |
| Drift over time (campaign start day) | redemption rate 7.5% (< 400) → 14.8% (500–586) → 14.0% (≥ 587); share with a prior redemption 6.0% → 35.4%; 65% of pairs before day 400 have no earlier campaign | history features mean less early on; time-forward test is essential |
| Who gets campaigns | targeted households: 1.30 baskets/week, $3,199 spend; never targeted: 0.35, $696 (medians) | results apply only to households the retailer already targets |
| Correlated features | coupon share ~ coupon discount 0.88; prior redemptions ~ has prior redemption 0.99; relevance share ~ amount 0.91; spend ~ products 0.92; baskets ~ baskets/week 0.90 | permute feature groups for importance |

Correction: the review first reported 16.4% / 9.3% / 44% for the drift row, from a binning error that put campaign 18 (start day 587) in the wrong period; the figures above are from the build's drift table.

## Features (all from before the campaign start)

- **Household behaviour**: frequency, basket size, spend and 12-week trend, recency, tenure, category shares, private-label share, promotion share, discount depth, coupon share and coupon discount, non-merchandise share.
- **Campaign history**: prior campaigns completed, prior redemptions, has prior redemption, prior redeemed campaigns.
- **Campaign**: type, duration, number of coupons and coupon products, relevance (spend and spend share on this campaign's coupon products).
- **Excluded**: `n_redemptions` (derived from the label), `campaign` id, absolute days (`start_day`, `end_day`, `first_day`, `last_day`), and Phase 3 segments (built from data after the campaign start).
- Missing `promo_spend_share` (40 pairs) is imputed inside the cross-validation pipeline.

## Validation

1. **Main**: 5-fold cross-validation with folds grouped by household (a household's campaigns are all in one fold) and balanced on the label. Imputation and scaling are fitted inside each training fold. Reported as mean ± SD across folds.
2. **Tuning**: inner 3-fold cross-validation, also grouped by household, inside each training fold.
3. **Time-forward check**: train on campaigns that **ended before day 587**, test on campaigns starting on or after day 587 (campaigns 18–25: 2,318 pairs, 324 redeemed). Campaigns still running on day 587 (14–17) are left out of training because their outcomes were not yet known. Campaign 18 holds 66% of the test redemptions, so results are also shown per campaign. 92% of test households also appear in training, as they would in practice.
4. **Reproducibility**: the data is held in one sorted partition (Spark's random starts and sampling depend on partitioning).

## Models (scikit-learn; changed from Spark MLlib during the build)

The plan first chose Spark MLlib. On Databricks serverless the nested cross-validation ran for over an hour without finishing: MLlib's boosted trees are built one tree at a time with several Spark jobs per tree, and every job pays a fixed scheduling cost, while the table has only 7,208 rows (the same run took about 15 minutes locally). The models now use scikit-learn 1.7.2 (preinstalled in serverless environment 6) on the driver, with Spark reading and preparing the data; the full analysis runs in under a minute locally. Validation, metrics and importance are unchanged. Spark MLlib remains in use where it fits, for the Phase 3 clustering.

- Baselines: always "no" (the noise floor); the past-coupon-user rule (has prior redemption); the best single-feature rule (pre-campaign coupon share).
- Logistic regression with an L1 penalty (penalty strength C tuned).
- Gradient-boosted trees (scikit-learn's histogram gradient boosting; depth 1–3 and 50–400 trees tuned, learning rate 0.1).
- Feature importance: the L1 path, and permutation importance on held-out folds with **correlated features permuted as a group**; within a correlated group, which feature the L1 penalty keeps is treated as arbitrary.

## Metrics

- **Primary, within campaign** (the targeting decision is who receives a given campaign): ROC AUC computed inside each campaign and averaged with weights equal to its number of redemptions; top-decile lift within campaign (redemption rate in the top 10% of scores ÷ the campaign's rate).
- Secondary: pooled ROC AUC, PR AUC (base rate 12.3% is the floor), pooled top-decile lift.
- All as mean ± SD across folds, plus the time-forward test overall and per campaign, and by campaign type.
- **Success bar**: the model is recommended only if it beats the coupon-share rule (0.783 within campaign) **fold by fold**: the paired per-fold difference, mean ± SD, must be clearly above 0. Otherwise the recommendation is to target on past coupon use.

## Targeting recommendation

- Capture curve: share of a campaign's redemptions captured by targeting the top X% of its eligible households by score, against the coupon-share rule and random targeting.
- Scope: choosing among households already eligible for a campaign; never-targeted households are outside the data the model learned from.
- Stated limitation and recommendation: redemption is not incremental sales; heavy coupon users may have bought anyway. The next step is a **randomised holdout** in future campaigns (some eligible households deliberately not sent the coupons) to measure the true effect of a campaign (an uplift model).

## Deliverables

- `src/retail_promo/analysis/response.py`: feature list, household-grouped folds, nested tuning, metrics (within-campaign AUC and lift), grouped permutation importance, capture curve.
- `notebooks/30_coupon_response.py` (Databricks source format, notebook rules); figures to `ops/figures` on Databricks and `reports/figures/` locally.
- `gold.campaign_response_scores`: out-of-fold score per household-campaign pair, for the dashboard.
- Tests: no household in both training and test of a fold; excluded columns never reach the model; within-campaign AUC, lift and capture on hand-made examples.

## Done when

- [ ] Tests pass
- [ ] Baselines, logistic regression and GBT reported with within-campaign and pooled metrics, mean ± SD, and the paired comparison with the coupon-share rule
- [ ] Time-forward results overall and per campaign
- [ ] Notebook runs on Databricks and matches the local run; scores written to gold
- [ ] Every figure saved, with summaries whose numbers match the displayed tables

## Future work

- Calibration (predicted probabilities vs observed rates), if scores are used to forecast redemption volume.
- An uplift study with randomised holdouts.
- Coupon cost data, for return on investment.
