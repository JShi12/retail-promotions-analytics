# Phase 3 plan: customer segmentation

2026-10-12 · approved plan (revised after review)

Phase 3 groups households by **how they buy on price and promotion**, so a promotions team can decide who to target with which kind of promotion. Spend, frequency, basket size, category mix, coupon redemption and demographics describe the segments; they do not define them. Spark MLlib k-means on household features built from `gold.household_features` and `silver.transactions`.

## Why this design (review findings)

The first draft clustered 11 all-round shopping features. Review tests showed the result depended on which features went in, and that its promotion measure partly reflected what and where households buy:

| Finding | Number | Consequence |
| --- | --- | --- |
| Segments change with the feature set | k = 3 agreement with the draft only 0.37–0.40 (adjusted Rand index, ARI; 1 = identical) for three alternative feature sets | state the purpose first; it chooses the features |
| Promotion share partly reflects category mix | draft "deal-seeking" segment: 28.9% actual vs 24.5% expected from its category mix (+4.9 points) | adjust for category |
| Promotion share partly reflects the main store | main store explains 11.4% of variance in promotion share, 9.1% in private-label share | adjust for store |
| In-store promotion buying ≠ coupon response | draft "deal-seeking" segment redeems campaign coupons at 13.2% vs 12.9% for "big-basket" | names describe behaviour only; redemption shown as a profile |
| Demographics describe heavy shoppers | with demographics: 1.52 vs 0.57 baskets/week (×2.7), $4,212 vs $1,318 spend (×3.2) | caveat next to every demographic result |
| Low dimension inflates silhouette | 4 adjusted features: shuffled-feature baseline already scores 0.31–0.39 | judge k by the gap over that baseline, not the raw silhouette |

## Features (inputs to k-means, standardised)

Each "adjusted" measure is the household's actual value minus the value expected from **category × store** rates, i.e. what an average shopper would show buying the same categories in the same stores. Positive = more than expected.

How the expected value is built:

1. **Benchmark rates** for every category × store, from all households' purchases pooled over all weeks, e.g. promo rate(c, s) = promoted spend in category c at store s ÷ all spend in c at s; discount rate(c, s) = loyalty discount ÷ shelf value in c at s; private-label rate likewise.
2. **Leave-one-out**: when computing a household's expected value, its own purchases are removed from each category × store total, so a heavy shopper in a small category-store cell is not compared with itself.
3. **Apply to the household's basket**: expected = Σ over its purchases (spend × rate for that category and store) ÷ Σ spend (shelf value for discount depth). Example: $60 of soft drinks at a 45% promo rate and $40 of milk at 10% give an expected promotion share of (27 + 4) / 100 = 31%; an actual 42% is +11 points.

Rates are pooled over weeks, not set per week, on purpose: a household that concentrates its shopping in heavy-promotion weeks is showing deal-prone behaviour, and per-week rates would remove that.

1. **Deal-proneness index**: mean of the standardised adjusted promotion share (spend on displayed or mailer-featured products, promotion-observed lines only) and the standardised adjusted discount depth (% off shelf price). The two correlate 0.80, so they are combined rather than counted twice.
2. **Adjusted private-label share** of spend.
3. **Coupon use**: log(1 + 100 × share of lines with a manufacturer coupon); the raw share is heavily skewed (skew 4.8) and on its own creates a 115-household segment.

Households: 2,402 with ≥ 10 baskets are used to fit. The 98 with fewer are assigned to the nearest segment afterwards and flagged `low_activity`. 9 households have no promotion-observed purchases; their adjusted promotion share is set to 0 (no excess) and flagged.

## Fitting and choosing k

- Spark MLlib Pipeline: VectorAssembler → StandardScaler → KMeans. Each fit is the lowest-cost of 10 random starts (single starts disagree: ARI 0.40–0.61 across seeds in the draft).
- Candidates k = 2–6. For each k:
    - silhouette, and the silhouette of the same data with each feature shuffled independently (keeps each feature's distribution, removes the structure between them); report the gap;
    - per-segment stability: bootstrap Jaccard similarity (Hennig 2007, "Cluster-wise assessment of cluster stability", *Computational Statistics & Data Analysis* 52); a segment is stable if its mean Jaccard ≥ 0.75.
- Choose the largest k whose segments are all stable, with a clear gap over the shuffled baseline, and which read as distinct behaviours.
- Preliminary (4 adjusted features before merging promotion and discount): k = 4 has silhouette 0.434 vs 0.329 shuffled (gap +0.105) and subsample ARI ≥ 0.94; k = 3 is unstable (ARI down to 0.57). The final choice is made on the 3 features above.

## Profiles (describe, do not define)

For each segment: size and share of spend; median actual vs expected promotion share, discount depth and private-label share; frequency, basket size and spend; category mix; main-store concentration; campaign coupon redemption rate (descriptive, from `gold.campaign_household`); demographics on the 801 households that have them, with counts and the heavy-shopper caveat.

Segment names describe measured behaviour only (e.g. "buys well above average on promotion"), never motives such as "deal-seeking".

## Use downstream

`gold.household_segments` uses two years of purchases, including weeks after every campaign. **Phase 4 must not use it as a feature** unless the segments are recomputed from pre-campaign data only. It is for profiling, the dashboard and targeting discussion.

## Deliverables

- `src/retail_promo/analysis/segments.py`: adjusted-feature builder (Spark), best-of-10 MLlib fitting, shuffled baseline, bootstrap Jaccard, profiles.
- `notebooks/20_customer_segments.py` (Databricks source format): goal, key terms, contents; every figure followed by a 2–4 bullet summary with checked numbers; figures saved to `reports/figures/`.
- `gold.household_segments`: household, segment, segment name, distance to centre, `low_activity`, `promo_imputed`.
- Figures: choosing k (silhouette vs shuffled baseline, stability); segment profiles (standardised feature means); actual vs expected promotion share by segment; size vs share of spend; redemption and demographics by segment (with counts).
- Tests: adjusted features on a hand-built example (a household buying only heavily promoted categories has zero excess); synthetic data with known clusters is recovered; Jaccard stability calculation.

## Done when

- [ ] Tests pass
- [ ] k chosen by the stated rule, with the gap and per-segment Jaccard shown
- [ ] Notebook runs on Databricks and writes `gold.household_segments`; numbers match the local run
- [ ] Every figure saved to `reports/figures/` with a summary whose numbers match the displayed tables

## Future work

- Centred log-ratio transform if category shares are ever used as inputs (they sum to ~1).
- Gaussian mixture model as a comparison.
- Whether segments respond differently to the Phase 2 promotions (needs a household × product panel).
