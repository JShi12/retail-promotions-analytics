# Phase 2 plan: promotion effectiveness

2026-10-09 · approved plan

Phase 2 estimates how much weekly unit sales rise in weeks when a product is in the mailer or on an in-store display, by category, with 95% confidence intervals (CIs). All estimates are associations, not causal effects: the retailer chooses what to promote and when, and the data cannot rule out that choice tracking demand the models do not see.

## What the data supports

Exploration of `gold.product_week_panel` (12,997 products bought in ≥ 26 weeks, weeks 9–101, 115 stores):

| Fact | Number | Design consequence |
| --- | --- | --- |
| Zero-sales product-weeks | 43.4% of 1,208,721 | count model that keeps zeros |
| Weekly units: mean vs variance | 2.27 vs 61.5 | heavily overdispersed; robust (clustered) standard errors |
| Mailer coverage when present | median 99% of store traffic; 6.5% of product-weeks | mailer is chain-wide, effectively on/off |
| Display coverage when present | median 3%, 75th percentile 8.9%, 95th 36%; 41% of product-weeks | display is local to a few stores |
| Discount depth | missing in all zero-sales weeks; mean 15.8% in promoted vs 7.5% in unpromoted weeks with sales | cannot enter the main model; mailer weeks carry deeper discounts |
| Mailer vs display correlation | 0.36 | promotions overlap |
| Products with both promoted and unpromoted weeks | 10,977 | within-product comparison possible |
| Raw contrast | mailer-only weeks 8.1 units vs 1.6 unpromoted (5.2×) | naive comparison grossly overstates lift |

## What is estimated, and what it is called

- **Mailer-week lift**: the % change in a product's weekly units in weeks it is in the mailer. Mailer weeks usually also carry a price cut and sometimes a display, and price cannot be controlled for in weeks without sales. So this is the lift of the whole promotion event that comes with a mailer feature, never "the effect of the mailer" alone.
- **In-store display lift**: because units add up over stores, the coefficient on the traffic-weighted display share approximates the relative lift *in the stores that have the display*. With the preliminary coefficient of about 1.9–2.0, that is roughly 3× the product's normal sales in those stores. Reported with the approximation stated.
- **Category (net) lift**: how much the whole category's units rise when part of it is promoted. Product-level lift overstates incremental sales because a promoted product partly takes sales from other products in its category; the category model nets that out.
- **Discount depth**: descriptive only (see model 3); not a price elasticity and not in the headline.

## Models

1. **Main product model.** Poisson pseudo-maximum likelihood (PPML) of weekly units on `display_share` and `mailer_share`, with product fixed effects (each product's normal sales level) and commodity × week fixed effects (each category's own seasonality and the household ramp-up). Standard errors clustered two ways, by product and by week.
    - Pooled over all products, and separately for each of the 20 selected categories.
    - The pooled lift is volume-weighted: high-selling products count more. Stated in the write-up.
2. **Category model.** Weekly category units (selected categories, weeks 9–101) on the volume-weighted share of the category in the mailer and on display, with category and week fixed effects, PPML, clustered by category. Comparing its lift with the product-level lift shows how much product lift is switching within the category.
3. **Discount depth, descriptive.** Ordinary least squares (OLS) of log units on display, mailer and discount depth, weeks with sales only, same fixed effects. Labelled descriptive: it conditions on a purchase having happened, and depth is computed from the same transactions as units.
4. **Timing checks.**
    - Lead placebo: add next week's mailer. A clear association with a promotion that has not happened yet would mean promotions are timed to demand the model does not see.
    - Post-promotion dip: add last week's mailer. A negative coefficient suggests customers stock up and buy less the following week, so the event's net gain is smaller than its in-week lift.

## Category selection

Top 20 commodities by panel sales among those with ≥ 20 products and ≥ 200 mailer product-weeks. This excludes cigarettes (0 mailer product-weeks) and carbonated water (159); the 20 selected cover 43.6% of panel sales (sales ranks 1–22). A category's display lift is reported only if it also has ≥ 1,000 display product-weeks, otherwise marked "not estimable". The plan first set 200; it was raised after beef (272 display product-weeks from 10 products) gave an unstable estimate of −10.6. At 1,000, beef, pork, deli meats and chicken have no display estimate. With 20 category CIs, about one may miss its true value by chance; stated next to the chart.

## Sensitivity checks

| Check | Main setting | Alternative |
| --- | --- | --- |
| Week window | 9–101 | 14–101 (after the household ramp-up) |
| Product threshold (`panel_min_weeks`) | 26 | 13 and 52 |
| Product lifecycle | all weeks | only weeks between a product's first and last purchase (17.9% of zero weeks fall outside) |
| Fixed effects | product + commodity × week | product + week |
| Standard errors | two-way cluster (product, week) | product only |

Preliminary pooled mailer-week lift across the variants tested so far (week window, lifecycle, fixed effects, clustering): 77.1% to 81.3%. The product-threshold variants are not tested yet.

## Scope note

Results describe the purchases of 2,500 frequent-shopper households across 115 stores, not total store sales.

## Implementation

- `src/retail_promo/analysis/promo_lift.py`: Spark builds the model tables (window, lag and lead columns, category selection, category-level aggregates), then pandas + [pyfixest](https://pyfixest.org) fits `fepois` / `feols`.
- Tests: a simulated panel with a known lift (the code must recover it within its CI), lag/lead construction, category selection rule.
- `notebooks/10_promo_effectiveness.py` (Databricks source format): goal, key terms and contents; every figure followed by a 2–4 bullet summary with numbers; figures saved to `reports/figures/`.
- Figures: raw vs fixed-effects lift (why the naive 5.2× is wrong); forest plot of mailer-week lift by category; display lift by category; product vs category lift.
- New gold table `gold.promo_lift_by_category` for the Phase 5 dashboard.
- New dependencies: `pyfixest`, `matplotlib`.
- First Databricks step: a one-cell `%pip install pyfixest` test, because Free Edition restricts outbound internet and PyPI access is unconfirmed.

## Done when

- [ ] Tests pass, including recovery of a known simulated lift
- [ ] Notebook runs on Databricks (or the documented fallback is used) and writes `gold.promo_lift_by_category`
- [ ] Every figure saved to `reports/figures/` with a summary whose numbers match the displayed tables
- [ ] Sensitivity table filled in; any estimate that moves materially is called out

## Future work

- Split mailer by placement (front/back page or wrap vs interior page) to show which slots work; needs a gold-table change.
- Promotion ROI proxy: incremental sales value per $ of loyalty discount given away (no cost or margin data, so not a true ROI).
- Better fallback if PyPI is blocked on Free Edition: run the same code locally and upload the small results table; check whether `statsmodels` is preinstalled on serverless environment 6.
