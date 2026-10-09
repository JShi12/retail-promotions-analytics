# Data: dunnhumby "The Complete Journey"

Household-level grocery transactions for 2,500 frequent shoppers of one US retailer over about
two years, plus demographics for a subset and direct-marketing (coupon campaign) history.

- Source: <https://www.dunnhumby.com/source-files/> (`dunnhumby_The-Complete-Journey.zip`, 134,716,463 bytes,
  SHA-256 `5e0a3d72fe8562fe0ab995f70fb58b74359e8ec4bbccd1521e2b137da0558f9a`, downloaded 2026-10-09).
- Unzipped: 8 CSVs (~848 MB) plus a PDF user guide. Raw files live in `data/raw/` (gitignored).
- **Terms.** The dunnhumby site T&Cs permit use "solely for research, personal or non-commercial
  purposes" and prohibit copying/storing/transmitting material "without the prior written
  permission of dunnhumby". They do not mention Source Files specifically. This repo therefore
  contains no raw or row-level data; tests use synthetic rows; published outputs are aggregates.
  If the repo is used commercially, ask sourcefiles@dunnhumby.com first.
- There are **no calendar dates**. `DAY` is a day index (1–711) and `WEEK_NO` a week index (1–102).
  Seasonality and holidays cannot be dated; use week effects.

## Entity map

```
hh_demographic (801 hh) ─┐
campaign_table ──────────┼── household_key ── transaction_data ── product_id ── product
 (hh × campaign, 7,208)  │        (2,500 hh)        │                         │
campaign_desc ── campaign ┘                         store_id,week_no,product_id ┘
coupon ── campaign, product_id        causal_data (product × store × week promo flags)
coupon_redempt ── household_key, campaign, coupon_upc
```

## Files at a glance

| File | Rows | Grain / key | Notes |
|---|---:|---|---|
| `transaction_data.csv` | 2,595,732 | one receipt line; no unique key column | 2,500 hh, 276,484 baskets, 582 stores, 92,339 products, days 1–711 |
| `causal_data.csv` | 36,786,524 | (product_id, store_id, week_no) **should** be unique | 115 stores, 68,377 products, weeks 9–101; 15,245 keys duplicated with conflicting values |
| `product.csv` | 92,353 | product_id (unique) | all transaction/causal/coupon products exist here |
| `hh_demographic.csv` | 801 | household_key (unique) | only 801 of 2,500 households (32%) |
| `campaign_desc.csv` | 30 | campaign (unique) | start/end day, type A/B/C |
| `campaign_table.csv` | 7,208 | (household_key, campaign) unique | 1,584 households received at least one campaign |
| `coupon.csv` | 124,548 | (coupon_upc, product_id, campaign) | 5,164 exact duplicate rows |
| `coupon_redempt.csv` | 2,318 | (household_key, day, coupon_upc, campaign) | 434 households |

Column names are inconsistent in case (`household_key` vs `PRODUCT_ID`, `display`/`mailer` lowercase).
Bronze lower-cases and snake-cases them.

## Column dictionary

Types are what silver should enforce. All raw fields are read as strings in bronze.

### transaction_data
| Column | Type | Meaning / observations |
|---|---|---|
| household_key | int | household, 1–2500 |
| basket_id | bigint | purchase occasion; each basket has exactly one household, day and store |
| day | int | 1–711 |
| product_id | int | joins to product |
| quantity | int | 0–89,638. Median 2, p99 = 10, p99.9 = 16,916 (see quality notes) |
| sales_value | decimal | dollars received by the retailer **after** loyalty and coupon-match discounts, not the customer's shelf price |
| store_id | int | 582 stores |
| retail_disc | decimal | loyalty-card discount, ≤ 0 (negative = discount); 36 lines are > 0 |
| trans_time | int | HHMM, 0–2359, always valid |
| week_no | int | 1–102; **`week_no = floor((day + 1) / 7) + 1`** holds for every row. Week 1 = days 1–5, week 102 = days 706–711 (partial) |
| coupon_disc | decimal | manufacturer-coupon discount, ≤ 0 (36,422 lines non-zero) |
| coupon_match_disc | decimal | retailer match of a manufacturer coupon, ≤ 0 (17,449 non-zero) |

Price formulas (user guide): loyalty price = (sales_value − (retail_disc + coupon_match_disc)) / quantity.
That is the regular shelf price the customer would see. Price is only observable when a unit was bought.

### causal_data (the promotion fields)
| Column | Type | Meaning |
|---|---|---|
| product_id | int | |
| store_id | int | 115 stores (cover 98.6% of transaction lines) |
| week_no | int | 9–101 only |
| display | string | `0` none, `1` store front, `2` store rear, `3` front end cap, `4` mid-aisle end cap, `5` rear end cap, `6` side-aisle end cap, `7` in-aisle, `9` secondary location, `A` in-shelf. No code `8`. |
| mailer | string | `0` not on ad, `A` interior page feature, `C` interior page line item, `D` front page feature, `F` back page feature, `H` wrap front feature, `J` wrap interior coupon, `L` wrap back feature, `P` interior page coupon, `X` free on interior page, `Z` free on front/back/wrap |

**Every row has `display != 0` or `mailer != 0`** (0 rows with both zero):

| Status | Rows |
|---|---:|
| display only (mailer = `0`) | 11,534,183 |
| mailer only (display = `0`) | 21,038,745 |
| both | 4,213,596 |

So the table lists only *promoted* product-store-weeks. **Assumption (to state in the report):
no row means not featured.** Supporting evidence: among transaction lines in scope, 40% of
lines with no causal row carry a loyalty discount versus 92% (mailer only), 62% (display only)
and 94% (both). Mailer code `A` accounts for 17.1M of the 25.2M mailer rows.

### product
| Column | Type | Meaning / observations |
|---|---|---|
| product_id | int | unique |
| manufacturer | int | code |
| department | string | 44 values; 15 rows are blank |
| brand | string | `National` 78,537 / `Private` 13,816 |
| commodity_desc | string | 277 values (15 blank) — the natural "category" level |
| sub_commodity_desc | string | 2,349 values (15 blank) |
| curr_size_of_product | string | free text (`22 LB`); blank for 30,607 rows (33%) |

### hh_demographic
Columns are anonymised by dunnhumby; the guide says values are ordinal. The guide's own
variable table for this file is garbled (names do not match descriptions), so the mapping below
comes from the data values.

| Column | Values | Notes |
|---|---|---|
| household_key | int | unique, 801 rows |
| classification_1 | `Age Group1`–`Age Group6` | ordinal; named like an age band |
| classification_2 | `X`, `Y`, `Z` | unordered |
| classification_3 | `Level1`–`Level12` | ordinal |
| classification_4 | `1`–`4`, `5+` | ordinal count |
| classification_5 | `Group1`–`Group6` | ordinal |
| homeowner_desc | Homeowner 504, Unknown 233, Renter 42, Probable Owner 11, Probable Renter 11 | |
| kid_category_desc | None/Unknown 558, 1 = 114, 2 = 60, 3+ = 69 | `None/Unknown` mixes "no kids" and "unknown" |

Do not attach real-world meanings (income, marital status, …) to the `classification_*` columns in
write-ups; they are not documented. Describe them as ordinal demographic groups.

### campaign_desc / campaign_table / coupon / coupon_redempt
| Table.Column | Type | Meaning / observations |
|---|---|---|
| campaign_desc.campaign | int | 1–30; **numbering is not chronological** (campaigns 26–30 run earliest, days 224–369) |
| campaign_desc.description | string | `TypeA` (5 campaigns), `TypeB` (19), `TypeC` (6) |
| campaign_desc.start_day / end_day | int | validity window of the coupons; days 224–719 (campaign 24 ends at day 719, after the last transaction day 711) |
| campaign_table.household_key, campaign | int | which household received which campaign; 0 duplicate pairs; `description` agrees with campaign_desc |
| coupon.coupon_upc | bigint | unique to household and campaign per the guide, but 171 UPCs appear in more than one campaign |
| coupon.campaign, product_id | int | products each coupon is redeemable on |
| coupon_redempt.household_key, day, coupon_upc, campaign | int/bigint | redemptions; every row falls inside its campaign window and matches a campaign_table pair |

Campaign semantics (user guide): a TypeA household received 16 coupons chosen from the pool by
prior purchase behaviour (which 16 is **not** in the data). In TypeB and TypeC every household
received all coupons of the campaign.

## Data-quality findings (inputs for silver and `dq_log`)

| # | Finding | Count | Proposed handling |
|---|---|---:|---|
| 1 | `causal_data` keys duplicated with conflicting display/mailer values | 15,245 keys | resolve by documented rule (e.g. keep the strongest promotion), log every collapsed row |
| 2 | Fuel / non-merchandise lines: departments `KIOSK-GAS`, `MISC SALES TRAN`, `COUP/STR & MFG`, commodity `COUPON/MISC ITEMS`. Quantity is in odd units (≈ 0.0023 per unit; all 23,103 lines with quantity ≥ 1000 are in these departments) | 30,454 lines | flag `is_non_merchandise`; exclude from price/units analysis, keep in spend totals |
| 3 | `quantity = 0` | 14,466 lines | flag; price undefined |
| 4 | `sales_value = 0` | 18,850 lines raw; 18,879 after rounding float noise (#12) | flag |
| 5 | `quantity = 0` with `sales_value > 0` | 67 raw; 38 after rounding (29 were float noise) | flag |
| 6 | `retail_disc > 0` (should be ≤ 0) | 36 raw; 10 after rounding (26 were float noise) | flag |
| 7 | `coupon` exact duplicate rows | 5,164 rows | dedupe, log |
| 8 | `coupon_redempt` same (household, day, coupon_upc) under two campaigns | 14 pairs (28 rows) | flag both rows |
| 9 | `product` blank department/commodity | 15 rows | flag, map to `UNKNOWN` |
| 10 | Products never sold | 14 of 92,353 | none needed |
| 11 | Households without demographics | 1,699 of 2,500 | keep; add `has_demographics` flag |
| 12 | Floating-point noise in money fields (e.g. `5.551115E-17` in `sales_value` / `retail_disc`) | 56 lines | round to 2 decimals (`decimal(12,2)`), log as `money_rounded` |

Checks that passed, worth encoding as DQ tests: no exact duplicate rows in transaction/causal/
campaign tables; no negative quantity or sales; `trans_time` valid; one household/day/store per
basket; all foreign keys (product, household, campaign, coupon) resolve; redemptions within
campaign windows.

## What the data supports for each phase

**Promotion and coupon fields exist: yes.**
- *Display and mailer:* `causal_data` (product × store × week). Present for 115 stores (98.5% of
  sales) and weeks 9–101 only. 22.7% of in-scope transaction lines fall in a promoted
  product-store-week.
- *Discount depth:* derivable from `retail_disc`, `coupon_disc`, `coupon_match_disc` and the price formula.
  No standalone shelf-price or promoted-price table.
- *Campaigns and coupons:* `campaign_desc`, `campaign_table`, `coupon`, `coupon_redempt`.

**Constraints that shape the analysis**
1. **Not randomised.** Retailers choose what to feature; featured items also carry more loyalty
   discount. Lift estimates are associational.
2. **Household panel, not store totals.** Units are what these 2,500 households bought. Many
   product-weeks have zero units, and shelf price is only observed when something was bought.
3. **Ramp-up.** Active households per week go from 88 (week 1) to 972 (week 13), then stay between
   1,119 and 1,428 for weeks 14–101. Weekly totals before week 14 reflect
   cohort entry, not demand. 519 households make their first purchase after day 100.
4. **Weeks 1–8 and 102** have no `causal_data`; week 102 is a partial week (6 days), week 1 is 5 days.
5. **Redemption is rare.** 889 of 7,208 household-campaign pairs (12.3%) have a redemption:
   TypeA 635/3,979 (16.0%), TypeB 210/2,655 (7.9%), TypeC 44/574 (7.7%). Every pair has transaction
   history before the campaign start. 1,548 pairs have a prior redemption by that household.
   Households appear in up to 17 campaigns, so folds must group by household.
6. **Demographics** cover 801 households (760 of the 1,584 campaign households).
7. Transactions total $8.06M in `sales_value`; loyalty discounts total $1.40M; coupon discounts $42.6K.

## Reproducing the inspection
The profiling scripts live outside the repo (scratch space). Their results are summarised above;
the `src/` DQ checks will recompute these numbers in Phase 1.
