# Key Insights — Amazon Review Analysis

**Dataset:** 12,000 cleaned reviews across 20 product categories
(synthetically generated with the same schema as the Kaggle Amazon Product
Reviews dataset — see note in `scripts/generate_dataset.py`).

## Sentiment Distribution
- Positive: 78.0%
- Negative: 12.0%
- Neutral: 10.0%

## Ratings
- Overall average rating: **4.04 / 5**
- Best-performing category: **Movies & TV** (avg 4.12)
- Weakest-performing category: **Baby Products** (avg 3.96)
- Verified purchases average rating: 4.03 vs Non-verified: 4.07

## Business Insights
1. Sentiment is strongly positive overall (78%), suggesting healthy customer
   satisfaction across the catalog.
2. Category-level rating gaps are relatively small (3.96–4.12), meaning no
   single category is dragging down brand perception — issues are more
   product-specific than category-specific (see top/bottom 10 product charts).
3. Negative reviews cluster around words like "broken", "defective",
   "cheap", and "stopped" — pointing to durability/QC as the top complaint
   theme (see word-cloud chart).
4. Positive reviews cluster around "great", "value", "quality", and "fast" —
   indicating price-to-quality ratio and shipping speed are key satisfaction
   drivers.
5. Review volume and average rating both show mild month-to-month
   fluctuation with no major seasonal collapse in satisfaction — worth
   monitoring over more months of real data.
6. Longer reviews tend to skew toward the extremes (1★ and 5★), while
   3★ reviews are shorter on average — customers write more when they feel
   strongly.
7. Helpful-vote counts don't correlate strongly with the rating itself,
   suggesting "helpfulness" is driven more by review detail/usefulness than
   sentiment direction.
8. Verified and non-verified purchases show nearly identical average
   ratings — little evidence of fake/incentivized-review skew in this data.
9. A small set of products drive most 1★ reviews (see bottom-10 product
   chart) — targeted QC review of just those SKUs could meaningfully lift
   the category average.
10. The top-10 best-rated products share consistent language ("durable",
    "easy to use") — good candidates for marketing copy reuse.
11. Review length distribution is right-skewed — most customers write brief
    reviews, so concise, scannable product descriptions may match customer
    communication style better than long-form copy.
12. Category × rating heatmap shows review volume is concentrated in a
    handful of categories — analytics/inventory attention should be
    weighted toward those high-volume categories first.

## Methodology Notes
- **Sentiment analysis** uses a custom lexicon-based polarity scorer (built
  from scratch) instead of TextBlob, since this environment has no internet
  access to install third-party NLP packages. The function signature is a
  drop-in replacement for TextBlob if you run this with internet access.
- **Word clouds** are rendered with a small custom matplotlib-based
  word-cloud function, since the `wordcloud` package also required internet
  to install.
- Swap `data/amazon_reviews.csv` for a real Kaggle CSV with matching column
  names to re-run this entire pipeline on real data with zero code changes.
