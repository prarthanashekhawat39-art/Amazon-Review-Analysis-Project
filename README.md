# Amazon Product Review Analysis

## Overview
This project performs an end-to-end analysis of Amazon product reviews using Python. It applies data cleaning, sentiment analysis, and visualization techniques to uncover patterns in customer feedback and product performance across multiple categories.

## Dataset
- 12,000+ Amazon product reviews
- Spans 20+ product categories
- Includes review text, ratings, and product metadata

## Methodology
1. **Data Cleaning & Preprocessing** — Handled missing values, duplicates, and text normalization using Pandas and NumPy
2. **Sentiment Analysis** — Classified reviews as Positive, Negative, or Neutral using a custom lexicon-based NLP scorer
3. **Exploratory Data Analysis** — Analyzed rating distributions, category-wise trends, and review patterns
4. **Data Visualization** — Created 18 charts including heatmaps, word clouds, and trend graphs using Matplotlib and Seaborn

## Key Results
- **Sentiment Distribution:** 78% Positive, 12% Negative, 10% Neutral
- Identified top-performing and underperforming product categories based on rating trends
- Derived 12 actionable business insights related to customer satisfaction and product quality

## Tools & Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn (Visualization)
- Scikit-learn / SciPy

## Project Structure

## How to Run
```bash
pip install -r requirements.txt
python generate_dataset.py
python amazon_review_analysis.py
