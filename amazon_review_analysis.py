"""
amazon_review_analysis.py
--------------------------
End-to-end analysis of Amazon product reviews.

Pipeline:
    1. Data Cleaning & Preprocessing
    2. Sentiment Analysis (custom lexicon-based scorer)
    3. Exploratory Data Analysis
    4. Data Visualization (15+ charts saved to /charts)

NOTE ON SENTIMENT ANALYSIS:
This environment has no internet access, so third-party NLP libraries like
TextBlob could not be installed. Instead, a lexicon-based sentiment scorer
is implemented from scratch (a simplified VADER/TextBlob-style approach):
each review is scored by counting positive vs. negative words from curated
word lists, normalized to a polarity score in [-1, 1]. This is a drop-in
replacement — if you have internet access, swap `analyze_sentiment()` for
`TextBlob(text).sentiment.polarity` with no other changes required.
"""

import re
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 110

DATA_PATH = "/home/claude/Amazon-Review-Analysis-Project/data/amazon_reviews.csv"
CLEAN_PATH = "/home/claude/Amazon-Review-Analysis-Project/data/amazon_reviews_cleaned.csv"
CHARTS_DIR = "/home/claude/Amazon-Review-Analysis-Project/charts"

# --------------------------------------------------------------------------
# 1. LEXICON-BASED SENTIMENT ANALYZER
# --------------------------------------------------------------------------
POSITIVE_WORDS = set("""
love loved loves great excellent amazing awesome fantastic perfect perfectly
happy satisfied recommend recommended good nice wonderful best value durable
comfortable easy impressive quality fast reliable smooth affordable beautiful
sturdy convenient efficient exceeded exceeds superb outstanding pleased
delighted worth reasonable stylish sleek helpful responsive premium solid
""".split())

NEGATIVE_WORDS = set("""
broke broken poor disappointed disappointing bad terrible worst waste
defective damaged cheap cheaply flimsy fragile malfunctioning malfunction
stopped unhelpful frustrating frustrated regret worse horrible awful
useless unreliable slow uncomfortable difficult overpriced faulty misleading
scam junk garbage disappoint issue issues problem problems complaint
""".split())

NEGATIONS = {"not", "no", "never", "n't", "without", "hardly"}


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def analyze_sentiment(text):
    """Returns (polarity_score, label). Polarity in [-1, 1]."""
    text = clean_text(text)
    tokens = text.split()
    if not tokens:
        return 0.0, "Neutral"

    score = 0
    negate = False
    for tok in tokens:
        if tok in NEGATIONS:
            negate = True
            continue
        if tok in POSITIVE_WORDS:
            score += -1 if negate else 1
        elif tok in NEGATIVE_WORDS:
            score += 1 if negate else -1
        negate = False

    polarity = score / max(len(tokens), 1) * 5  # scale to roughly [-1, 1]
    polarity = max(-1.0, min(1.0, polarity))

    if polarity > 0.05:
        label = "Positive"
    elif polarity < -0.05:
        label = "Negative"
    else:
        label = "Neutral"
    return round(polarity, 3), label


# --------------------------------------------------------------------------
# 2. LOAD & CLEAN DATA
# --------------------------------------------------------------------------
def load_and_clean():
    df = pd.read_csv(DATA_PATH)
    before = len(df)

    df = df.drop_duplicates(subset=["review_id"])
    df["review_text"] = df["review_text"].fillna("")
    df["review_title"] = df["review_title"].fillna("")
    df["helpful_votes"] = df["helpful_votes"].fillna(0).astype(int)
    df["review_date"] = pd.to_datetime(df["review_date"])
    df["review_text_clean"] = df["review_text"].apply(clean_text)
    df["review_length"] = df["review_text_clean"].apply(lambda t: len(t.split()))
    df = df[df["rating"].between(1, 5)]

    after = len(df)
    print(f"Cleaning: {before} -> {after} rows ({before - after} removed)")
    return df


# --------------------------------------------------------------------------
# 3. WORD-FREQUENCY "WORD CLOUD" (no external wordcloud lib -> built from scratch)
# --------------------------------------------------------------------------
STOPWORDS = set("""
the a an and is it this that to of for in on with as was were are be been
i my me we our you your it's its very so just but not no product
""".split())


def top_words(texts, n=25):
    words = []
    for t in texts:
        for w in t.split():
            if w not in STOPWORDS and len(w) > 2:
                words.append(w)
    return Counter(words).most_common(n)


def plot_pseudo_wordcloud(word_freq, title, save_path, cmap):
    """A simple from-scratch word-cloud-style plot (no `wordcloud` package needed)."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis("off")
    max_freq = max(f for _, f in word_freq)
    rng = np.random.default_rng(7)
    colors = plt.get_cmap(cmap)(np.linspace(0.3, 0.9, len(word_freq)))
    placed = []

    for i, (word, freq) in enumerate(word_freq):
        size = 14 + (freq / max_freq) * 46
        for _ in range(60):
            x, y = rng.uniform(0.05, 0.95), rng.uniform(0.1, 0.9)
            if all(abs(x - px) > 0.12 or abs(y - py) > 0.09 for px, py in placed):
                placed.append((x, y))
                break
        ax.text(x, y, word, fontsize=size, color=colors[i], ha="center", va="center",
                 fontweight="bold", alpha=0.9, transform=ax.transAxes)
    ax.set_title(title, fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


# --------------------------------------------------------------------------
# 4. MAIN
# --------------------------------------------------------------------------
def main():
    df = load_and_clean()

    # sentiment analysis
    sentiments = df["review_text"].apply(analyze_sentiment)
    df["sentiment_score"] = sentiments.apply(lambda x: x[0])
    df["sentiment_label"] = sentiments.apply(lambda x: x[1])

    df.to_csv(CLEAN_PATH, index=False)
    print(f"Cleaned + scored dataset saved -> {CLEAN_PATH}")

    charts = 0

    # ---- Chart 1: Rating distribution ----
    plt.figure(figsize=(8, 5))
    sns.countplot(data=df, x="rating", hue="rating", palette="YlOrBr", legend=False)
    plt.title("Overall Rating Distribution")
    plt.xlabel("Rating"); plt.ylabel("Number of Reviews")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/01_rating_distribution.png"); plt.close(); charts += 1

    # ---- Chart 2: Sentiment distribution pie ----
    plt.figure(figsize=(7, 7))
    sentiment_counts = df["sentiment_label"].value_counts()
    colors_map = {"Positive": "#4CAF50", "Neutral": "#FFC107", "Negative": "#F44336"}
    plt.pie(sentiment_counts.values, labels=sentiment_counts.index, autopct="%1.1f%%",
            colors=[colors_map[k] for k in sentiment_counts.index], startangle=90)
    plt.title("Sentiment Distribution")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/02_sentiment_distribution.png"); plt.close(); charts += 1

    # ---- Chart 3: Avg rating by category ----
    cat_rating = df.groupby("category")["rating"].mean().sort_values()
    plt.figure(figsize=(9, 8))
    sns.barplot(x=cat_rating.values, y=cat_rating.index, hue=cat_rating.index,
                palette="RdYlGn", legend=False)
    plt.title("Average Rating by Category")
    plt.xlabel("Average Rating"); plt.ylabel("Category")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/03_avg_rating_by_category.png"); plt.close(); charts += 1

    # ---- Chart 4: Sentiment by category (stacked bar) ----
    sent_cat = pd.crosstab(df["category"], df["sentiment_label"], normalize="index") * 100
    sent_cat = sent_cat[["Positive", "Neutral", "Negative"]]
    sent_cat.plot(kind="barh", stacked=True, figsize=(9, 8),
                  color=["#4CAF50", "#FFC107", "#F44336"])
    plt.title("Sentiment Composition by Category (%)")
    plt.xlabel("Percentage"); plt.ylabel("Category")
    plt.legend(title="Sentiment", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/04_sentiment_by_category.png"); plt.close(); charts += 1

    # ---- Chart 5: Review length distribution ----
    plt.figure(figsize=(8, 5))
    sns.histplot(df["review_length"], bins=30, kde=True, color="#3F51B5")
    plt.title("Distribution of Review Length (words)")
    plt.xlabel("Number of Words"); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/05_review_length_distribution.png"); plt.close(); charts += 1

    # ---- Chart 6: Review length vs rating boxplot ----
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="rating", y="review_length", hue="rating", palette="coolwarm", legend=False)
    plt.title("Review Length vs Rating")
    plt.xlabel("Rating"); plt.ylabel("Review Length (words)")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/06_review_length_vs_rating.png"); plt.close(); charts += 1

    # ---- Chart 7: Monthly average rating trend ----
    monthly = df.set_index("review_date").resample("ME")["rating"].mean()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly.index, monthly.values, marker="o", color="#009688")
    plt.title("Average Rating Trend Over Time")
    plt.xlabel("Month"); plt.ylabel("Average Rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/07_monthly_rating_trend.png"); plt.close(); charts += 1

    # ---- Chart 8: Monthly review volume trend ----
    monthly_count = df.set_index("review_date").resample("ME").size()
    plt.figure(figsize=(10, 5))
    plt.plot(monthly_count.index, monthly_count.values, marker="o", color="#FF5722")
    plt.title("Review Volume Trend Over Time")
    plt.xlabel("Month"); plt.ylabel("Number of Reviews")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/08_monthly_review_volume.png"); plt.close(); charts += 1

    # ---- Chart 9: Correlation heatmap ----
    corr_df = df[["rating", "sentiment_score", "review_length", "helpful_votes"]].corr()
    plt.figure(figsize=(7, 6))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/09_correlation_heatmap.png"); plt.close(); charts += 1

    # ---- Chart 10: Top 10 categories by review count ----
    top_cats = df["category"].value_counts().head(10)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=top_cats.values, y=top_cats.index, hue=top_cats.index,
                palette="Blues_r", legend=False)
    plt.title("Top 10 Categories by Review Volume")
    plt.xlabel("Number of Reviews"); plt.ylabel("Category")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/10_top_categories_by_volume.png"); plt.close(); charts += 1

    # ---- Chart 11: Verified vs non-verified avg rating ----
    verified_avg = df.groupby("verified_purchase")["rating"].mean()
    plt.figure(figsize=(6, 5))
    sns.barplot(x=verified_avg.index, y=verified_avg.values, hue=verified_avg.index,
                palette="Set2", legend=False)
    plt.title("Average Rating: Verified vs Non-Verified Purchases")
    plt.xlabel("Verified Purchase"); plt.ylabel("Average Rating")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/11_verified_vs_nonverified.png"); plt.close(); charts += 1

    # ---- Chart 12: Helpful votes vs rating scatter ----
    plt.figure(figsize=(8, 5))
    sns.stripplot(data=df.sample(min(1500, len(df)), random_state=1), x="rating", y="helpful_votes",
                  hue="rating", palette="viridis", alpha=0.5, legend=False)
    plt.title("Helpful Votes vs Rating")
    plt.xlabel("Rating"); plt.ylabel("Helpful Votes")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/12_helpful_votes_vs_rating.png"); plt.close(); charts += 1

    # ---- Chart 13: Sentiment score distribution (KDE) ----
    plt.figure(figsize=(8, 5))
    sns.kdeplot(df["sentiment_score"], fill=True, color="#673AB7")
    plt.title("Sentiment Score Distribution")
    plt.xlabel("Sentiment Polarity Score"); plt.ylabel("Density")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/13_sentiment_score_distribution.png"); plt.close(); charts += 1

    # ---- Chart 14 & 15: Pseudo word clouds for positive / negative reviews ----
    pos_texts = df.loc[df["sentiment_label"] == "Positive", "review_text_clean"]
    neg_texts = df.loc[df["sentiment_label"] == "Negative", "review_text_clean"]
    plot_pseudo_wordcloud(top_words(pos_texts, 25), "Most Frequent Words — Positive Reviews",
                          f"{CHARTS_DIR}/14_wordcloud_positive.png", "Greens")
    charts += 1
    plot_pseudo_wordcloud(top_words(neg_texts, 25), "Most Frequent Words — Negative Reviews",
                          f"{CHARTS_DIR}/15_wordcloud_negative.png", "Reds")
    charts += 1

    # ---- Chart 16: Category x Rating heatmap ----
    pivot = pd.crosstab(df["category"], df["rating"])
    plt.figure(figsize=(9, 9))
    sns.heatmap(pivot, cmap="YlGnBu", annot=False)
    plt.title("Review Count Heatmap: Category vs Rating")
    plt.xlabel("Rating"); plt.ylabel("Category")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/16_category_rating_heatmap.png"); plt.close(); charts += 1

    # ---- Chart 17: Top 10 best-rated products (min 5 reviews) ----
    prod_stats = df.groupby("product_name")["rating"].agg(["mean", "count"])
    prod_stats = prod_stats[prod_stats["count"] >= 5]
    best = prod_stats.sort_values("mean", ascending=False).head(10)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=best["mean"], y=best.index, hue=best.index, palette="Greens_r", legend=False)
    plt.title("Top 10 Best-Rated Products (min. 5 reviews)")
    plt.xlabel("Average Rating"); plt.ylabel("Product")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/17_top10_best_products.png"); plt.close(); charts += 1

    # ---- Chart 18: Bottom 10 worst-rated products (min 5 reviews) ----
    worst = prod_stats.sort_values("mean", ascending=True).head(10)
    plt.figure(figsize=(9, 6))
    sns.barplot(x=worst["mean"], y=worst.index, hue=worst.index, palette="Reds_r", legend=False)
    plt.title("Bottom 10 Worst-Rated Products (min. 5 reviews)")
    plt.xlabel("Average Rating"); plt.ylabel("Product")
    plt.tight_layout(); plt.savefig(f"{CHARTS_DIR}/18_bottom10_worst_products.png"); plt.close(); charts += 1

    print(f"\n Generated {charts} charts in {CHARTS_DIR}")

    # ---------------------------------------------------------------
    # Key results summary (printed + used to build insights.md)
    # ---------------------------------------------------------------
    sentiment_pct = (df["sentiment_label"].value_counts(normalize=True) * 100).round(1)
    best_cat = cat_rating.idxmax()
    worst_cat = cat_rating.idxmin()

    summary = {
        "total_reviews": len(df),
        "num_categories": df["category"].nunique(),
        "avg_rating": round(df["rating"].mean(), 2),
        "positive_pct": sentiment_pct.get("Positive", 0),
        "negative_pct": sentiment_pct.get("Negative", 0),
        "neutral_pct": sentiment_pct.get("Neutral", 0),
        "best_category": best_cat,
        "best_category_avg": round(cat_rating.max(), 2),
        "worst_category": worst_cat,
        "worst_category_avg": round(cat_rating.min(), 2),
        "verified_avg_rating": round(verified_avg.get("Yes", 0), 2),
        "nonverified_avg_rating": round(verified_avg.get("No", 0), 2),
    }
    print("\nSUMMARY:", summary)
    return summary


if __name__ == "__main__":
    main()
