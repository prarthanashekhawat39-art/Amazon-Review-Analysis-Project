"""
generate_dataset.py
--------------------
Generates a realistic, synthetic Amazon product-review dataset that mirrors
the column structure of the popular Kaggle "Amazon Product Reviews" datasets:

    review_id, product_id, product_name, category, rating,
    review_title, review_text, review_date, verified_purchase, helpful_votes

NOTE: No internet access is available in this environment, so a real Kaggle
file couldn't be downloaded directly. This generator produces data with the
SAME schema as the Kaggle dataset, so amazon_review_analysis.py will run
unchanged if you later drop in a real Kaggle CSV with matching column names.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

N_REVIEWS = 12000  # scale up/down as needed

CATEGORIES = [
    "Electronics", "Books", "Home & Kitchen", "Beauty & Personal Care",
    "Sports & Outdoors", "Toys & Games", "Clothing", "Shoes",
    "Grocery & Gourmet Food", "Health & Household", "Automotive",
    "Pet Supplies", "Office Products", "Baby Products",
    "Garden & Outdoor", "Musical Instruments", "Movies & TV",
    "Video Games", "Tools & Home Improvement", "Jewelry"
]

PRODUCT_ADJECTIVES = ["Pro", "Ultra", "Classic", "Deluxe", "Essential", "Max",
                      "Plus", "Lite", "Prime", "Elite", "Compact", "Smart"]
PRODUCT_NOUNS = {
    "Electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Smart Watch", "Power Bank", "USB-C Cable"],
    "Books": ["Novel", "Cookbook", "Self-Help Guide", "Biography", "Notebook Set"],
    "Home & Kitchen": ["Blender", "Air Fryer", "Cutlery Set", "Coffee Maker", "Storage Containers"],
    "Beauty & Personal Care": ["Face Serum", "Hair Dryer", "Lip Balm Set", "Electric Razor", "Body Wash"],
    "Sports & Outdoors": ["Yoga Mat", "Resistance Bands", "Water Bottle", "Camping Tent", "Hiking Backpack"],
    "Toys & Games": ["Building Blocks", "Puzzle Set", "RC Car", "Board Game", "Action Figure"],
    "Clothing": ["T-Shirt", "Jacket", "Jeans", "Hoodie", "Socks Pack"],
    "Shoes": ["Running Shoes", "Sandals", "Sneakers", "Boots", "Slippers"],
    "Grocery & Gourmet Food": ["Coffee Beans", "Protein Bars", "Green Tea", "Snack Pack", "Olive Oil"],
    "Health & Household": ["Vitamin Supplements", "First Aid Kit", "Air Purifier", "Thermometer", "Hand Sanitizer"],
    "Automotive": ["Car Vacuum", "Phone Mount", "Seat Covers", "Jump Starter", "Dash Cam"],
    "Pet Supplies": ["Dog Leash", "Cat Litter Box", "Pet Bed", "Chew Toys", "Pet Grooming Kit"],
    "Office Products": ["Desk Organizer", "Wireless Mouse", "Notebook", "Stapler", "Sticky Notes"],
    "Baby Products": ["Baby Monitor", "Diaper Bag", "Baby Carrier", "Pacifier Set", "Baby Bottles"],
    "Garden & Outdoor": ["Garden Hose", "Patio Umbrella", "Plant Pots", "Lawn Chair", "Solar Lights"],
    "Musical Instruments": ["Acoustic Guitar", "Keyboard", "Drum Sticks", "Guitar Strings", "Microphone"],
    "Movies & TV": ["Blu-ray Set", "Streaming Device", "HDMI Cable", "Projector Screen", "Remote Control"],
    "Video Games": ["Game Controller", "Gaming Headset", "Game Console Stand", "Gaming Mouse Pad", "Memory Card"],
    "Tools & Home Improvement": ["Drill Set", "Tool Box", "LED Light Bulbs", "Paint Roller", "Tape Measure"],
    "Jewelry": ["Silver Necklace", "Earrings Set", "Bracelet", "Ring", "Watch"]
}

POSITIVE_SNIPPETS = [
    "absolutely love this product", "works perfectly and exceeded my expectations",
    "great value for the price", "excellent build quality", "highly recommend this to everyone",
    "fast shipping and well packaged", "exactly as described and works great",
    "amazing quality for the price", "very happy with this purchase",
    "durable and works as expected", "best purchase I've made this year",
    "super comfortable and stylish", "customer service was excellent",
    "easy to use and set up", "impressive performance overall"
]

NEGATIVE_SNIPPETS = [
    "broke after a few days of use", "poor quality and not worth the money",
    "very disappointed with this purchase", "does not work as advertised",
    "stopped working within a week", "cheaply made and feels fragile",
    "waste of money, would not recommend", "arrived damaged and defective",
    "customer service was unhelpful", "not as described in the listing",
    "terrible experience overall", "started malfunctioning almost immediately",
    "flimsy material and poor stitching", "regret buying this product",
    "worse than expected, very frustrating"
]

NEUTRAL_SNIPPETS = [
    "it's okay, does the job but nothing special", "average quality for the price",
    "works fine but nothing extraordinary", "decent product overall",
    "meets basic expectations", "some pros and some cons",
    "not bad, not great either", "acceptable but could be improved",
    "does what it says, nothing more", "reasonable for occasional use"
]

TITLES_POS = ["Great purchase!", "Highly recommend", "Exceeded expectations", "Very satisfied", "Excellent product"]
TITLES_NEG = ["Very disappointed", "Would not recommend", "Poor quality", "Not worth it", "Broke quickly"]
TITLES_NEU = ["It's okay", "Average product", "Does the job", "Mixed feelings", "Decent for the price"]


def random_date():
    start = datetime(2023, 1, 1)
    end = datetime(2025, 12, 31)
    delta = end - start
    return start + timedelta(days=random.randint(0, delta.days))


def build_review(rating):
    """Builds review title/text with sentiment-appropriate language tied to rating."""
    if rating >= 4:
        snippets = random.sample(POSITIVE_SNIPPETS, k=random.randint(1, 3))
        title = random.choice(TITLES_POS)
    elif rating == 3:
        snippets = random.sample(NEUTRAL_SNIPPETS, k=random.randint(1, 2))
        title = random.choice(TITLES_NEU)
    else:
        snippets = random.sample(NEGATIVE_SNIPPETS, k=random.randint(1, 3))
        title = random.choice(TITLES_NEG)
    text = ". ".join(snippets).capitalize() + "."
    return title, text


def generate():
    rows = []
    # rating distribution skewed positive, like real Amazon data
    rating_choices = [1, 2, 3, 4, 5]
    rating_weights = [0.06, 0.07, 0.12, 0.28, 0.47]

    # pre-build a product catalog (product_id, name, category)
    catalog = []
    pid = 1000
    for cat in CATEGORIES:
        for noun in PRODUCT_NOUNS[cat]:
            for adj in random.sample(PRODUCT_ADJECTIVES, k=2):
                catalog.append((pid, f"{adj} {noun}", cat))
                pid += 1

    for i in range(N_REVIEWS):
        product_id, product_name, category = random.choice(catalog)
        rating = int(np.random.choice(rating_choices, p=rating_weights))
        title, text = build_review(rating)
        rows.append({
            "review_id": f"R{100000 + i}",
            "product_id": f"P{product_id}",
            "product_name": product_name,
            "category": category,
            "rating": rating,
            "review_title": title,
            "review_text": text,
            "review_date": random_date().strftime("%Y-%m-%d"),
            "verified_purchase": random.choices(["Yes", "No"], weights=[0.85, 0.15])[0],
            "helpful_votes": int(np.random.exponential(scale=3))
        })

    df = pd.DataFrame(rows)

    # inject a small amount of realistic messiness for the cleaning step to handle
    dup_sample = df.sample(frac=0.01, random_state=RANDOM_SEED)
    df = pd.concat([df, dup_sample], ignore_index=True)  # duplicates
    missing_idx = df.sample(frac=0.008, random_state=1).index
    df.loc[missing_idx, "review_text"] = np.nan  # missing values

    return df


if __name__ == "__main__":
    df = generate()
    out_path = "/home/claude/Amazon-Review-Analysis-Project/data/amazon_reviews.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.head())
