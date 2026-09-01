import os
import sys
import pandas as pd
import random

# --------------------------------------------------
# Fix Python path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from ranking.features import build_features

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

ITEMS_PATH = os.path.join(RAW_DIR, "items.parquet")
CLICKS_PATH = os.path.join(PROCESSED_DIR, "clicks.parquet")

# --------------------------------------------------
# Load data
# --------------------------------------------------
items = pd.read_parquet(ITEMS_PATH).set_index("item_id")
clicks = pd.read_parquet(CLICKS_PATH)

print("✅ Items loaded:", items.shape)
print("✅ Clicks loaded:", clicks.shape)

# --------------------------------------------------
# Build training data with RANDOM NEGATIVES
# --------------------------------------------------
rows = []

NEGATIVES_PER_POSITIVE = 5
ALL_ITEM_IDS = items.index.tolist()

for r in clicks.itertuples():
    # -------- positive --------
    pos_item = items.loc[r.item_id]

    pos_features = build_features(
        query=r.query,
        item=pos_item,
        bm25_score=1.0
    )

    rows.append({
        "f1": pos_features[0],
        "f2": pos_features[1],
        "f3": pos_features[2],
        "f4": pos_features[3],
        "label": 1
    })

    # -------- negatives --------
    neg_ids = random.sample(
        ALL_ITEM_IDS,
        NEGATIVES_PER_POSITIVE
    )

    for neg_id in neg_ids:
        if neg_id == r.item_id:
            continue

        neg_item = items.loc[neg_id]
        neg_features = build_features(
            query=r.query,
            item=neg_item,
            bm25_score=0.0
        )

        rows.append({
            "f1": neg_features[0],
            "f2": neg_features[1],
            "f3": neg_features[2],
            "f4": neg_features[3],
            "label": 0
        })

train_df = pd.DataFrame(rows)

# --------------------------------------------------
# Save
# --------------------------------------------------
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "train.parquet")
train_df.to_parquet(OUTPUT_PATH, index=False)

print("✅ Training data created:", train_df.shape)
print("Label distribution:")
print(train_df["label"].value_counts())
