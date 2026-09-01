import os
import numpy as np
import pandas as pd

# --------------------------------------------------
# Make project root discoverable (IMPORTANT)
# --------------------------------------------------
import sys
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from retrieval.searcher import retrieve

# --------------------------------------------------
# Paths (absolute, Windows-safe)
# --------------------------------------------------
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

os.makedirs(PROCESSED_DIR, exist_ok=True)

ITEMS_PATH = os.path.join(RAW_DIR, "items.parquet")
QUERIES_PATH = os.path.join(RAW_DIR, "queries.parquet")

# --------------------------------------------------
# Load data
# --------------------------------------------------
items = pd.read_parquet(ITEMS_PATH)
queries = pd.read_parquet(QUERIES_PATH)

print("✅ Items loaded:", items.shape)
print("✅ Queries loaded:", queries.shape)

# --------------------------------------------------
# Click simulation
# --------------------------------------------------
logs = []

NUM_QUERIES = min(10_000, len(queries))

for q in queries.sample(NUM_QUERIES, random_state=42)["query"]:
    results = retrieve(q, k=20)

    for pos, (item_id, _) in enumerate(results):
        position_bias = 1 / np.log2(pos + 2)

        # click probability with position bias
        if np.random.rand() < 0.1 * position_bias:
            logs.append({
                "query": q,
                "item_id": int(item_id),
                "position": pos
            })

# --------------------------------------------------
# Save clicks
# --------------------------------------------------
clicks_df = pd.DataFrame(logs)

OUTPUT_PATH = os.path.join(PROCESSED_DIR, "clicks.parquet")
clicks_df.to_parquet(OUTPUT_PATH, index=False)

print(f"✅ Click logs generated: {len(clicks_df)} rows")
print(f"📁 Saved at: {OUTPUT_PATH}")
