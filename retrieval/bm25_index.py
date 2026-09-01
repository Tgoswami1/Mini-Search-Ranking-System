import pickle
import pandas as pd
import os
from rank_bm25 import BM25Okapi

# ✅ ENSURE FOLDER EXISTS
os.makedirs("data/processed", exist_ok=True)

# Load items
df = pd.read_parquet("data/raw/items.parquet")

# Build corpus
corpus = (df["title"] + " " + df["description"]).str.lower().str.split().tolist()

bm25 = BM25Okapi(corpus)

# Save index + lookup
with open("data/processed/bm25.pkl", "wb") as f:
    pickle.dump(bm25, f)

with open("data/processed/item_ids.pkl", "wb") as f:
    pickle.dump(df["item_id"].tolist(), f)

print("✅ BM25 index saved in data/processed/")
