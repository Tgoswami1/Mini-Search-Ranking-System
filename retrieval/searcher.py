import pickle
import os
import numpy as np

# -------------------------------------------------
# Load BM25 index and item ids
# -------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BM25_PATH = os.path.join(BASE_DIR, "data", "processed", "bm25.pkl")
ITEM_IDS_PATH = os.path.join(BASE_DIR, "data", "processed", "item_ids.pkl")

if not os.path.exists(BM25_PATH):
    raise FileNotFoundError("BM25 index not found. Run bm25_index.py first.")

with open(BM25_PATH, "rb") as f:
    bm25 = pickle.load(f)

with open(ITEM_IDS_PATH, "rb") as f:
    item_ids = pickle.load(f)

# -------------------------------------------------
# Retrieval function
# -------------------------------------------------

def retrieve(query: str, k: int = 100):
    """
    Retrieve top-k items using BM25

    Args:
        query (str): search query
        k (int): number of results

    Returns:
        List of tuples: (item_id, bm25_score)
    """
    if not query or not query.strip():
        return []

    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    top_idx = np.argsort(scores)[::-1][:k]

    results = [
        (int(item_ids[i]), float(scores[i]))
        for i in top_idx
        if scores[i] > 0
    ]

    return results
