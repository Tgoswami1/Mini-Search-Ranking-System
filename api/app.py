from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
import sys
import time
from collections import defaultdict, deque
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from retrieval.searcher import retrieve
from ranking.features import build_features
from ranking.ranker import score

# --------------------------------------------------
# App
# --------------------------------------------------
app = FastAPI()

# --------------------------------------------------
# Paths
# --------------------------------------------------
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

ITEMS_PATH = os.path.join(RAW_DIR, "items.parquet")
CLICKS_LIVE_PATH = os.path.join(PROCESSED_DIR, "clicks_live.parquet")

os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------
# Load data
# --------------------------------------------------
items_df = pd.read_parquet(ITEMS_PATH).set_index("item_id")

# --------------------------------------------------
# In-memory stores
# --------------------------------------------------
query_events = deque()                 # (timestamp, query)
query_counter = defaultdict(int)

# --------------------------------------------------
# Schemas
# --------------------------------------------------
class Item(BaseModel):
    item_id: int
    title: str
    category: str
    price: float
    brand: str
    description: str


class ClickEvent(BaseModel):
    user_id: str | None
    query: str
    item_id: int
    position: int
    ts: float | None = None


# --------------------------------------------------
# Root
# --------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "Mini Search System",
        "endpoints": [
            "/search",
            "/items/bulk",
            "/feedback/click",
            "/top_queries"
        ]
    }

# --------------------------------------------------
# Search
# --------------------------------------------------
@app.get("/search")
def search(q: str, k: int = 20, user_id: str | None = None):
    candidates = retrieve(q, k=50)

    if not candidates:
        return {"items": []}

    bm25_scores = np.array([s for _, s in candidates])
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-6)

    ranked = []
    for (item_id, bm25_score), bm25_n in zip(candidates, bm25_norm):
        item = items_df.loc[item_id]
        features = build_features(q, item, bm25_score)
        ltr_score = score(features)

        final_score = 0.7 * ltr_score + 0.3 * bm25_n

        ranked.append({
            "id": int(item_id),
            "score": float(final_score)
        })

    return {"items": sorted(ranked, key=lambda x: x["score"], reverse=True)[:k]}

# --------------------------------------------------
# Bulk item ingestion
# --------------------------------------------------
@app.post("/items/bulk")
def bulk_items(items: list[Item]):
    global items_df

    new_df = pd.DataFrame([i.dict() for i in items]).set_index("item_id")
    items_df = pd.concat([items_df, new_df])
    items_df.to_parquet(ITEMS_PATH)

    return {"status": "ok", "items_added": len(new_df)}

# --------------------------------------------------
# Reindex (BM25 rebuild hook)
# --------------------------------------------------
@app.post("/reindex")
def reindex():
    # Placeholder: in real system we rebuild BM25 here
    return {"status": "ok", "message": "Reindex triggered (BM25 rebuild stub)"}

# --------------------------------------------------
# Click feedback
# --------------------------------------------------
@app.post("/feedback/click")
def click_feedback(event: ClickEvent):
    ts = event.ts or time.time()

    # Persist click
    row = {
        "user_id": event.user_id,
        "query": event.query,
        "item_id": event.item_id,
        "position": event.position,
        "ts": ts
    }

    df = pd.DataFrame([row])
    if os.path.exists(CLICKS_LIVE_PATH):
        df_existing = pd.read_parquet(CLICKS_LIVE_PATH)
        df = pd.concat([df_existing, df])

    df.to_parquet(CLICKS_LIVE_PATH, index=False)

    # Track query frequency
    query_events.append((ts, event.query))
    query_counter[event.query] += 1

    return {"status": "ok"}

# --------------------------------------------------
# Top queries (sliding window)
# --------------------------------------------------
@app.get("/top_queries")
def top_queries(window: str = "5m"):
    now = time.time()
    window_sec = int(window.replace("m", "")) * 60

    while query_events and query_events[0][0] < now - window_sec:
        _, q = query_events.popleft()
        query_counter[q] -= 1
        if query_counter[q] <= 0:
            del query_counter[q]

    top = sorted(query_counter.items(), key=lambda x: x[1], reverse=True)[:10]
    return {"top_queries": top}
