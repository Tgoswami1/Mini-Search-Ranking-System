import pandas as pd
from ranking.features import build_features

items = pd.read_parquet("data/raw/items.parquet").set_index("item_id")
clicks = pd.read_parquet("data/processed/clicks.parquet")

rows = []

for r in clicks.itertuples():
    item = items.loc[r.item_id]
    rows.append({
        "features": build_features(r.query, item, 1.0),
        "label": 1
    })

pd.DataFrame(rows).to_parquet("data/processed/train.parquet")
print("✅ Training data ready")
