import pandas as pd

PATH = "data/raw/items.parquet"

items = pd.read_parquet(PATH)

# Restore item_id if missing
if "item_id" not in items.columns:
    print("⚠ item_id missing — restoring from index")
    items = items.reset_index().rename(columns={"index": "item_id"})

items.to_parquet(PATH, index=False)

print("✅ items.parquet schema repaired")
print(items.head())
