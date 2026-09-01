import pandas as pd
import sys

size = int(sys.argv[1])  # e.g. 10000

items = pd.read_parquet("data/raw/items.parquet")

# Ensure item_id exists
if "item_id" not in items.columns:
    items = items.reset_index().rename(columns={"index": "item_id"})

reduced = items.sample(size, random_state=42)

# ALWAYS save with item_id as column
reduced.to_parquet("data/raw/items.parquet", index=False)

print(f"✅ Index reduced to {size} items (item_id preserved)")
