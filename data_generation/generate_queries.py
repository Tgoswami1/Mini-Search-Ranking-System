import pandas as pd
import random

items = pd.read_parquet("data/raw/items.parquet")

queries = []

for _ in range(100_000):
    row = items.sample(1).iloc[0]
    q = random.choice([
        row["brand"],
        row["category"],
        row["title"].split()[0],
        f"{row['brand']} {row['category']}"
    ])
    queries.append({"query": q})

pd.DataFrame(queries).to_parquet("data/raw/queries.parquet", index=False)
print("✅ Queries generated")
