import random
import pandas as pd
import os

# ✅ ALWAYS DO THIS
os.makedirs("data/raw", exist_ok=True)

N = 50_000

categories = {
    "electronics": ["Samsung", "Apple", "Sony"],
    "fashion": ["Nike", "Adidas", "Puma"],
    "books": ["Penguin", "HarperCollins"]
}

products = {
    "electronics": ["smartphone", "headphones", "laptop"],
    "fashion": ["running shoes", "jacket", "t-shirt"],
    "books": ["novel", "biography", "history book"]
}

rows = []

for i in range(N):
    cat = random.choice(list(categories))
    brand = random.choice(categories[cat])
    prod = random.choice(products[cat])

    rows.append({
        "item_id": i,
        "title": f"{brand} {prod}",
        "category": cat,
        "brand": brand,
        "price": round(random.uniform(10, 2000), 2),
        "description": f"{brand} {prod} with high quality build"
    })

df = pd.DataFrame(rows)
df.to_parquet("data/raw/items.parquet", index=False)

print("✅ items.parquet created at data/raw/")
