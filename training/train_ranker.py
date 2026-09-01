import os
import sys
import pandas as pd
import lightgbm as lgb
import joblib

# --------------------------------------------------
# Fix Python path
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(PROCESSED_DIR, "train.parquet")

# --------------------------------------------------
# Load training data
# --------------------------------------------------
df = pd.read_parquet(TRAIN_PATH)

X = df[["f1", "f2", "f3", "f4"]]
y = df["label"]

print("✅ Training samples:", X.shape)

# --------------------------------------------------
# Train model
# --------------------------------------------------
model = lgb.LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    num_leaves=31
)

model.fit(X, y)

# --------------------------------------------------
# Save model
# --------------------------------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "ranker_lgbm.pkl")
joblib.dump(model, MODEL_PATH)

print("✅ Ranker trained")
print("📁 Model saved at:", MODEL_PATH)
