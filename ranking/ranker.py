import joblib
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "ranker_lgbm.pkl")

model = joblib.load(MODEL_PATH)

FEATURE_COLUMNS = ["f1", "f2", "f3", "f4"]

def score(features):
    """
    Predict ranking score (probability of relevance)
    """
    X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
    return model.predict_proba(X)[0][1]
