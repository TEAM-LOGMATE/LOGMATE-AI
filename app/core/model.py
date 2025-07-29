import joblib
import os

def load_model(filename: str):
    full_path = os.path.join("app", "model", filename)
    return joblib.load(full_path)
