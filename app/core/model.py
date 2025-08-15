import joblib
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def load_model(filename: str):
    full_path = os.path.join("app", "model", filename)
    return joblib.load(full_path)
