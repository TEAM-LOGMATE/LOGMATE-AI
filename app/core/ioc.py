import os
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IOC_PATH = os.path.join(BASE_DIR, "model", "ioc_keywords.pkl")

IOC_PATTERNS = joblib.load(IOC_PATH)
