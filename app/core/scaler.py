import json
import os
import numpy as np

# 스케일링 파라미터 로드
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PARAM_PATH = os.path.join(BASE_DIR, "model", "scaler_params.json")

with open(PARAM_PATH, "r") as f:
    params = json.load(f)

SCORE_MIN = params["score_min"]
SCORE_MAX = params["score_max"]

def scale_score(raw_score, score_min=SCORE_MIN, score_max=SCORE_MAX):
    raw_score = np.clip(raw_score, score_min, score_max)
    norm_score = (raw_score - score_min) / (score_max - score_min)
    inverted_score = 1 - norm_score
    return round(inverted_score * 100, 2)
