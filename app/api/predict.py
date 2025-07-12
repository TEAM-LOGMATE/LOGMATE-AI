from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import numpy as np
import joblib
import os
import re

router = APIRouter()        # FastAPI 생성

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))     #경로 설정
MODEL_PATH = joblib.load("app/model/isolation_model.pkl")
FEATURE_PATH = joblib.load("app/model/features.pkl")
METHOD_COL_PATH = joblib.load("app/model/method_cols.pkl")

class LogLine(BaseModel):
    log: str

def parse_log_line(log: str) -> dict: # 로그 파싱 함수 
    pattern = r'(\S+) - - \[(.*?)\] "(\S+) (\S+) \S+" (\d{3}) (\d+) "(.*?)" "(.*?)"'
    match = re.match(pattern, log)
    if not match:
        raise ValueError("Invalid log format")

    ip, timestamp, method, url, status, size, referer, user_agent = match.groups()
    return {
        "method": method,
        "url": url,
        "status": int(status),
        "size": int(size),
        "referer": referer,
        "user-agent": user_agent
    }    

def extract_features(parsed: dict) -> dict:
    url = parsed["url"]
    agent = parsed["user-agent"]
    referer = parsed["referer"]

    url_length = len(url)
    url_depth = url.count('/')
    has_query_param = 1 if '?' in url else 0
    special_char_count = sum([url.count(c) for c in ['&', '=', '%', '$', '#', '@', '!', '*']])

    agent_length = len(agent)
    ref_exists = 0 if referer == "-" else 1

    # IOC 키워드 
    uri_ioc_count = sum([1 for ioc in ["login", "admin", ".exe"] if ioc in url.lower()])
    ua_ioc_count = sum([1 for ioc in ["python", "curl", "scan"] if ioc in agent.lower()])
    ioc_total_count = uri_ioc_count + ua_ioc_count

    return {
        "status": parsed["status"],
        "size": parsed["size"],
        "url_length": url_length,
        "url_depth": url_depth,
        "has_query_param": has_query_param,
        "agent_length": agent_length,
        "ref_exists": ref_exists,
        "url_special_char_count": special_char_count,
        "uri_ioc_count": uri_ioc_count,
        "ua_ioc_count": ua_ioc_count,
        "ioc_total_count": ioc_total_count,
    }
    
def scale_score(raw_score, score_min=-0.1804, score_max=0.2810):        # min-max 스케줄링 적용
    raw_score = np.clip(raw_score, score_min, score_max)
    norm_score = (raw_score - score_min) / (score_max - score_min)      # 0~1 정규화
    inverted_score = 1 - norm_score                                     # 높을수록 비정상 로그
    return round(inverted_score * 100, 2)   

@router.post("/predict_line")
def predict_line(input_data: LogLine):
    try:
        parsed = parse_log_line(input_data.log)
        features = extract_features(parsed)

        method = parsed["method", "unknown"]         # method one-hot 인코딩
        method_onehot = {
            col: 1 if col == f"method_{method}" else 0
            for col in METHOD_COL_PATH
        }
        if not any(method_onehot.values()):  # unknown 처리
            method_onehot = {col: 1 if col == "method_UNKNOWN" else 0 for col in METHOD_COL_PATH}

        # 모든 피처 병합
        full_features = {**features, **method_onehot}

        # 순서 맞춰서 vector 생성
        input_vector = [full_features.get(f, 0) for f in FEATURE_PATH]
        input_vector = np.array(input_vector).reshape(1, -1)

        # 점수 예측 (값이 작을수록 이상치)
        score = MODEL_PATH.score_samples(input_vector)[0]

        return {"score": score}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))