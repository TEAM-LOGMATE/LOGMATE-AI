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
    special_char_count = len(re.findall(r"[^\w/]", str(url)))


    agent_length = len(agent)
    ref_exists = 0 if referer == "-" else 1

    # IOC 키워드 분리
    IOC_PATTERNS = joblib.load(os.path.join("app", "model", "ioc_keywords.pkl"))
    url_ioc_keywords = IOC_PATTERNS.get("url", [])
    ua_ioc_keywords = IOC_PATTERNS.get("user_agent", [])


    def count_ioc(text: str, patterns=IOC_PATTERNS) -> int:
        text = str(text).lower()
        return sum(1 for pattern in patterns if pattern in text)

    uri_ioc_count = count_ioc(url, url_ioc_keywords)
    ua_ioc_count = count_ioc(agent, ua_ioc_keywords)
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

        print("parsed : ", parsed) ## 잠시 테스트용
        print("features", features)

        method = parsed.get("method", "UNKNOWN")
        if f"method_{method}" not in METHOD_COL_PATH:
            method = "UNKNOWN"

        method_onehot = {
            col: 1 if col == f"method_{method}" else 0
            for col in METHOD_COL_PATH
        }


        # 모든 피처 병합
        full_features = {**features, **method_onehot}

        # 순서 맞춰서 vector 생성
        input_vector = [full_features.get(f, 0) for f in FEATURE_PATH]
        input_vector = np.array(input_vector).reshape(1, -1)

        print("method_onehot", method_onehot)## 잠시 테스트용
        print("input_vector", input_vector)
        print("full_features", full_features)
        print("FEATURE_PATH:", FEATURE_PATH) 
        print("Missing features:", [f for f in FEATURE_PATH if f not in full_features])

        # 점수 예측 (값이 클수록 이상치)
        raw_score = MODEL_PATH.decision_function(input_vector)[0]
        scaled_score = scale_score(raw_score)
        return {"raw_score": round(raw_score, 5), "score": scaled_score}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))