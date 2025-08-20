from fastapi import APIRouter, HTTPException, Request
from collections import Counter
from threading import Lock
from app.core.model import load_model
from app.core.scaler import scale_score
from app.core.ioc import IOC_PATTERNS
import numpy as np
import re
import json
import gzip
import requests 

router = APIRouter()

# 전역 메모리 저장소
log_storage = []
status_counter = Counter()
counter_lock = Lock()

# 모델 & 피처 로드 (서버 시작 시 1회)
MODEL_PATH = load_model("isolation_model.pkl")
FEATURE_PATH = load_model("features.pkl")
METHOD_COL_PATH = load_model("method_cols.pkl")
 
API_SERVER_URL = "http://127.0.0.1:9000/api/logs" ##수정하기

def extract_features(parsed: dict) -> dict:
    url = parsed["url"]
    agent = parsed["user_agent"]
    referer = parsed["referer"]

    url_length = len(url)
    url_depth = url.count('/')
    has_query_param = int('?' in url)
    special_char_count = len(re.findall(r"[^\w/]", str(url)))

    agent_length = len(agent)
    ref_exists = int(referer != "-")

    url_ioc_keywords = IOC_PATTERNS.get("url", [])
    ua_ioc_keywords = IOC_PATTERNS.get("user_agent", [])

    def count_ioc(text: str, patterns) -> int:
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

@router.post("/receive_logs")
async def receive_logs(request: Request):
    try:
        encoding = request.headers.get("Content-Encoding", "").lower()
        raw_body = await request.body()

        if encoding == "gzip":
            raw_body = gzip.decompress(raw_body)

        try:
            logs = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        if not isinstance(logs, list):
            raise HTTPException(status_code=400, detail="Expected a JSON array of logs")

        results = []
        for log in logs:
            parsed = {
                "method": log.get("method"),
                "url": log.get("url"),
                "status": log.get("statusCode"),
                "size": log.get("bytesSent"),
                "referer": log.get("referer"),
                "user_agent": log.get("userAgent"),
            }

            features = extract_features(parsed)

            method = parsed.get("method", "UNKNOWN")
            if f"method_{method}" not in METHOD_COL_PATH:
                method = "UNKNOWN"
            method_onehot = {col: int(col == f"method_{method}") for col in METHOD_COL_PATH}

            full_features = {**features, **method_onehot}
            input_vector = np.array([full_features.get(f, 0) for f in FEATURE_PATH]).reshape(1, -1)

            raw_score = MODEL_PATH.decision_function(input_vector)[0]
            scaled_score = scale_score(raw_score)

            results.append({
                "log": log,
                "score": scaled_score
            })

            with counter_lock:
                status_counter[parsed["status"]] += 1

            log_storage.append(log)

        try:
            res = requests.post(API_SERVER_URL, json=results, timeout=5)
            res.raise_for_status()
        except requests.RequestException as e:
            print(f"[WARN] API 서버 전송 실패: {e}")

        return {"count": len(results), "results": results}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

