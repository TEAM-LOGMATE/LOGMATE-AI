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
import logging

router = APIRouter()

# 로깅 설정 추가
logger = logging.getLogger("uvicorn.access")

# 전역 메모리 저장소 (필요 없다면 삭제 가능)
log_storage = []
status_counter = Counter()
counter_lock = Lock()

# 모델 & 피처 로드 (서버 시작 시 1회)
MODEL_PATH = load_model("isolation_model.pkl")
FEATURE_PATH = load_model("features.pkl")
METHOD_COL_PATH = load_model("method_cols.pkl")


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


def compute_score_for_log(log: dict) -> float:
    parsed = {
        "method": log.get("method"),
        "url": log.get("url"),
        "status": log.get("statusCode"),
        "size": log.get("bytesSent"),
        "referer": log.get("referer"),
        "user_agent": log.get("userAgent"),
    }

    # 필수값 검증
    for k in ["method", "url", "status", "size", "referer", "user_agent"]:
        if parsed.get(k) is None:
            raise ValueError(f"Missing required field: {k}")

    features = extract_features(parsed)

    method = parsed.get("method", "UNKNOWN")
    if f"method_{method}" not in METHOD_COL_PATH:
        method = "UNKNOWN"
    method_onehot = {col: int(col == f"method_{method}") for col in METHOD_COL_PATH}

    full_features = {**features, **method_onehot}
    input_vector = np.array([full_features.get(f, 0) for f in FEATURE_PATH]).reshape(1, -1)

    raw_score = MODEL_PATH.decision_function(input_vector)[0]
    return float(scale_score(raw_score))


@router.post("/receive_logs")
async def score(request: Request):
    try:
        # 바디 수신 (+gzip 지원)
        encoding = request.headers.get("Content-Encoding", "").lower()
        raw_body = await request.body()
        if encoding == "gzip":
            raw_body = gzip.decompress(raw_body)

        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")

        # 단일 로그만 허용
        if isinstance(payload, list):
            if len(payload) != 1:
                raise HTTPException(status_code=400, detail="Expected a single log object, not an array")
            log = payload[0]
        elif isinstance(payload, dict):
            log = payload
        else:
            raise HTTPException(status_code=400, detail="Expected a single log object")

        #logger 사용
        logger.info(f"[STREAM] Received log: {json.dumps(log, ensure_ascii=False)[:500]}")

        # 점수 계산
        score_value = compute_score_for_log(log)

        logger.info(f"[SCORE] Calculated: {score_value:.4f}")

        # 내부 집계/저장 (원하면 제거 가능)
        with counter_lock:
            status = log.get("statusCode")
            if status is not None:
                status_counter[status] += 1
        log_storage.append(log)

        # 클라이언트에 점수 응답만 반환
        return {"score": score_value}

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"[ERROR] {str(e)}")  # ✅ 예외도 로깅
        raise HTTPException(status_code=500, detail=str(e))
