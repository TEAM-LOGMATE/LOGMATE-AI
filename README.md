# 🧩 LogMate AI Server - Analyzer
![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Build](https://img.shields.io/badge/build-FastAPI-success.svg)
> 로그 이상탐지와 IOC 매칭을 수행하는 **AI 분석 서버**

**AI Server**는 LogMate 시스템의 **AI 분석 컴포넌트**로, 수집된 웹 로그를 기반으로 **이상탐지(모델 추론)**, **IOC 키워드 매칭**, **스코어링(0~100)** 을 수행하며 분석 결과를 API 서버로 전송합니다.

---

## 🧰 주요 기능

### 모델/설정 기능
| 기능 분류 | 기능 설명 |
|----------|-----------|
| 모델 로딩 | - `isolation_model.pkl` 모델 로딩 (joblib)<br>- `features.pkl` / `method_cols.pkl` 기반 입력 벡터 정렬 |
| 스코어링 | - `scaler_params.json`(score_min, score_max) 기반 정규화(0~100, 높을수록 비정상) |
| IOC 로딩 | - `ioc_keywords.pkl` 로 URL/UA IOC 키워드 세트 로딩 |

### 분석/전송 기능
| 기능 분류 | 기능 설명 |
|----------|-----------|
| 로그 분석 | - 배열 형태의 로그 수신(`/receive_logs`)<br>- 특징 생성(길이/깊이/특수문자/IOC 카운트/메서드 원핫 등) |
| 결과 전송 | - 분석 결과를 `API_SERVER_URL` 로 POST 전송(실패 시 경고 로그)<br>- 상태 확인 엔드포인트(`/`) 제공 |

---

## 📂 클래스 다이어그램
(추가 예정)

---

## 📂 패키지 구조
```
/app/
│
├── main.py                  # FastAPI 앱 진입점
│
├── api/                     # 엔드포인트 라우터
│   └── predict.py           # /receive_logs (로그 수신/분석/전송)
│
├── core/                    # AI 핵심 로직
│   ├── model.py             # 모델 로더 (joblib)
│   ├── scaler.py            # 스코어 스케일링(0~100)
│   └── ioc.py               # IOC 키워드 로딩
│
├── model/                   # 모델/피처/파라미터 파일
│   ├── isolation_model.pkl
│   ├── features.pkl
│   ├── method_cols.pkl
│   ├── ioc_keywords.pkl
│   └── scaler_params.json
│
├── notebook/                # 실험/학습 노트북
│   └── IF_model.ipynb
│
└── utils/                   # 유틸리티
    └── parser.py            # 액세스 로그 파서(정규식)
```


---

## 📦 설치 및 실행

### 1. 요구 사항
- Python 3.9+
- 운영 체제: Linux/macOS/Windows
- Docker (Docker 로 실행 시)

### 2. 실행 방법

#### ✅ 로컬 실행
```bash
git clone https://github.com/your-org/logmate-ai-server.git
cd logmate-ai-server

python -m venv venv
# Windows: venv\Scripts\activate
source venv/bin/activate

pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 9000
```
---

#### ✅ Docker로 실행
```bash
docker build -t logmate-ai-server .
docker run -p 9000:9000 logmate-ai-server
```

### 📄 오픈소스 라이선스
본 프로젝트는 아래의 오픈소스 라이브러리를 사용합니다.
- FastAPI - MIT License
- Uvicorn - BSD-3-Clause License
- Scikit-learn - BSD-3-Clause License
- NumPy - BSD-3-Clause License
- Joblib - BSD-3-Clause License
- Requests - Apache License 2.0
각 라이브러리는 해당 라이선스에 따라 사용됩니다.

---

### 📄 라이선스
본 프로젝트는 **Apache License 2.0** 을 따릅니다. 자세한 내용은 [LICENSE](./LICENSE)를 참고하세요.  

---

### 🙏 기여 가이드
- PR: [.github/pull_request_template.md](.github/pull_request_template.md)  
- Issue: [.github/ISSUE_TEMPLATE/issue_report.md](.github/ISSUE_TEMPLATE/issue_report.md)

---
### 📲 연락처
email: ey8968@naver.com

---

