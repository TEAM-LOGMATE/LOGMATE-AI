from fastapi import FastAPI
from app.api import predict

app = FastAPI(title="AI Log Anomaly Detection API")

app.include_router(predict.router)

@app.get("/")
def root():
    return {"message": "AI Log API is running"}
