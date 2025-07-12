from fastapi import FastAPI
from app.api import predict

app = FastAPI()
app.include_router(predict.router)

@app.get("/")
def read_root():
    return {"message": "test"}