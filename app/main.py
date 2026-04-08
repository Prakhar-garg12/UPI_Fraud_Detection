from fastapi import FastAPI

app = FastAPI(
    title="UPI Fraud Detection API",
    description="Basic health check endpoint",
    version="1.0.0"
)


@app.get("/health")
async def health():
    return {"status": "ok"}