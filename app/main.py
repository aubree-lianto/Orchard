from fastapi import FastAPI
from app.routers import chat

app = FastAPI(title="Orchard API", description="LLM orchestration gateway")

app.include_router(chat.router)

@app.get("/")
def root():
    return {"status": "ok", "message": "Welcome to Orchard API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"status": "ok"}


