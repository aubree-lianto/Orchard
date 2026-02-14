from fastapi import FastAPI
from app.routers import chat
from app.core.errors import register_error_handlers

app = FastAPI(title="Orchard API", description="LLM orchestration gateway")

register_error_handlers(app)

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


