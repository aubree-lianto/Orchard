from fastapi import FastAPI
from api.routers import chat
from api.core.errors import register_error_handlers
from api.core.middleware import LoggingMiddleware
import logging

# Configure root logger for uvicorn
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Ensure api.middleware logger is configured
middleware_logger = logging.getLogger("api.middleware")
middleware_logger.setLevel(logging.INFO)
middleware_logger.propagate = True

app = FastAPI(title="Orchard API", description="LLM orchestration gateway")

# Register middleware BEFORE error handlers
app.add_middleware(LoggingMiddleware)

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


