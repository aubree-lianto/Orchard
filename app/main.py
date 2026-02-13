from fastapi import FastAPI

app = FastAPI(title="Orchard API", description="LLM orchestration gateway")


@app.get("/")
def root():
    return {"status": "ok", "message": "Welcome to Orchard API"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    return {"status": "ok"}


