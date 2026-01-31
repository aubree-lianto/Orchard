"""
Mock LLM Inference Server

Because my laptop is absolute BOOTY CHEEKS and cannot run vllm, this file provides a fake LLM server that mimics the OpenAPI format.
Use for local development on Windows :skull:
"""
from fastapi import FastAPI
from pydantic import BaseModel 


""" App Initialization """
app = FastAPI()


"""
Health and Status Endpoints

Endpoints let you verify if server is running or not
"""
# So does Orchard just fetch from these two endpoints..? and it says ok yay it works!
@app.get("/")
def root():
    return {"status": "running", "message": "Mock LLM server is up!"}


@app.get("/health")
def health():
    return {"status": "healthy"}

# We import pydantic for strict schema validation
# OpenAI API expects specific field names and types
# Using Pydantic ensures that incoming requests match expected format
class ChatRequest(BaseModel):
    model: str
    messages: list


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    return {
        "id": "mock",
        "object": "chat.completion",
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "Mock response for dev"
            }
        }]
    }
