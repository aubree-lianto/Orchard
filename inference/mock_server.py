"""
Mock LLM Inference Server

Because my laptop is absolute BOOTY CHEEKS and cannot run vllm, this file provides a fake LLM server that mimics the OpenAPI format.
Use for local development on Windows :skull:
"""
from fastapi import FastAPI
from pydantic import BaseModel
import logging
import uvicorn

# Added proper logging and error handling
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

""" App Initialization """
app = FastAPI()

logger.info("Mock LLM Server initializing...")


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
    try:
        logger.debug(f"Received chat request: {req}")
        response = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Mock response for dev"
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }
        logger.debug(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    logger.info("Starting mock server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8001)
