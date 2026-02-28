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
    termperature: float = 0.7
    max_tokens: int = 2048
    tools: list = []


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        logger.debug(f"Received chat request: {req}")

        last_user_content = ""
        for msg in reversed(req.messages):
            if isinstance(msg,dict) and msg.get("role") == "user":
                last_user_content = msg.get("content", "").lower()
                break

        # Initialize tool calls as None 
        tool_calls = None

        # Added basic trigger keywords for current predefined tool calls 
        if "calculate" in last_user_content:
            # User asked about calculation → prepare calculator_tool call
            tool_calls = [{
                "id": "call_mock_001",  # Unique ID for this tool call
                "type": "function",      # Type of tool call
                "function": {
                    "name": "calculator_tool",    # Which tool to execute
                    "arguments": '{"expression": "2 + 2"}'  # Parameters as JSON string
                }
            }]
            reply_content = "I will calculate that for you."
            
        elif "echo" in last_user_content:
            # User asked to echo → prepare echo_tool call
            tool_calls = [{
                "id": "call_mock_002",
                "type": "function",
                "function": {
                    "name": "echo_tool",
                    "arguments": '{"message": "hello"}'
                }
            }]
            reply_content = "Echoing your message."
            
        elif "search" in last_user_content:
            # User asked to search → prepare search_tool call
            tool_calls = [{
                "id": "call_mock_003",
                "type": "function",
                "function": {
                    "name": "search_tool",
                    "arguments": '{"query": "mock query"}'
                }
            }]
            reply_content = "Searching for information."
            
        else:
            # No keyword match → just return plain text response
            reply_content = "Mock response for dev"

        response = {
            "id": "mock",
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": reply_content
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        # Add tool_calls to response IF they exist
        if tool_calls:
            response["tool_calls"] = tool_calls

        logger.debug(f"Returning response: {response}")
        return response
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    logger.info("Starting mock server on http://localhost:8001")
    uvicorn.run(app, host="0.0.0.0", port=8000)
