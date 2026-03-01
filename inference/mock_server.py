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
"""

@app.get("/")
def root():
    return {"status": "running", "message": "Mock LLM server is up!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


class ChatRequest(BaseModel):
    model: str
    messages: list
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: list = []


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    """
    Mock LLM chat endpoint (OpenAI-compatible).
    
    Routes based on keywords to trigger research tools:
      - "search" / "find" → search_tool
      - "fetch" / "read" / "document" → fetch_tool
      - "retrieve" / "knowledge" → retrieval_tool
      - No match → plain text response
    """
    try:
        logger.debug(f"Received chat request: {req}")

        last_user_content = ""
        for msg in reversed(req.messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_content = msg.get("content", "").lower()
                break

        # Initialize tool calls as None (no tools by default)
        tool_calls = None

        # ===== RESEARCH TOOL ROUTING =====
        
        if "search" in last_user_content or "find" in last_user_content:
            # User wants to search for sources
            tool_calls = [{
                "id": "call_research_001",
                "type": "function",
                "function": {
                    "name": "search_tool",
                    "arguments": '{"query": "research topic", "source": "web", "limit": 5}'
                }
            }]
            reply_content = "I'll search for relevant sources on that topic."
            
        elif "fetch" in last_user_content or "read" in last_user_content or "document" in last_user_content:
            # User wants to fetch/read a document
            tool_calls = [{
                "id": "call_research_002",
                "type": "function",
                "function": {
                    "name": "fetch_tool",
                    "arguments": '{"url": "https://example.com/paper", "max_length": 8000}'
                }
            }]
            reply_content = "I'll fetch and read that document for you."
            
        elif "retrieve" in last_user_content or "knowledge" in last_user_content or "context" in last_user_content:
            # User wants to retrieve from knowledge base
            tool_calls = [{
                "id": "call_research_003",
                "type": "function",
                "function": {
                    "name": "retrieval_tool",
                    "arguments": '{"query": "research context", "collection": "general", "limit": 5}'
                }
            }]
            reply_content = "I'll retrieve relevant context from the knowledge base."
            
        else:
            # No keyword match → plain text response
            reply_content = "I understand your research request. Let me help synthesize findings on that topic."

        # Construct OpenAI-compatible response
        response = {
            "id": "mock_chatcmpl_001",
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

        # If we prepared tool calls, add them to response
        if tool_calls:
            response["tool_calls"] = tool_calls

        logger.debug(f"Returning response with tool_calls={bool(tool_calls)}")
        return response
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"error": str(e), "status": "error"}


if __name__ == "__main__":
    logger.info("Starting mock server on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)

