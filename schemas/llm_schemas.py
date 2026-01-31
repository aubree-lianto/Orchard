"""

This file is what we define as the interface contract -> Define the structure for how backend will communicate with LLM and vice versa
File should be imported by API endpoints, mock servers, and any LLM adapters
Structure is based on OpenAI API -> Compatibility with LangChain/LangGraph
Designed to match OpenAI /v1/chat/completions format.

"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime


"""
An example of a request that OrchardAPI can send to our MockLLM server

# create request
req = ModelRequest(model="gpt-mock", messages=[Message(role="user", content="Hello")])
# send via provider (factory returns a ModelClient)
client = get_model_client()
resp = client.chat(req)
print(resp.output_text)

"""

class Message(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ModelRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: Optional[List[dict]] = None


class ModelResponse(BaseModel):
    model: str
    output_text: str
    tool_calls: Optional[List[dict]] = None
    usage: Dict[str, int]
