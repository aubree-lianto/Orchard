from __future__ import annotations

import os
# For API keys later
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from typing import Optional

import requests

from schemas.llm_schemas import ModelRequest, ModelResponse

# implementation of an interface/contract
# by inheriting ABC from the ABC metaclass -> telling python not to let anyone instantiate this directly
class ModelClient(ABC):
    # Decorator here says -> any child class that inherits MUST write their own version of this specific function
    # Chat function
    # request:ModelRequest -> tells us request must be of type ModelRequest (Pydantic Model)
    # The arrow (->) says function will return a ModelResponse type (Pydantic Model defined in LLM_schemas)
    @abstractmethod
    def chat(self, request:ModelRequest) -> ModelResponse:
        # If a child class calls this without its own code
        # Program will crash and show an error 
        raise NotImplementedError

# me learning OOP for the first time
# Inherits the modelClient class 
class MockModelClient(ModelClient):
    # Constructor that runs at initialization 
    # base_url argument -> Just tells us which server the mock client is on -> defaults to port:8000
    # Will need to implement .env file later
    def __init__(self, base_url: Optional[str] = None, timeout: int = 10):
        self.base_url = base_url or os.getenv("MOCK_MODEL_SERVER_URL", "http://localhost:8000")
        # Timeout define to prevent application from freezing
        self.timeout = timeout
        # Just keeps session open
        self.session = requests.Session()

    # No need abstract decorator here cuz this is inheriting from ModelClient class
    # Ts function declaration fulfills the requirement
    def chat(self, request:ModelRequest) -> ModelResponse:

        # FastAPI base_url for mock server
        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"

        # Payload aka request -> apparently very standard for OpenAI API

        payload = {
            "model": request.model,
            "messages": [m.dict() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tools": request.tools,
        }

        # Push your payload/request to mock server -> Store in variable response
        response = self.session.post(url, json=payload, timeout=self.timeout)
        # Safety check in case server crashed
        response.raise_for_status()
        # Store as json dictionary
        data = response.json()

        # In LLMs, answer is usually inside a list called choices
        # These lines just parse the response
        choices = data.get("choices", [])
        content = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")

        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        tool_calls = data.get("tool_calls") or None

        # Return the parsed model response
        return ModelResponse(model=request.model, output_text=content, tool_calls=tool_calls, usage=usage)

    
# Inherits modelClient class -> modelClient for the real vLLM
class VLLMModelClient(ModelClient):

    def __init__(self, base_url: Optional[str] = None, default_model:str = None, timeout: int = 10, headers: dict = None, streaming: bool = False):
        self.base_url = base_url or os.getenv("VLLM_SERVER_URL", "http://localhost:8080")
        # Timeout define to prevent application from freezing
        self.timeout = timeout
        # Fallback model name is request.model is empty
        self.default_model = default_model
        # For api keys
        self.headers = headers or {}
        # Whether to support streaming responses
        self.streaming = streaming

    # Define the base chat functions
    def chat(self, request: ModelRequest) -> ModelResponse:
        # choose model (request overrides client default)
        model = request.model or self.default_model
        if not model:
            raise ValueError("no model specified in request or client.default_model")

        # simple sync client doesn't support streaming
        if self.streaming:
            raise NotImplementedError("streaming not supported by this VLLMModelClient stub")

        url = f"{self.base_url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [m.dict() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "tools": request.tools,
        }

        resp = requests.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        choices = data.get("choices", [])
        content = ""
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content", "")

        usage = data.get("usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
        tool_calls = data.get("tool_calls") or None

        return ModelResponse(model=model, output_text=content, tool_calls=tool_calls, usage=usage)
