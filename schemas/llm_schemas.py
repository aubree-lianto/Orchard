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
    """
    We define the following as a single message in a conversation. 
    It defines role (who said it) as well as content (what was said)

    Follows OpenAI format:
      - "system": instructions for the LLM (context)
      - "user": input from human (question)
      - "assistant": response from LLM (answer)
      - "tool": result from executing a tool (feedback loop)
    
    Example (tool result):
      Message(role="tool", content="Tool calculator_tool returned: 4.0")
    """
    role: Literal["system", "user", "assistant", "tool"]
    content: str


class ModelRequest(BaseModel):
    """
    Request sent to an LLM service (mock or vLLM)

    FIELDS:
    - model (str) -> tells backend which model to use e.g "gpt-mock", "meta-llama/Llama-2-70b"
    - messages (List[Message]) -> conversation history from all roles
    - temperature (float) -> controls how "creative" the LLM is; higher number = more creativity
    - max_tokens (int) -> how many tokens LLM can generate
    - tools (Optional[List[dict]]) -> functions that LLM can invoke

    FULL EXAMPLE REQUEST:
      req = ModelRequest(
          model="gpt-mock",
          messages=[
              Message(role="system", content="You are helpful."),
              Message(role="user", content="What is 2+2?")
          ],
          temperature=0.7,
          max_tokens=100,
          tools=[
              {
                  "type": "function",
                  "function": {
                      "name": "calculator_tool",
                      "description": "Evaluates math expressions",
                      "parameters": { ... }
                  }
              }
          ]
      )
    
    """
    model: str
    messages: List[Message]
    temperature: float = 0.7
    max_tokens: int = 2048
    tools: Optional[List[dict]] = None


class ModelResponse(BaseModel):
    """
    Response from LLM service 

    FIELDS:
    - model (str) -> which model processed req
    - output_text (str) -> text response from model
    - tool_calls (Optional[List[dict]]) -> which tools were invoked
    - usage (Dict[str,int]) -> token count
    """
    model: str
    output_text: str
    tool_calls: Optional[List[dict]] = None
    usage: Dict[str, int]
