from pydantic import BaseModel, Field
from typing import List,Dict, Any, Optional
from datetime import datetime

class AgentState(BaseModel):
    messages: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Chat messages in OpenAI format: [{role, content}, ...]"
    )

    # Tool-related state
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Pending tool calls: [{tool_name, args, id}, ...]"
    )
    
    # Agent reasoning trace
    intermediate_steps: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Log of agent actions and observations"
    )
    
    # Execution metadata
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional context: {user_id, session_id, request_id, ...}"
    )
    
    # Tracking
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When this state was created"
    )
    
    iteration: int = Field(
        default=0,
    )

    class Config:
        arbitrary_types_allowed = True  # Allow datetime

class ToolCall(BaseModel):
    """Represents a single tool invocation."""
    tool_name: str
    arguments: Dict[str, Any]
    call_id: str = Field(default_factory=lambda: str(__import__('uuid').uuid4()))


class IntermediateStep(BaseModel):
    """Represents one step in the agent's reasoning process."""
    action: str  # "tool_call" | "llm_response" | "observation"
    detail: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)