# HTTP layer that uses the agent

from fastapi import APIRouter, HTTPException
from app.core.provider import get_model_client
from app.core.errors import APIError
from schemas.llm_schemas import ModelRequest, ModelResponse
from app.core.agent_state import AgentState
from agents.research_agent import RESEARCH_GRAPH

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
# Accept request from client via Pydantic model
# Forward to ModelClient via backend
# Initially accidentally defined as an async function
def chat(request: ModelRequest) -> ModelResponse:
    try:
        # 1. convert HTTP request -> agent state
        initial_state = AgentState(
            messages = [{"role": msg.role, "content": msg.content}
            for msg in request.messages]
        )

        # 2. Run the research agent (multi-turn loop)
        final_state = RESEARCH_GRAPH.invoke(initial_state)
        
        # 3. Extract response & return
        final_message = final_state.messages[-1]["content"]
        return ModelResponse(
            model=request.model,
            output_text=final_message,
            tool_calls=None,
            usage={"total_tokens": 0}
        )
    
    # otherwise throw exception
    except Exception as e:
        raise APIError(
            message=f"Failed to process chat request: {str(e)}",
            code="CHAT_ERROR",
            status_code=500
        )