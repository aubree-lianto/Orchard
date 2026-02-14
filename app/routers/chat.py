from fastapi import APIRouter, HTTPException
from app.core.provider import get_model_client
from app.core.errors import APIError
from schemas.llm_schemas import ModelRequest, ModelResponse

router = APIRouter(prefix="/chat", tags=["chat"])

@router.post("")
# Accept request from client via Pydantic model
# Forward to ModelClient via backend
def chat(request: ModelRequest) -> ModelResponse:
    try:
        # Initialize client (abstraction handles mock vs vLLM)
        client = get_model_client()
        
        # Forward request to backend
        response = client.chat(request)
        
        return response
    except Exception as e:
        raise APIError(
            message=f"Failed to process chat request: {str(e)}",
            code="CHAT_ERROR",
            status_code=500
        )