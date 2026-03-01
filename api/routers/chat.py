"""
Chat Router: HTTP API endpoints for chat/research.

This is the HTTP layer—routers should be thin.

Responsibility:
  - Parse HTTP requests
  - Validate input
  - Call service layer
  - Return HTTP responses
  - Handle HTTP-level errors

Flow:
  HTTP POST /chat
    → router.chat()
      → research_service.run()
        → RESEARCH_GRAPH (agent)
      → HTTP response

Router should NOT:
  - Call agent directly (call service)
  - Contain business logic (live in service)
  - Know about tool execution (that's service/agent job)
"""

from fastapi import APIRouter
from api.core.errors import APIError
from schemas.llm_schemas import ModelRequest, ModelResponse
from api.services.research_service import research_service
import logging

logger = logging.getLogger(__name__)

# Create router for /chat endpoints
router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("")
def chat(request: ModelRequest) -> ModelResponse:
    """
    Chat/Research endpoint.
    
    Accepts a research query, runs the agent, returns structured response.
    
    Flow (thin router):
      1. Accept request
      2. Call service (business logic)
      3. Return response
      4. Handle errors
    
    Args:
        request: ModelRequest with:
            model: which LLM model to use
            messages: conversation history
            temperature: sampling parameter
            max_tokens: response length limit
            tools: available functions
    
    Returns:
        ModelResponse with:
            model: echo of request model
            output_text: final response from agent
            tool_calls: None (agent is done)
            usage: token counts
    
    Raises:
        APIError: If processing fails
    """
    try:
        # Call service (all heavy lifting happens here)
        # Service handles:
        #   - request_id generation
        #   - state conversion
        #   - agent execution
        #   - response extraction
        #   - error handling
        response = research_service.run(request)
        return response
    
    except Exception as e:
        # Wrap any error as APIError for HTTP response
        logger.error(f"Chat endpoint error: {str(e)}")
        raise APIError(
            message=f"Failed to process chat request: {str(e)}",
            code="CHAT_ERROR",
            status_code=500
        )