"""
Research Service: Application business logic layer.

The service layer sits between routers and agents:
  - Routers call services (not agents directly)
  - Services orchestrate agent execution
  - Services handle input validation, logging, error handling
  - Services are where business logic lives

Architecture:
  Router → Service → Agent → Tools

This enforces clean separation:
  - Router: HTTP handling only
  - Service: Application logic
  - Agent: Workflow orchestration
  - Tools: External capabilities

Why a service layer?
  1. Decouples HTTP from agent logic
  2. Allows multiple routers to share same service
  3. Makes testing easier (mock service in tests)
  4. Centralizes error handling + logging
  5. Future: easier to add caching, rate limiting, etc.
"""

import logging
import uuid
from api.core.agent_state import AgentState
from api.agents.research_agent import RESEARCH_GRAPH
from schemas.llm_schemas import ModelRequest, ModelResponse

logger = logging.getLogger(__name__)


class ResearchService:
    """
    Research Service: Orchestrates the research agent.
    
    Responsibility:
      - Convert user requests to agent state
      - Execute the research workflow
      - Extract final response
      - Handle errors gracefully
      - Log execution metrics
    
    This is where "business logic" lives—not in routers, not in agents.
    """
    
    def __init__(self):
        """Initialize the research service."""
        pass
    
    def run(self, request: ModelRequest) -> ModelResponse:
        """
        Execute a research request using the agent.
        
        Flow:
          1. Create request_id for tracing
          2. Convert ModelRequest → AgentState
          3. Run RESEARCH_GRAPH (the agent loop)
          4. Extract final response
          5. Return as ModelResponse
        
        Args:
            request: User request (from HTTP layer)
                model: which model to use
                messages: conversation history
                temperature: sampling parameter
                max_tokens: response length limit
                tools: available functions
        
        Returns:
            ModelResponse with:
                model: echo back request model
                output_text: final LLM response
                tool_calls: (should be None after completion)
                usage: token counts
        
        Raises:
            Exception: If agent execution fails (will be caught by router)
        
        Example:
            service = ResearchService()
            response = service.run(request)
        """
        # ====================================================================
        # STEP 1: CREATE REQUEST ID FOR TRACING
        # ====================================================================
        
        # Generate unique request ID for this execution
        # Used to tie all logs together
        # Example: "abc12345"
        request_id = str(uuid.uuid4())[:8]
        
        # Log the incoming request
        logger.info(
            f"[Request:{request_id}] Research request received | "
            f"model={request.model} | messages={len(request.messages)} | "
            f"temperature={request.temperature}"
        )

        # ====================================================================
        # STEP 2: CONVERT HTTP REQUEST TO AGENT STATE
        # ====================================================================
        
        # Create initial state for the agent
        # This is what the agent will process
        # AgentState contains:
        #   - messages: conversation history (converted from ModelRequest)
        #   - tool_calls: will be set by agent
        #   - iteration: will increment as loop runs
        #   - metadata: includes request_id for logging
        #
        # Why convert?
        #   - ModelRequest is HTTP contract (external)
        #   - AgentState is agent contract (internal)
        #   - Service handles this conversion
        initial_state = AgentState(
            messages=list(request.messages),
            metadata={"request_id": request_id, "model": request.model}
        )

        # ====================================================================
        # STEP 3: EXECUTE THE RESEARCH GRAPH
        # ====================================================================
        
        # Run the agent loop
        # RESEARCH_GRAPH.invoke() will:
        #   1. Start at entry point (node_llm_call)
        #   2. Call LLM
        #   3. Check if tools needed (router_decision)
        #   4. Execute tools if needed (node_tool_executor)
        #   5. Loop back to LLM
        #   6. Return when complete
        #
        # The final_state contains the complete execution:
        #   - messages: full conversation + response
        #   - iteration: how many LLM calls were made
        #   - intermediate_steps: execution trace
        #   - tool_calls: None (should be None at end)
        logger.info(
            f"[Request:{request_id}] Starting research graph execution"
        )
        
        final_state = RESEARCH_GRAPH.invoke(initial_state)
        
        # ====================================================================
        # STEP 4: EXTRACT FINAL RESPONSE
        # ====================================================================
        
        # Get the last message (the final response from the assistant)
        # final_state.messages looks like:
        #   [
        #     {"role": "user", "content": "What is 2+2?"},
        #     {"role": "assistant", "content": "I'll calculate that"},
        #     {"role": "tool", "content": "Tool returned: 4.0"},
        #     {"role": "assistant", "content": "The answer is 4."}  ← This is what user sees
        #   ]
        final_message = final_state["messages"][-1]["content"]

        # ====================================================================
        # STEP 5: LOG EXECUTION METRICS
        # ====================================================================
        
        # Log that we're done with metrics
        logger.info(
            f"[Request:{request_id}] Research completed | "
            f"iterations={final_state['iteration']} | "
            f"messages={len(final_state['messages'])}"
        )
        
        # ====================================================================
        # STEP 6: RETURN MODELRESPONSE
        # ====================================================================
        
        # Convert agent result back to HTTP response contract
        # This is the inverse of step 2
        # ModelResponse is what client expects (external contract)
        response = ModelResponse(
            model=request.model,  # Echo back the model
            output_text=final_message,  # The final answer
            tool_calls=None,  # Should be None (no pending tools)
            usage={"total_tokens": 0}  # TODO: track actual tokens
        )
        
        logger.info(
            f"[Request:{request_id}] Response generated | "
            f"response_length={len(final_message)}"
        )
        
        return response


# Create service instance (stateless, can be singleton)
# Used by routers
research_service = ResearchService()
