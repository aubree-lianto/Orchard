from langgraph.graph import StateGraph, END
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import Union
from app.core.agent_state import AgentState
from app.core.provider import get_model_client
from app.agents.tools import TOOLS, get_tool_by_name
import logging
import json

logger = logging.getLogger(__name__)

# Set up basic graph nodes

# Calls the LLM via ModelClient
def node_llm_call(state: AgentState) -> AgentState:
    
    logger.info(f"[LLM Call] Iteration {state.iteration}, Messages: {len(state.messages)}")

    client = get_model_client()

    from schemas.llm_schemas import ModelRequest, Message

    tools_as_openai = [convert_to_openai_function(tool) for tool in TOOLS]

    request = ModelRequest(
        model = "gpt-mock",
        tools = tools_as_openai,
        messages = [
            Message(role=msg["role"], content = msg["content"])
            for msg in state.messages
        ]
    )

    response = client.chat(request)

    state.messages.append({
        "role": "assistant",
        "content": response.output_text
    })

    normalized_tool_calls = None
    if response.tool_calls:
        normalized_tool_calls = []
        for tc in response.tool_calls:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", "{}")
            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            normalized_tool_calls.append({
                "tool_name": fn.get("name", ""),
                "arguments": arguments
            })
    state.tool_calls = normalized_tool_calls

    state.iteration += 1

    state.intermediate_steps.append({
        "action": "llm_response",
        "detail": {
            "tokens_used": response.usage.get("total_tokens", 0),
            "has_tool_calls": bool(response.tool_calls)
        }
    })

    logger.info(f"[LLM Call] Response: {response.output_text[:50]}...")
    return state


# Placeholder for future execution
def node_tool_executor(state: AgentState) -> AgentState:
    """
    Node 2: Execute tool calls if any.
    
    Input: AgentState with potential tool_calls
    Output: AgentState with tool results added to messages
    """
    if not state.tool_calls:
        logger.info("No tool calls to execute")
        return state
    
    logger.info(f"[Tool Executor] Executing {len(state.tool_calls)} tool calls")
    
    for tool_call in state.tool_calls:
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("arguments", {})
        
        tool = get_tool_by_name(tool_name)

        if not tool:
            logger.warning(f"[Tool Executor] Unknown tool: {tool_name}")
            tool_result = f"Error: Unknown tool '{tool_name}'"
        else: 
            try:
                logger.info(f"[Tool Executor] Calling {tool_name} with {args}")
                # Call the tool with unpacked arguments
                tool_result = tool.invoke(args)
                logger.info(f"[Tool Executor] {tool_name} returned: {tool_result}")
            except Exception as e:
                tool_result = f"Error executing {tool_name}: {str(e)}"
                logger.error(tool_result)
        
        # Add observation to messages
        state.messages.append({
            "role": "tool",  # Tool responses come as observations
            "content": f"Tool {tool_name} returned: {tool_result}"
        })

        state.intermediate_steps.append({
            "action": "tool_call",
            "detail": {
                "tool": tool_name,
                "args": args,
                "result": tool_result
            }
        })
    
    return state

def router_decision(state: AgentState) -> str:
    """
    Decide whether to continue loop or end.
    
    Input: AgentState after LLM call
    Output: "continue" (execute tools) or "end" (return response)
    """
    has_tool_calls = bool(state.tool_calls)
    max_iterations = 5
    
    if state.iteration >= max_iterations:
        logger.info(f"Max iterations ({max_iterations}) reached")
        return "end"
    
    if has_tool_calls:
        logger.info("Tool calls detected, continuing loop")
        return "continue"
    
    logger.info("No tool calls, ending loop")
    return "end"

def build_research_graph() -> StateGraph:
    """
    Build the LangGraph state graph.
    
    Returns a compiled graph where all nodes operate on AgentState.
    """
    graph = StateGraph(AgentState)
    
    # Add nodes
    graph.add_node("llm_call", node_llm_call)
    graph.add_node("tool_executor", node_tool_executor)
    
    # Define flow
    graph.set_entry_point("llm_call")
    
    graph.add_conditional_edges(
        "llm_call",
        router_decision,
        {
            "continue": "tool_executor",
            "end": END
        }
    )
    
    graph.add_edge("tool_executor", "llm_call")
    
    logger.info("Research graph built successfully")
    return graph.compile()


# Compile graph on module load
RESEARCH_GRAPH = build_research_graph()