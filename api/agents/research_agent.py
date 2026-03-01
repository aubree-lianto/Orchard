from langgraph.graph import StateGraph, END
from langchain_core.utils.function_calling import convert_to_openai_function
from typing import Union
from api.core.agent_state import AgentState
from api.core.provider import get_model_client
from api.tools import TOOLS, get_tool_by_name
import logging
import json
import time

logger = logging.getLogger(__name__)

# Set up basic graph nodes

def node_llm_call(state: AgentState) -> AgentState:
    """
    Basic node_llm_call 

    Given basic convo history + tools -> provide an answer 
    LLM then response with text answer + tool calls
    """

    # Request id from state metadata
    request_id = state.metadata.get("request_id", "?")

    # Log node entry w necessary context
    # Shows which iteration of LLM call this as well as conversation length
    logger.info(f"[Request:{request_id}] --> node: llm_call | iteration={state.iteration} | messages={len(state.messages)}")

    # Record start time to measure execution latency
    t0 = time.time()

    # Get ModelClient instance 
    client = get_model_client()

    from schemas.llm_schemas import ModelRequest, Message

    # convert each tool into OpenAI function schema
    # all tools were decorated w LangChain tool decorator -> convert to necessary function schema
    tools_as_openai = [convert_to_openai_function(tool) for tool in TOOLS]

    # Define model request for LLM
    request = ModelRequest(
        model = "gpt-mock",
        tools = tools_as_openai,
        messages = [
            Message(role=msg["role"], content = msg["content"])
            for msg in state.messages
        ]
    )

    # Fetch response
    response = client.chat(request)

    # Append message to state
    state.messages.append({
        "role": "assistant",
        "content": response.output_text
    })

    # Parse tool calls and normalize 
    normalized_tool_calls = None

    # If tool_calls were present
    if response.tool_calls:
        normalized_tool_calls = []

        # process each tool call
        for tc in response.tool_calls:
            fn = tc.get("function", {})
            # extract arguments from function call (aka responses)
            raw_args = fn.get("arguments", "{}")

            arguments = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

            # normalize tool calls
            normalized_tool_calls.append({
                "tool_name": fn.get("name", ""),
                "arguments": arguments
            })
        
    state.tool_calls = normalized_tool_calls

    # increment iteration counter -> helps to enforce max_iterations limit
    state.iteration += 1

    # log execution to intermediate steps
    state.intermediate_steps.append({
        "action": "llm_response",
        "detail": {
            "tokens_used": response.usage.get("total_tokens", 0),
            "has_tool_calls": bool(response.tool_calls)
        }
    })
    
    # calculates how long node took
    elapsed = time.time() - t0

    # log node exit and return state
    logger.info(
        f"[Request:{request_id}] <-- node: llm_call | "
        f"elapsed={elapsed:.3f}s | tokens={response.usage.get('total_tokens',0)} | "
        f"tool_calls={len(normalized_tool_calls) if normalized_tool_calls else 0}"
    )
    return state


def node_tool_executor(state: AgentState) -> AgentState:
    """
    Node 2: Execute tool calls if any.

    Input: AgentState with potential tool_calls
    Output: AgentState with tool results added to messages
    """
    request_id = state.metadata.get("request_id", "?")

    # checks if ModelRequest included any tool_calls
    # otherwise we just return the default state
    if not state.tool_calls:
        logger.info(f"[Request:{request_id}] --> node: tool_executor | no tool calls, skipping")
        return state

    # log tool_executor entry
    logger.info(f"[Request:{request_id}] --> node: tool_executor | pending={len(state.tool_calls)}")

    # loop through each tool call
    for tool_call in state.tool_calls:
        
        # get tool name and arguments
        tool_name = tool_call.get("tool_name")
        args = tool_call.get("arguments", {})

        # extra validation to ensure tool_name is valid
        tool = get_tool_by_name(tool_name)

        # exit if no such tool exists
        if not tool:
            logger.warning(f"[Request:{request_id}] tool: {tool_name} | ERROR: unknown tool")
            tool_result = f"Error: Unknown tool '{tool_name}'"
        else:
            try:
                # figure out latency time
                t0 = time.time()

                # invoke tool with parsed arguments
                tool_result = tool.invoke(args)

                # calculate latency
                elapsed = time.time() - t0
                logger.info(
                    f"[Request:{request_id}] tool: {tool_name} | "
                    f"args={args} | result={str(tool_result)[:80]} | elapsed={elapsed:.3f}s"
                )
            except Exception as e:
                tool_result = f"Error executing {tool_name}: {str(e)}"
                logger.error(f"[Request:{request_id}] tool: {tool_name} | ERROR: {e}")

        # Add observation to messages
        state.messages.append({
            "role": "tool",
            "content": f"Tool {tool_name} returned: {tool_result}"
        })

        state.intermediate_steps.append({
            "action": "tool_call",
            "detail": {
                "tool": tool_name,
                "args": args,
                "result": tool_result,
                "iteration": state.iteration
            }
        })

    logger.info(f"[Request:{request_id}] <-- node: tool_executor")
    return state

def router_decision(state: AgentState) -> str:
    """
    Decide whether to continue loop or end
    Also known as "gate" -> conditional edge that routes to the next responsible node

    Input: AgentState after LLM call
    Output: "continue" (execute tools) or "end" (return response)
    """
    request_id = state.metadata.get("request_id", "?")
    has_tool_calls = bool(state.tool_calls)
    max_iterations = 5

    # if exceeds max iterations -> exit
    if state.iteration >= max_iterations:
        logger.info(f"[Request:{request_id}] router: llm_call --> end | reason=max_iterations")
        return "end"

    if has_tool_calls:
        logger.info(f"[Request:{request_id}] router: llm_call --> tool_executor | tool_calls={len(state.tool_calls)}")
        return "continue"

    logger.info(f"[Request:{request_id}] router: llm_call --> end | reason=no_tool_calls")
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