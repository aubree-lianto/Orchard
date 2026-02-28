"""
Test AgentState and LangGraph agent integration.
"""

import sys
import os 

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from app.core.agent_state import AgentState
from app.agents.research_agent import RESEARCH_GRAPH
from schemas.llm_schemas import ModelRequest, Message
import json


def test_agent_state():
    print("=" * 60)
    print("TEST 1: AgentState Creation")
    print("=" * 60)
    
    state = AgentState(
        messages=[
            {"role": "user", "content": "What is 2+2?"}
        ],
        metadata={"user_id": "test_user"}
    )
    
    print(f"✅ State created:")
    print(f"   - Messages: {len(state.messages)}")
    print(f"   - Iteration: {state.iteration}")
    print(f"   - Metadata: {state.metadata}")
    
    print("\n" + "=" * 60)
    print("TEST 2: LangGraph Agent Execution")
    print("=" * 60)
    
    initial_state = AgentState(
        messages=[
            {"role": "user", "content": "Hello, what can you do?"}
        ]
    )
    
    print(f"Input state: {initial_state.messages}")
    
    final_state_dict = RESEARCH_GRAPH.invoke(initial_state)
    final_state = AgentState(**final_state_dict)
    
    print(f"✅ Graph execution complete:")
    print(f"   - Final messages: {len(final_state.messages)}")
    print(f"   - Iterations: {final_state.iteration}")
    print(f"   - Intermediate steps: {len(final_state.intermediate_steps)}")
    print(f"   - Final response: {final_state.messages[-1]['content'][:50]}...")
    
    print("\n" + "=" * 60)
    print("TEST 3: Tool Invocation Path")
    print("=" * 60)

    initial_state = AgentState(
        messages=[
            {"role": "user", "content": "Please calculate 2 + 2 for me"}
        ]
    )

    final_state_dict = RESEARCH_GRAPH.invoke(initial_state)
    final_state = AgentState(**final_state_dict)

    # Graph should have looped: llm_call → tool_executor → llm_call
    assert final_state.iteration >= 2, (
        f"Expected >= 2 iterations (llm -> tool -> llm), got {final_state.iteration}"
    )

    # At least one tool_call step in intermediate_steps
    tool_steps = [s for s in final_state.intermediate_steps if s.get("action") == "tool_call"]
    assert len(tool_steps) >= 1, "Expected at least one tool_call in intermediate_steps"

    # The tool that ran should be calculator_tool
    tool_names = [s["detail"]["tool"] for s in tool_steps]
    assert "calculator_tool" in tool_names, f"Expected calculator_tool, got: {tool_names}"

    # A message with role "tool" should exist
    tool_messages = [m for m in final_state.messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1, "Expected at least one message with role='tool'"

    print(f"✅ Tool invocation complete:")
    print(f"   - Iterations: {final_state.iteration}")
    print(f"   - Tool calls executed: {tool_names}")
    print(f"   - Tool result: {tool_messages[0]['content']}")

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_agent_state()