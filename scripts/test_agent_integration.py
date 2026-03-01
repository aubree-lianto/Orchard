"""
Agent Integration Tests (Mock Backend)

Covers:
- LLM-only path (no tool call)
- Tool invocation path (calculator_tool)
- Invalid input handling (empty messages, unknown tool)

Requires mock server running — handled automatically by conftest.py fixture.
"""

import sys
import os
import pytest
from unittest.mock import patch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from app.core.agent_state import AgentState
from app.agents.research_agent import RESEARCH_GRAPH
from schemas.llm_schemas import ModelResponse


# --- Path 1: LLM-only (no tool call) ---

def test_llm_only_path():
    """Message with no keyword triggers no tool call — graph exits after 1 iteration."""
    state = AgentState(messages=[{"role": "user", "content": "Hello, what can you do?"}])
    result = RESEARCH_GRAPH.invoke(state)
    final = AgentState(**result)

    assert final.iteration == 1
    assert len(final.messages) == 2  # user + assistant
    assert final.messages[-1]["role"] == "assistant"
    assert not any(s.get("action") == "tool_call" for s in final.intermediate_steps)


# --- Path 2: Tool invocation ---

def test_tool_invocation_path():
    """'calculate' keyword triggers calculator_tool — graph loops llm→tool→llm."""
    state = AgentState(messages=[{"role": "user", "content": "Please calculate 2 + 2 for me"}])
    result = RESEARCH_GRAPH.invoke(state)
    final = AgentState(**result)

    assert final.iteration >= 2

    tool_steps = [s for s in final.intermediate_steps if s.get("action") == "tool_call"]
    assert len(tool_steps) >= 1

    tool_names = [s["detail"]["tool"] for s in tool_steps]
    assert "calculator_tool" in tool_names

    tool_messages = [m for m in final.messages if m.get("role") == "tool"]
    assert len(tool_messages) >= 1
    assert "4.0" in tool_messages[0]["content"]


# --- Path 3: Invalid input handling ---

def test_empty_messages_returns_response():
    """Empty messages list — graph still completes, returning an assistant response."""
    state = AgentState(messages=[])
    result = RESEARCH_GRAPH.invoke(state)
    final = AgentState(**result)

    assert final.iteration == 1
    assert len(final.messages) == 1  # only the assistant reply, no user message
    assert final.messages[-1]["role"] == "assistant"


def test_unknown_tool_gracefully_handled():
    """
    If the LLM returns an unknown tool name, node_tool_executor should
    append an error message rather than crash, and the graph should complete.
    """
    fake_tool_response = ModelResponse(
        model="gpt-mock",
        output_text="Using a tool.",
        tool_calls=[{
            "id": "call_x",
            "type": "function",
            "function": {"name": "nonexistent_tool", "arguments": "{}"}
        }],
        usage={"total_tokens": 10}
    )
    fake_final_response = ModelResponse(
        model="gpt-mock",
        output_text="Done.",
        tool_calls=None,
        usage={"total_tokens": 5}
    )

    with patch("app.agents.research_agent.get_model_client") as mock_client:
        instance = mock_client.return_value
        # First call returns unknown tool; second call ends the loop
        instance.chat.side_effect = [fake_tool_response, fake_final_response]

        state = AgentState(messages=[{"role": "user", "content": "Do something"}])
        result = RESEARCH_GRAPH.invoke(state)
        final = AgentState(**result)

    error_messages = [m for m in final.messages if "Error: Unknown tool" in m.get("content", "")]
    assert len(error_messages) >= 1
