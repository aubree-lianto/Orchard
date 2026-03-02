"""
Tools: External capabilities for the agent.

Tools are callable functions that the agent can invoke.
They represent connections to the external world (search, calculation, retrieval, etc.).

Structure:
  - Each tool is a separate module (search.py, fetch.py, etc.)
  - All tools exported from __init__.py
  - Agent imports tools but does not implement them
"""

from api.tools.research import search_tool, fetch_tool, retrieval_tool, TOOLS

def get_tool_by_name(name: str):
    return next((t for t in TOOLS if t.name == name), None)

__all__ = [
    "search_tool",
    "fetch_tool",
    "retrieval_tool",
    "TOOLS",
    "get_tool_by_name"
]

