"""
Tools: External capabilities for the agent.

Tools are callable functions that the agent can invoke.
They represent connections to the external world (search, calculation, retrieval, etc.).

Structure:
  - Each tool is a separate module (search.py, fetch.py, etc.)
  - All tools exported from __init__.py
  - Agent imports tools but does not implement them
"""

from api.tools.basic import echo_tool, calculator_tool, search_tool

# Export tools as a list for easy registration
TOOLS = [echo_tool, calculator_tool, search_tool]


def get_tool_by_name(tool_name: str):
    """
    Helper function to retrieve a tool by name.
    
    Args:
        tool_name: The name of the tool to retrieve
        
    Returns:
        The tool object, or None if not found
        
    Example:
        tool = get_tool_by_name("calculator_tool")
        if tool:
            result = tool.invoke({"expression": "2+2"})
    """
    tool_map = {tool.name: tool for tool in TOOLS}
    return tool_map.get(tool_name)
