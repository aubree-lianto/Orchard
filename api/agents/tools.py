"""

Basic LangChain tool definitions for the research agent

These tools are callable functions that the agent can invoke
Tools have names, descriptions, and structured input/output schemas

"""

from langchain.tools import tool
from typing import Union
import logging

logger = logging.getLogger(__name__)

@tool
def echo_tool(message: str) -> str:
    """
    Echo tool: Returns the input message unchanged.
    
    Useful for testing tool execution flow.
    
    Args:
        message: The message to echo back
        
    Returns:
        The same message
    """
    logger.info(f"[Echo Tool] Recevied: {message}")
    return f"Echo: {message}"

@tool 
def calculator_tool(expression: str) -> Union[float, str]:
    """
    Calculator tool: Evaluates simple mathematical expressions.
    
    Supports: +, -, *, /, and parentheses.
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2" or "10 * 5")
        
    Returns:
        The result of the calculation, or error message if invalid
    """
    try:
        logger.info(f"[Calculator Tool] Evaluating: {expression}")
        result = eval(expression)
        logger.info(f"[Calculator Tool] Result: {result}")
        return float(result)
    except Exception as e:
        error_msg = f"Calculation error: {str(e)}"
        logger.error(error_msg)
        return error_msg

@tool
def search_tool(query: str) -> str:
    """
    Search tool: Searches for information (stubbed for now).
    
    In production, this would query a search engine or vector database.
    
    Args:
        query: The search query string
        
    Returns:
        Mock search results
    """
    logger.info(f"[Search Tool] Searching for: {query}")
    # Stub implementation
    results = f"Mock search results for: '{query}'"
    logger.info(f"[Search Tool] Returning: {results}")
    return results

# Export tools as a list for easy registration
TOOLS = [echo_tool, calculator_tool, search_tool]

def get_tool_by_name(tool_name: str):
    """Helper function to retrieve a tool by name."""
    tool_map = {tool.name: tool for tool in TOOLS}
    return tool_map.get(tool_name)