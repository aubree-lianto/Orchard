"""
Basic Tool Definitions: Core tools for research agent.

These tools are callable functions that the agent can invoke.
Tools have names, descriptions, and structured input/output schemas.

Why separate from agents?
  - Tools are external capabilities, not orchestration
  - Agent imports tools but does not own them
  - Tools can be reused across multiple agents
  - Prevents circular dependencies
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
        
    Example:
        echo_tool.invoke({"message": "hello"})
        → "Echo: hello"
    """
    logger.info(f"[Echo Tool] Received: {message}")
    return f"Echo: {message}"


@tool 
def calculator_tool(expression: str) -> Union[float, str]:
    """
    Calculator tool: Evaluates simple mathematical expressions.
    
    Supports: +, -, *, /, and parentheses.
    
    When LLM decides to use this:
      - LLM provides expression: "2+2"
      - Tool evaluates it: 4.0
      - Result returned to LLM for final answer
    
    Args:
        expression: A mathematical expression (e.g., "2 + 2" or "10 * 5")
        
    Returns:
        The result of the calculation, or error message if invalid
        
    Example:
        calculator_tool.invoke({"expression": "2+2"})
        → 4.0
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
    
    In production, this would:
      - Query a search engine (Google, Bing, etc.)
      - Query a vector database (for semantic search)
      - Query a knowledge base
    
    Currently: Returns mock results
    
    Args:
        query: The search query string
        
    Returns:
        Search results (currently mocked)
        
    Example:
        search_tool.invoke({"query": "Python best practices"})
        → "Mock search results for: 'Python best practices'"
        
    Future:
        Should integrate with:
          - Semantic search (vector DB)
          - Web search API
          - Knowledge base retrieval
    """
    logger.info(f"[Search Tool] Searching for: {query}")
    # Stub implementation
    results = f"Mock search results for: '{query}'"
    logger.info(f"[Search Tool] Returning: {results}")
    return results
