"""
LangChain tool definitions for the research agent.

Tools are callable functions that the agent can invoke.
Tools have names, descriptions, and structured input/output schemas.
"""

from langchain.tools import tool
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


@tool
def search_tool(query: str, source: str = "web", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Search for relevant sources (academic papers, web, knowledge base).
    
    Args:
        query: Research query
        source: "web", "arxiv", "scholar", or "internal"
        limit: Max results to return
    
    Returns:
        List of sources with title, url, snippet, source, relevance_score
    """
    logger.info(f"[Search] query={query} | source={source} | limit={limit}")
    
    # STUB: Mock results for now
    # TODO: Replace with:
    #   - arXiv API: requests.get("https://api.arxiv.org/query")
    #   - Web: Tavily Search API or Google Search
    #   - Internal: Vector store query
    
    results = [
        {
            "title": f"Research Paper: {query}",
            "url": f"https://example.com/paper-{i}",
            "snippet": f"This paper discusses {query} in detail...",
            "source": source,
            "relevance_score": 0.95 - (i * 0.05)
        }
        for i in range(min(limit, 3))
    ]
    
    logger.info(f"[Search] Found {len(results)} results")
    return results


@tool
def fetch_tool(url: str, max_length: int = 8000) -> Dict[str, Any]:
    """
    Fetch and parse document content from URL.
    
    Handles HTTP, PDF, arXiv abstracts, HTML cleaning.
    
    Args:
        url: Document URL or arXiv ID (e.g., "1706.03762")
        max_length: Max characters to return
    
    Returns:
        Dict with title, url, content, length, source_type, metadata
    """
    logger.info(f"[Fetch] url={url} | max_length={max_length}")
    
    # STUB: Mock document for now
    # TODO: Replace with:
    #   - arXiv: fetch abs + pdf parsing
    #   - Web: requests.get() + BeautifulSoup
    #   - PDF: pypdf2 or pdfplumber
    
    mock_content = f"""
    Title: {url}
    
    Abstract. This document presents important research findings.
    
    Introduction. The field has evolved significantly.
    The current work builds upon previous research.
    
    Methods. We employ state-of-the-art techniques.
    Results include comprehensive benchmarks.
    
    Conclusion. This work advances the field meaningfully.
    """[:max_length]
    
    logger.info(f"[Fetch] Retrieved {len(mock_content)} chars")
    
    return {
        "title": f"Document: {url}",
        "url": url,
        "content": mock_content,
        "length": len(mock_content),
        "source_type": "paper",
        "metadata": {"fetched_at": "2024-03-01"}
    }


@tool
def retrieval_tool(query: str, collection: str = "general", limit: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks from knowledge base / vector store.
    
    Later: Replace with Redis + embeddings, Pinecone, Weaviate, etc.
    
    Args:
        query: Semantic search query
        collection: Knowledge base ("general", "papers", "faq")
        limit: Max results
    
    Returns:
        List of chunks with id, content, similarity, source, metadata
    """
    logger.info(f"[Retrieval] query={query} | collection={collection}")
    
    # STUB: Mock chunks for now
    # TODO: Replace with vector store query
    
    chunks = [
        {
            "id": f"chunk_{i}",
            "content": f"Context about {query}: Knowledge base entry {i}",
            "similarity": 0.95 - (i * 0.1),
            "source": f"doc_{i}",
            "metadata": {"collection": collection}
        }
        for i in range(min(limit, 3))
    ]
    
    logger.info(f"[Retrieval] Found {len(chunks)} chunks")
    return chunks

# Export tools
TOOLS = [search_tool, fetch_tool, retrieval_tool]


def get_tool_by_name(tool_name: str):
    """Retrieve tool by name."""
    tool_map = {tool.name: tool for tool in TOOLS}
    return tool_map.get(tool_name)