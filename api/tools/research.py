"""
LangChain tool definitions for the research agent.

Tools are callable functions that the agent can invoke.
Tools have names, descriptions, and structured input/output schemas.
"""
import os
import re
import requests
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from bs4 import BeautifulSoup
from langchain.tools import tool
from schemas.tools import SearchResult, SearchResults, FetchResult, RetrievalChunk, RetrievalChunks
from api.core.settings import settings
import logging

logger = logging.getLogger(__name__)

@tool
def search_tool(query: str, source: str = "web", limit: int = 5) -> SearchResults:
    """
    Search for relevant sources (academic papers, web, knowledge base).
    
    Args:
        query: Research query
        source: "web", "arxiv", "scholar", or "internal"
        limit: Max results to return
    
    Returns:
        List of sources with title, url, snippet, source, relevance_score
    """
    # Log when search tool was called
    logger.info(f"[Search] query={query} | source={source} | limit={limit}")

    # Fetch source
    if source == "arxiv":
        results = _search_arxiv(query, limit)
    elif source == "scholar":
        results = _search_semantic_scholar(query, limit)
    else:
        raise NotImplementedError(f"Search source '{source}' is not implemented")

    logger.info(f"[Search] Found {len(results)} results")
    return results

# define function to search semantic scholar
def _search_semantic_scholar(query: str, limit: int) -> list:

    # Hit semantic scholar /paper search endpoint
    # find papers lol
    response = requests.get(
        "https://api.semanticscholar.org/graph/v1/paper/search",
        params={
            "query": query,
            "limit": limit,
            "fields": "title,abstract,url,year,authors"
        },
        # 10 second timeout in case something happens
        timeout=10 
    )

    # raise exception if HTTP satus is error
    response.raise_for_status()

    results = []

    # Iterate over papers returned by API
    for i, paper in enumerate(response.json().get("data", [])):

        # extract abstract -> truncate to 300 chars
        abstract = (paper.get("abstract") or "")[:300]

        # create SearchResult pydantic model with all required fields
        # Relevance score: first result = 1.0 -> decreased by 0.05 per rank
        results.append(SearchResult(
            title=paper.get("title", ""),
            url=paper.get("url", ""),
            snippet=abstract,
            source="scholar",
            relevance_score=round(1.0 - (i * 0.05), 2)
        ).model_dump())

    # convert pydantic model to dict for LangChain
    # LangChain tools can't serialzie pydantic models directly
    return results

# Define a function to search arxiv similar to previous function
def _search_arxiv(query: str, limit: int) -> list:
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": f"all:{query}", "start": 0, "max_results": limit},
        timeout=10
    )
    response.raise_for_status()

    # ArXiv for some reason returns Atom XML format
    # ET = Element tree which parses XML
    root = ET.fromstring(response.text)

    # Define XML namespace for Atom
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    results = []

    for i, entry in enumerate(root.findall("atom:entry", ns)):
        title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
        url = (entry.findtext("atom:id", default="", namespaces=ns) or "").strip()
        summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
        results.append(SearchResult(
            title=title,
            url=url,
            snippet=summary[:300],
            source="arxiv",
            relevance_score=round(1.0 - (i * 0.05), 2)
        ).model_dump())

    return results


@tool
def fetch_tool(url: str, max_length: int = 8000) -> FetchResult:
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

    arxiv_id = _extract_arxiv_id(url)
    if arxiv_id:
        result = _fetch_arxiv(arxiv_id, max_length)
    else:
        result = _fetch_web(url, max_length)

    logger.info(f"[Fetch] Retrieved {result['length']} chars from {url}")
    return result


def _extract_arxiv_id(url: str) -> str | None:
    # Matches arxiv.org/abs/XXXX.XXXXX or bare IDs like 1706.03762
    match = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)", url)
    if match:
        return match.group(1)
    if re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}", url):
        return url
    return None


def _fetch_arxiv(arxiv_id: str, max_length: int) -> dict:
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"id_list": arxiv_id},
        timeout=10
    )
    response.raise_for_status()

    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entry = root.find("atom:entry", ns)

    title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
    summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
    published = (entry.findtext("atom:published", default="", namespaces=ns) or "").strip()
    authors = [a.findtext("atom:name", namespaces=ns) for a in entry.findall("atom:author", ns)]

    content = summary[:max_length]

    result = FetchResult(
        title=title,
        url=f"https://arxiv.org/abs/{arxiv_id}",
        content=content,
        length=len(content),
        source_type="arxiv",
        metadata={
            "authors": authors,
            "published": published,
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }
    )
    
    return result.model_dump()


def _fetch_web(url: str, max_length: int) -> dict:
    response = requests.get(url, timeout=10, headers={"User-Agent": "Orchard/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    title = soup.title.string.strip() if soup.title else url
    content = soup.get_text(separator="\n")[:max_length]

    result = FetchResult(
        title=title,
        url=url,
        content=content,
        length=len(content),
        source_type="web",
        metadata={"fetched_at": datetime.now(timezone.utc).isoformat()}
    )
    return result.model_dump()


@tool
def retrieval_tool(query: str, collection: str = "general", limit: int = 5) -> RetrievalChunks:
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
    
    chunks = []
    
    for i in range(min(limit, 3)):
        chunk = RetrievalChunk(
            id=f"chunk_{i}",
            content=f"Context about {query}: Knowledge base entry {i}",
            similarity=0.95 - (i * 0.1),
            source=f"doc_{i}",
            metadata={"collection": collection}
        )
        chunks.append(chunk.model_dump())
    
    logger.info(f"[Retrieval] Found {len(chunks)} chunks")
    return chunks

# Export tools
TOOLS = [search_tool, fetch_tool, retrieval_tool]