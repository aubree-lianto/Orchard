"""
Tool Response Schemas: Data contracts for tool outputs.

These Pydantic models validate that tools return properly structured data.

Structure:
  - SearchResult: output of search_tool
  - FetchResult: output of fetch_tool
  - RetrievalChunk: items in retrieval_tool output list
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SearchResult(BaseModel):
    """
    Single result from search_tool.
    
    Represents a source document discovered via search.
    """
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL or identifier")
    snippet: str = Field(..., description="Text excerpt from document")
    source: str = Field(..., description="Source type: 'web', 'arxiv', 'scholar', 'internal'")
    relevance_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Relevance score 0-1 (higher = more relevant)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "snippet": "The dominant sequence transduction models are based on...",
                "source": "arxiv",
                "relevance_score": 0.98
            }
        }


class FetchResult(BaseModel):
    """
    Result from fetch_tool.
    
    Represents fetched and parsed document content.
    """
    title: str = Field(..., description="Document title")
    url: str = Field(..., description="Document URL or identifier")
    content: str = Field(..., description="Extracted text content (cleaned)")
    length: int = Field(..., ge=0, description="Character count of content")
    source_type: str = Field(
        ..., 
        description="Source type: 'paper', 'web', 'pdf', 'arxiv', 'plain_text'"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata (authors, date, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "title": "Attention Is All You Need",
                "url": "https://arxiv.org/abs/1706.03762",
                "content": "Abstract. The dominant sequence transduction models...",
                "length": 8432,
                "source_type": "arxiv",
                "metadata": {
                    "authors": ["Vaswani et al."],
                    "year": 2017,
                    "pages": "11"
                }
            }
        }


class RetrievalChunk(BaseModel):
    """
    Single chunk retrieved from knowledge base.
    
    Represents a semantically meaningful chunk from vector store or KB.
    """
    id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., description="Chunk text content")
    similarity: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="Similarity score to query (0-1, higher = more relevant)"
    )
    source: str = Field(..., description="Original document source")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Chunk metadata (page, section, timestamp, etc.)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "id": "chunk_abc123",
                "content": "The attention mechanism allows the model to focus on relevant parts of the input...",
                "similarity": 0.95,
                "source": "paper_attention_2017",
                "metadata": {
                    "page": 5,
                    "section": "Attention Mechanism",
                    "collection": "ml_papers"
                }
            }
        }


# Type aliases for convenience
SearchResults = List[SearchResult]
RetrievalChunks = List[RetrievalChunk]