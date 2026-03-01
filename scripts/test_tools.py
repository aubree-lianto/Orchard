"""
Test research tools independently using pytest.

Tests validate:
  - Tool invocation and response structures
  - Data types returned
  - Required fields in responses
"""

import sys
import os
import pytest

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from api.tools import search_tool, fetch_tool, retrieval_tool, TOOLS


# ===== TOOL STRUCTURE TESTS =====

class TestSearchTool:
    """Test search_tool for web/arXiv document discovery."""
    
    def test_search_tool_invocation(self):
        """Test that search_tool can be invoked."""
        result = search_tool.invoke({
            "query": "machine learning",
            "source": "web",
            "limit": 3
        })
        assert result is not None
        
    def test_search_tool_returns_list(self):
        """Test that search_tool returns a list."""
        result = search_tool.invoke({
            "query": "neural networks",
            "source": "arxiv",
            "limit": 5
        })
        assert isinstance(result, list), "search_tool should return a list"
        
    def test_search_result_structure(self):
        """Test that search results have required fields."""
        result = search_tool.invoke({
            "query": "transformers",
            "source": "web",
            "limit": 2
        })
        if result:  # If we get results
            for item in result:
                assert isinstance(item, dict), "Each result should be a dict"
                assert "title" in item, "Result should have 'title' field"
                assert "url" in item, "Result should have 'url' field"
                assert "snippet" in item, "Result should have 'snippet' field"
                assert "relevance_score" in item, "Result should have 'relevance_score' field"


class TestFetchTool:
    """Test fetch_tool for document content retrieval."""
    
    def test_fetch_tool_invocation(self):
        """Test that fetch_tool can be invoked."""
        result = fetch_tool.invoke({
            "url": "https://example.com/paper",
            "max_length": 5000
        })
        assert result is not None
        
    def test_fetch_tool_returns_dict(self):
        """Test that fetch_tool returns a dictionary."""
        result = fetch_tool.invoke({
            "url": "https://arxiv.org/abs/2101.00000",
            "max_length": 8000
        })
        assert isinstance(result, dict), "fetch_tool should return a dict"
        
    def test_fetch_result_structure(self):
        """Test that fetch results have required fields."""
        result = fetch_tool.invoke({
            "url": "https://example.com",
            "max_length": 4000
        })
        assert "title" in result, "Result should have 'title' field"
        assert "url" in result, "Result should have 'url' field"
        assert "content" in result, "Result should have 'content' field"
        assert "length" in result, "Result should have 'length' field"
        assert "source_type" in result, "Result should have 'source_type' field"
        assert "metadata" in result, "Result should have 'metadata' field"


class TestRetrievalTool:
    """Test retrieval_tool for knowledge base queries."""
    
    def test_retrieval_tool_invocation(self):
        """Test that retrieval_tool can be invoked."""
        result = retrieval_tool.invoke({
            "query": "model architecture",
            "collection": "general",
            "limit": 3
        })
        assert result is not None
        
    def test_retrieval_tool_returns_list(self):
        """Test that retrieval_tool returns a list."""
        result = retrieval_tool.invoke({
            "query": "training techniques",
            "collection": "ml_papers",
            "limit": 5
        })
        assert isinstance(result, list), "retrieval_tool should return a list"
        
    def test_retrieval_result_structure(self):
        """Test that retrieval results have required fields."""
        result = retrieval_tool.invoke({
            "query": "optimization",
            "collection": "general",
            "limit": 2
        })
        if result:  # If we get results
            for item in result:
                assert isinstance(item, dict), "Each result should be a dict"
                assert "id" in item, "Result should have 'id' field"
                assert "content" in item, "Result should have 'content' field"
                assert "similarity" in item, "Result should have 'similarity' field"
                assert "source" in item, "Result should have 'source' field"


# ===== TOOL REGISTRY TESTS =====

class TestToolRegistry:
    """Test that tools are properly registered in TOOLS list."""
    
    def test_tools_available(self):
        """Test that TOOLS list is populated."""
        assert TOOLS is not None, "TOOLS should be defined"
        assert len(TOOLS) > 0, "TOOLS should contain at least one tool"
        
    def test_research_tools_in_registry(self):
        """Test that research tools are in the registry."""
        tool_names = {tool.name for tool in TOOLS}
        assert "search_tool" in tool_names, "search_tool should be in TOOLS"
        assert "fetch_tool" in tool_names, "fetch_tool should be in TOOLS"
        assert "retrieval_tool" in tool_names, "retrieval_tool should be in TOOLS"
        
    def test_tool_descriptions(self):
        """Test that tools have descriptions."""
        for tool in TOOLS:
            assert tool.name is not None, "Tool should have a name"
            assert tool.description is not None, "Tool should have a description"
            assert len(tool.description) > 0, "Tool description should not be empty"
            

# ===== INTEGRATION TESTS =====

class TestToolIntegration:
    """Test tools working together in sequence."""
    
    def test_search_then_fetch_workflow(self):
        """Test realistic workflow: search for papers, then fetch one."""
        # Search
        search_results = search_tool.invoke({
            "query": "attention mechanisms",
            "source": "arxiv",
            "limit": 2
        })
        assert isinstance(search_results, list)
        
        # If we have results, try to fetch one
        if search_results and len(search_results) > 0:
            url = search_results[0].get("url")
            if url:
                fetch_result = fetch_tool.invoke({
                    "url": url,
                    "max_length": 5000
                })
                assert isinstance(fetch_result, dict)
    
    def test_tools_handle_edge_cases(self):
        """Test tools with edge case inputs."""
        # Empty query
        result1 = search_tool.invoke({
            "query": "",
            "source": "web",
            "limit": 1
        })
        assert result1 is not None
        
        # Large limit
        result2 = retrieval_tool.invoke({
            "query": "test",
            "collection": "general",
            "limit": 100
        })
        assert isinstance(result2, list)
        
        # Bad URL
        result3 = fetch_tool.invoke({
            "url": "not-a-real-url",
            "max_length": 1000
        })
        assert result3 is not None


# ===== PYTEST FIXTURES =====

@pytest.fixture
def sample_query():
    """Provide a sample research query."""
    return "machine learning optimization"


@pytest.fixture
def sample_url():
    """Provide a sample URL."""
    return "https://example.com/research-paper"


# ===== STANDALONE TEST RUNNER =====

if __name__ == "__main__":
    # Run tests with pytest programmatically
    pytest.main([__file__, "-v", "--tb=short"])
