"""
Test LangChain tool definitions independently.
"""

import sys
import os

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from api.tools import TOOLS, echo_tool, calculator_tool, search_tool


def test_tools():
    print("=" * 60)
    print("TEST: LangChain Tool Definitions")
    print("=" * 60)
    
    # Test 1: Echo Tool
    print("\n[1] Testing Echo Tool")
    result = echo_tool.invoke({"message": "Hello, world!"})
    print(f"✅ Input: 'Hello, world!'")
    print(f"   Output: {result}")
    assert "Hello, world!" in result
    
    # Test 2: Calculator Tool
    print("\n[2] Testing Calculator Tool")
    result = calculator_tool.invoke({"expression": "2 + 2"})
    print(f"✅ Input: '2 + 2'")
    print(f"   Output: {result}")
    assert result == 4.0
    
    # Test 3: Search Tool
    print("\n[3] Testing Search Tool")
    result = search_tool.invoke({"query": "LangChain documentation"})
    print(f"✅ Input: 'LangChain documentation'")
    print(f"   Output: {result}")
    assert "LangChain documentation" in result
    
    # Test 4: Tool Schema
    print("\n[4] Tool Schema Visibility")
    print(f"✅ Available tools: {len(TOOLS)}")
    for tool in TOOLS:
        print(f"   - {tool.name}: {tool.description[:50]}...")
    
    print("\n" + "=" * 60)
    print("✅ ALL TOOL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_tools()