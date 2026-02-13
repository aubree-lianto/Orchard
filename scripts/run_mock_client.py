"""
Quick test script: verify MockModelClient works end-to-end with mock-server
Run inside orchestrator container: python scripts/run_mock_client.py
"""

import sys
import os

# Add repo root to path so imports work
sys.path.insert(0, '/app')

from inference.model_client import MockModelClient
from schemas.llm_schemas import ModelRequest, Message

def main():
    print("=" * 60)
    print("ORCHARD ORCHESTRATOR TEST")
    print("=" * 60)
    
    try:
        # Initialize client (reads MOCK_MODEL_SERVER_URL from .env)
        print("\n[1/3] Initializing MockModelClient...")
        client = MockModelClient()
        print(f"✅ Client initialized: {client.base_url}")
        
        # Create a test request
        print("\n[2/3] Creating test request...")
        req = ModelRequest(
            model="gpt-mock",
            messages=[Message(role="user", content="Hello, who are you?")]
        )
        print(f"✅ Request created: {req}")
        
        # Send request to mock-server
        print("\n[3/3] Sending request to mock-server...")
        resp = client.chat(req)
        print(f"✅ Received response:")
        print(f"   - Model: {resp.model}")
        print(f"   - Output: {resp.output_text}")
        print(f"   - Usage: {resp.usage}")
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED - END-TO-END FLOW WORKS!")
        print("=" * 60)
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("=" * 60)
        return 1

if __name__ == "__main__":
    exit(main())