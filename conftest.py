# conftest.py  (project root)
import sys
import os
import threading
import time
import pytest
import uvicorn

repo_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)


@pytest.fixture(scope="session", autouse=True)
def mock_server():
    """Start mock LLM server for the test session, shut it down after."""
    from inference.mock_server import app

    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Wait for server to be ready
    import requests
    for _ in range(20):
        try:
            requests.get("http://localhost:8000/health", timeout=1)
            break
        except Exception:
            time.sleep(0.2)

    yield  # tests run here

    server.should_exit = True
    thread.join(timeout=5)
