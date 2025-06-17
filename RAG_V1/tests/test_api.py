import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_query_endpoint():
    response = client.post("/query", json={"question": "What is this system?", "session_id": "test1"})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "references" in data
    assert isinstance(data["references"], list)
