import pytest
from fastapi.testclient import TestClient
from db.database import init_db
from main import app

init_db()
client = TestClient(app)

def test_list_models():
    response = client.get("/api/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert len(data["models"]) > 0

def test_chat_completions_proxy():
    payload = {
        "model": "wormhole-auto",
        "messages": [
            {"role": "user", "content": "How do I format a JSON string in Python?"}
        ]
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "id" in data
    assert "choices" in data
    assert len(data["choices"]) > 0
    assert "wormhole_metadata" in data
    assert "cost_savings_usd" in data["wormhole_metadata"]

def test_analytics_logs():
    response = client.get("/api/logs")
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "logs" in data

def test_dataset_export():
    response = client.get("/api/dataset/export?target=router")
    assert response.status_code == 200
    assert "application/x-jsonlines" in response.headers["content-type"]
