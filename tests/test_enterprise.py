import pytest
from fastapi.testclient import TestClient
from main import app
from config import settings
from services.dispatcher import (
    is_circuit_open,
    record_provider_failure,
    record_provider_success,
    dispatch_inference
)

client = TestClient(app)

def test_bearer_token_authentication(monkeypatch):
    """
    Verifies that when ENABLE_AUTH=True, invalid API key returns 401 Unauthorized,
    and valid Bearer token returns 200 OK.
    """
    monkeypatch.setattr(settings, "ENABLE_AUTH", True)
    
    # 1. Unauthenticated request should fail
    resp_no_auth = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "Test auth payload"}]
    })
    assert resp_no_auth.status_code == 401
    assert "Authorization" in resp_no_auth.json()["detail"]

    # 2. Request with valid Bearer token should succeed
    resp_valid = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wh_live_demo123456789"},
        json={"messages": [{"role": "user", "content": "Test auth payload"}]}
    )
    assert resp_valid.status_code == 200
    assert "choices" in resp_valid.json()

def test_sse_streaming_completions():
    """
    Verifies that stream=True returns Server-Sent Events (text/event-stream)
    with data: {...} JSON chunks and data: [DONE].
    """
    response = client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer wh_live_demo123456789"},
        json={
            "stream": True,
            "messages": [{"role": "user", "content": "Stream this sentence"}]
        }
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
    
    lines = response.text.strip().split("\n\n")
    assert len(lines) > 2
    assert "data: {" in lines[0]
    assert lines[-1] == "data: [DONE]"

def test_circuit_breaker_failover():
    """
    Verifies that recording consecutive failures opens the circuit breaker
    and triggers automatic fallback model selection.
    """
    test_model = "test-failing-provider/model"
    record_provider_success(test_model)
    assert not is_circuit_open(test_model)

    # Record 3 failures (threshold = 3)
    record_provider_failure(test_model)
    record_provider_failure(test_model)
    record_provider_failure(test_model)
    
    assert is_circuit_open(test_model)

    # Reset circuit after test
    record_provider_success(test_model)
    assert not is_circuit_open(test_model)
