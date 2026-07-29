"""
Basic tests. Run with: pytest tests/
Coming from TypeScript: pytest ≈ Jest, assert ≈ expect().toBe()
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_list_providers():
    response = client.get("/providers")
    assert response.status_code == 200
    data = response.json()
    assert "openai" in data["available"]


def test_rate_limit():
    from gateway.rate_limiter import RateLimiter
    limiter = RateLimiter(max_requests=5, window_seconds=60)

    for i in range(5):
        limiter.check("test-key")

    with pytest.raises(Exception):
        limiter.check("test-key")
