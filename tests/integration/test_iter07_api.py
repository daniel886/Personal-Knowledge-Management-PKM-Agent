"""Iter 7 — FastAPI app instantiation, /health, and OpenAPI listing."""
from __future__ import annotations

from fastapi.testclient import TestClient


def _make_client():
    from api.server import create_app
    app = create_app()
    return TestClient(app)


def test_health():
    with _make_client() as client:
        # Note: lifespan triggers DB init; ok with in-memory sqlite-like file
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "version" in body


def test_root_html():
    with _make_client() as client:
        r = client.get("/")
        assert r.status_code == 200
        # html or fallback
        assert "PKM" in r.text


def test_openapi_routes():
    with _make_client() as client:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        paths = r.json()["paths"].keys()
        assert "/api/ingest" in paths
        assert "/api/search" in paths
        assert "/api/chat" in paths
        assert "/api/review" in paths
        assert "/api/tasks" in paths


def test_search_route_validates_input():
    """Search route should reject malformed body."""
    with _make_client() as client:
        r = client.post("/api/search", json={})  # missing 'query'
        assert r.status_code == 422
