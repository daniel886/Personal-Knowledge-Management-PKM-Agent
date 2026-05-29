"""Iter 6 — Pydantic v2 schema roundtrips & validation."""
from __future__ import annotations

import json
from datetime import datetime

import pytest
from pydantic import ValidationError

from models.schemas import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResult,
    ReviewRequest,
    SearchRequest,
    SearchResultItem,
    SourceType,
)


def test_ingest_request_url():
    req = IngestRequest(source_type=SourceType.WEB, url="https://x.io/a")
    payload = json.loads(req.model_dump_json())
    assert payload["source_type"] == "web"
    assert payload["url"].startswith("https://")


def test_ingest_request_invalid_url():
    with pytest.raises(ValidationError):
        IngestRequest(source_type=SourceType.WEB, url="not-a-url")


def test_search_request_defaults():
    r = SearchRequest(query="x")
    assert r.k == 5
    assert r.source_type is None


def test_chat_response_with_history():
    r = ChatResponse(
        answer="hello",
        sources=[SearchResultItem(id="1", title="t", snippet="s", score=0.9, source="x")],
        history=[ChatMessage(role="user", content="hi"), ChatMessage(role="assistant", content="ok")],
    )
    payload = r.model_dump()
    assert payload["history"][0]["role"] == "user"


def test_review_request_period_validation():
    r = ReviewRequest(period="weekly")
    assert r.period == "weekly"
    with pytest.raises(ValidationError):
        ReviewRequest(period="yearly")


def test_ingest_result_serialization():
    res = IngestResult(
        knowledge_id="k1",
        title="t",
        summary="...",
        source_type=SourceType.WEB,
        source="https://x",
    )
    js = json.loads(res.model_dump_json())
    assert js["chunks_indexed"] == 0
    assert "created_at" in js


def test_chat_request_validates_role():
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", history=[ChatMessage(role="other", content="x")])
