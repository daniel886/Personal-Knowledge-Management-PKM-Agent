"""Lightweight tests verifying core wiring without hitting external APIs."""
from __future__ import annotations

import json

from tools.tagger import _parse


def test_parse_tags_from_json() -> None:
    text = '["机器学习", "transformer", "论文"]'
    assert _parse(text) == ["机器学习", "transformer", "论文"]


def test_parse_tags_with_prefix() -> None:
    text = "  Sure, here are the tags: [\"AI\", \"#deep_learning\"] "
    out = _parse(text)
    assert "AI" in out
    assert "deep_learning" in out


def test_parse_tags_csv_fallback() -> None:
    out = _parse("机器学习, 神经网络, 计算机视觉")
    assert "机器学习" in out
    assert "神经网络" in out


def test_schemas_roundtrip() -> None:
    from models.schemas import IngestRequest, SourceType

    req = IngestRequest(source_type=SourceType.WEB, url="https://example.com")
    payload = json.loads(req.model_dump_json())
    assert payload["source_type"] == "web"
    assert payload["url"].startswith("https://example.com")
