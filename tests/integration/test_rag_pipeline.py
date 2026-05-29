"""Tests for tools/rag_pipeline + RAG-enabled chat workflow.

Covers:
- _approx_tokens / _history_tokens (CJK + ASCII)
- _format_history empty/non-empty
- _parse_subqueries: pure JSON, fenced JSON, regex fallback, garbage,
  oversized items, duplicate handling, fallback inclusion
- rewrite_query: max_n=1 short-circuit, LLM success, LLM failure → fallback
- retrieve_with_rewrite: merge/dedupe across sub-queries, per-query failure,
  early stop on k cap
- compress_history: empty, under budget, over budget with LLM success,
  over budget with LLM failure (truncation only), too-few-messages short-circuit
- chat_workflow run_chat end-to-end with rewrite_query=True and
  compress_history=True (LLM patched, retrieval stubbed)
- ChatRequest/ChatResponse round-trip with new fields
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from models.schemas import ChatMessage, ChatRequest, ChatResponse, SearchResultItem
from tools import rag_pipeline as rp
from langchain_core.runnables import Runnable


# --------------------------------------------------------------------- helpers
def _hit(id_: str, title: str, score: float = 0.5) -> SearchResultItem:
    return SearchResultItem(
        id=id_, title=title, snippet="s", score=score, source="src"
    )


class _FakeLLM(Runnable):
    """Minimal Runnable LLM stub so ``prompt | llm`` produces a valid chain."""

    def __init__(self, payload: str, *, raise_exc: Exception | None = None):
        self._payload = payload
        self._raise = raise_exc

    def invoke(self, _input, _config=None, **_kw):  # pragma: no cover - sync path
        if self._raise:
            raise self._raise
        return SimpleNamespace(content=self._payload)

    async def ainvoke(self, _input, _config=None, **_kw):
        if self._raise:
            raise self._raise
        return SimpleNamespace(content=self._payload)


# ===================================================================== tokens
def test_rp1_approx_tokens_empty():
    assert rp._approx_tokens("") == 0


def test_rp2_approx_tokens_ascii():
    # "hello world" -> 11 chars, max(1, 11//4) = 2
    assert rp._approx_tokens("hello world") == 2


def test_rp3_approx_tokens_cjk():
    # 5 CJK chars + max(1, 0//4)=1 -> 6 (the helper always adds at least 1)
    assert rp._approx_tokens("你好世界呀") == 6


def test_rp4_approx_tokens_mixed():
    # 2 CJK + 4 ASCII -> 2 + max(1, 4//4) = 3
    assert rp._approx_tokens("你好abcd") == 3


def test_rp5_history_tokens_sum():
    h = [ChatMessage(role="user", content="你好"),
         ChatMessage(role="assistant", content="hello world!!")]
    # m1: 2 CJK + max(1,0)=1 = 3
    # m2: 0 CJK + max(1, 13//4)=3 = 3
    assert rp._history_tokens(h) == 6


# ====================================================== format / parse subqs
def test_rp6_format_history_empty():
    assert rp._format_history([]) == "(empty)"


def test_rp7_format_history_truncates():
    h = [ChatMessage(role="user", content=f"m{i}") for i in range(10)]
    out = rp._format_history(h, max_msgs=3)
    assert out.count("\n") == 2  # 3 messages -> 2 separators
    assert "m9" in out and "m7" in out and "m0" not in out


def test_rp8_parse_subqs_plain_json():
    out = rp._parse_subqueries('["a", "b", "c"]', fallback="q", max_n=3)
    # fallback "q" prepended, then cap at max_n=3
    assert out[0] == "q"
    assert "a" in out and "b" in out
    assert len(out) == 3


def test_rp9_parse_subqs_fenced_json():
    out = rp._parse_subqueries('```json\n["x","y"]\n```', fallback="q", max_n=4)
    assert out[0] == "q"
    assert "x" in out and "y" in out


def test_rp10_parse_subqs_regex_fallback():
    # malformed JSON but contains bracketed list
    out = rp._parse_subqueries('prefix ["foo","bar"] suffix', fallback="q", max_n=3)
    assert "foo" in out and "bar" in out


def test_rp11_parse_subqs_garbage_returns_fallback():
    out = rp._parse_subqueries("totally not json", fallback="origQ", max_n=3)
    assert out == ["origQ"]


def test_rp12_parse_subqs_dedup_and_size_filter():
    huge = "x" * 300  # oversized -> filtered
    out = rp._parse_subqueries(
        f'["dup", "dup", "{huge}", "ok"]', fallback="orig", max_n=5
    )
    # "dup" once + "ok" + fallback prepended = 3 entries
    assert "dup" in out
    assert "ok" in out
    assert "orig" in out
    assert huge not in out


def test_rp13_parse_subqs_non_list_top_level():
    # JSON object, not list -> regex fallback also fails -> fallback only
    out = rp._parse_subqueries('{"k":"v"}', fallback="origQ", max_n=2)
    assert out == ["origQ"]


# ============================================================== rewrite_query
def test_rp14_rewrite_query_max_n_one_short_circuits():
    out = asyncio.run(rp.rewrite_query("question?", max_n=1))
    assert out == ["question?"]


def test_rp15_rewrite_query_success(monkeypatch):
    fake = _FakeLLM('["a", "b"]')
    monkeypatch.setattr(rp, "get_chat_llm", lambda **_k: fake)
    out = asyncio.run(rp.rewrite_query("origQ", max_n=3))
    assert out[0] == "origQ"
    assert "a" in out and "b" in out


def test_rp16_rewrite_query_llm_failure_fallback(monkeypatch):
    def boom(**_k):
        raise RuntimeError("llm offline")
    monkeypatch.setattr(rp, "get_chat_llm", boom)
    out = asyncio.run(rp.rewrite_query("origQ", max_n=3))
    assert out == ["origQ"]


# ======================================================= retrieve_with_rewrite
def test_rp17_retrieve_merge_and_dedupe(monkeypatch):
    # First sub-query returns h1,h2; second returns h2 (dup) + h3
    calls = {"n": 0}

    async def fake_rewrite(q, *, history=None, max_n=3):
        return ["q1", "q2"]

    async def fake_hybrid(query, *, k=6, source_type=None):
        calls["n"] += 1
        if query == "q1":
            return [_hit("a", "A"), _hit("b", "B")]
        if query == "q2":
            return [_hit("b", "B"), _hit("c", "C")]
        return []

    monkeypatch.setattr(rp, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    hits, subs = asyncio.run(rp.retrieve_with_rewrite("q", k=10, max_subqueries=2))
    assert [h.id for h in hits] == ["a", "b", "c"]
    assert subs == ["q1", "q2"]
    assert calls["n"] == 2


def test_rp18_retrieve_subquery_failure_is_skipped(monkeypatch):
    async def fake_rewrite(q, *, history=None, max_n=3):
        return ["q1", "q2"]

    async def fake_hybrid(query, *, k=6, source_type=None):
        if query == "q1":
            raise RuntimeError("vector store down")
        return [_hit("z", "Z")]

    monkeypatch.setattr(rp, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    hits, _subs = asyncio.run(rp.retrieve_with_rewrite("q", k=5))
    assert [h.id for h in hits] == ["z"]


def test_rp19_retrieve_caps_at_k(monkeypatch):
    async def fake_rewrite(q, *, history=None, max_n=3):
        return ["q1", "q2", "q3"]

    async def fake_hybrid(query, *, k=6, source_type=None):
        return [_hit(f"{query}-{i}", "T") for i in range(5)]

    monkeypatch.setattr(rp, "rewrite_query", fake_rewrite)
    monkeypatch.setattr(rp, "hybrid_search", fake_hybrid)
    hits, _subs = asyncio.run(rp.retrieve_with_rewrite("q", k=3))
    assert len(hits) == 3


# ================================================================ compression
def test_rp20_compress_empty_returns_empty():
    out, summary = asyncio.run(rp.compress_history([]))
    assert out == [] and summary is None


def test_rp21_compress_under_budget_noop():
    h = [ChatMessage(role="user", content="hi")]
    out, summary = asyncio.run(rp.compress_history(h, token_budget=1000))
    assert out == h and summary is None


def test_rp22_compress_too_few_messages_noop():
    # Tokens > budget but only 2 msgs and keep_recent=4 -> noop
    h = [
        ChatMessage(role="user", content="你" * 200),
        ChatMessage(role="assistant", content="好" * 200),
    ]
    out, summary = asyncio.run(rp.compress_history(h, token_budget=50, keep_recent=4))
    assert out == h and summary is None


def test_rp23_compress_success_summarises(monkeypatch):
    fake = _FakeLLM("早期讨论了X和Y。")
    monkeypatch.setattr(rp, "get_chat_llm", lambda **_k: fake)
    h = [ChatMessage(role="user" if i % 2 == 0 else "assistant",
                     content="内容" * 50)
         for i in range(10)]
    out, summary = asyncio.run(
        rp.compress_history(h, token_budget=50, keep_recent=2)
    )
    assert summary == "早期讨论了X和Y。"
    assert out[0].role == "system"
    assert "previously:" in out[0].content
    assert len(out) == 1 + 2  # summary + 2 tail


def test_rp24_compress_llm_failure_truncates_only(monkeypatch):
    fake = _FakeLLM("", raise_exc=RuntimeError("boom"))
    monkeypatch.setattr(rp, "get_chat_llm", lambda **_k: fake)
    h = [ChatMessage(role="user", content="内容" * 50) for _ in range(8)]
    out, summary = asyncio.run(
        rp.compress_history(h, token_budget=50, keep_recent=2)
    )
    assert summary is None
    assert len(out) == 2  # just the tail, no summary message
    assert all(m.role == "user" for m in out)


# ================================================== chat workflow integration
def test_rp25_run_chat_with_rewrite_and_compress(monkeypatch):
    """End-to-end: run_chat with both RAG toggles on, all LLM/IO patched."""
    import workflows.chat_workflow as cw

    async def fake_compress(history, *, token_budget=1200, keep_recent=4):
        # Pretend we compressed 6 messages to 1 summary + 2 tail
        return (
            [ChatMessage(role="system", content="previously: gist")] + history[-2:],
            "gist",
        )

    async def fake_retrieve_with_rewrite(question, *, history=None, k=6,
                                         max_subqueries=3, source_type=None):
        return ([_hit("a", "A"), _hit("b", "B")], ["sub1", "sub2"])

    async def fake_llm_ainvoke(self, msgs):
        return SimpleNamespace(content="final answer")

    monkeypatch.setattr(cw, "compress_history", fake_compress)
    monkeypatch.setattr(cw, "retrieve_with_rewrite", fake_retrieve_with_rewrite)

    class _LLM:
        async def ainvoke(self, msgs):
            return SimpleNamespace(content="final answer")

    monkeypatch.setattr(cw, "get_chat_llm", lambda **_k: _LLM())
    # Force rebuild of the cached graph so patches take effect
    cw._CHAT_GRAPH = None

    history = [ChatMessage(role="user" if i % 2 == 0 else "assistant",
                           content=f"m{i}") for i in range(6)]
    resp = asyncio.run(cw.run_chat(
        "今天天气如何？",
        history=history,
        rewrite_query=True,
        compress_history=True,
        history_token_budget=50,
        max_subqueries=2,
    ))
    assert isinstance(resp, ChatResponse)
    assert resp.answer == "final answer"
    assert resp.subqueries == ["sub1", "sub2"]
    assert resp.compressed_history_summary == "gist"
    assert [s.id for s in resp.sources] == ["a", "b"]
    # Reset cached graph for downstream tests
    cw._CHAT_GRAPH = None


def test_rp26_run_chat_default_path_no_rag(monkeypatch):
    """run_chat with default flags should call hybrid_search, not rewrite."""
    import workflows.chat_workflow as cw

    called = {"hybrid": 0, "rewrite": 0}

    async def fake_hybrid(query, *, k=6, source_type=None):
        called["hybrid"] += 1
        return [_hit("h", "H")]

    async def fake_rewrite(*a, **kw):
        called["rewrite"] += 1
        return ([_hit("x", "X")], ["x"])

    class _LLM:
        async def ainvoke(self, msgs):
            return SimpleNamespace(content="ok")

    monkeypatch.setattr(cw, "hybrid_search", fake_hybrid)
    monkeypatch.setattr(cw, "retrieve_with_rewrite", fake_rewrite)
    monkeypatch.setattr(cw, "get_chat_llm", lambda **_k: _LLM())
    cw._CHAT_GRAPH = None

    resp = asyncio.run(cw.run_chat("hi"))
    assert resp.answer == "ok"
    assert called["hybrid"] == 1 and called["rewrite"] == 0
    assert resp.subqueries == ["hi"]
    assert resp.compressed_history_summary is None
    cw._CHAT_GRAPH = None


# ============================================================ schemas updates
def test_rp27_chat_request_validates_new_fields():
    req = ChatRequest(message="hi", rewrite_query=True, compress_history=True,
                      history_token_budget=500, max_subqueries=4)
    assert req.rewrite_query is True
    assert req.history_token_budget == 500
    assert req.max_subqueries == 4


def test_rp28_chat_request_rejects_out_of_range_budget():
    with pytest.raises(Exception):
        ChatRequest(message="hi", history_token_budget=100)  # < ge=200
    with pytest.raises(Exception):
        ChatRequest(message="hi", max_subqueries=10)  # > le=5


def test_rp29_chat_response_carries_diagnostics():
    resp = ChatResponse(
        answer="a",
        sources=[],
        history=[],
        subqueries=["a", "b"],
        compressed_history_summary="gist",
    )
    assert resp.subqueries == ["a", "b"]
    assert resp.compressed_history_summary == "gist"


# =========================================================== api integration
def test_rp30_api_chat_accepts_rag_params(monkeypatch):
    """POST /api/chat with rewrite_query=true is forwarded to agent.chat."""
    from fastapi.testclient import TestClient
    import api.routes.chat as chat_route
    from api.server import app

    captured = {}

    class FakeAgent:
        async def chat(self, message, history=None, *, use_memory=True, top_k=6,
                       **kwargs):
            captured.update(kwargs)
            return ChatResponse(
                answer=f"a:{message}", sources=[], history=[],
                subqueries=kwargs.get("rewrite_query") and ["sq"] or [],
            )

    monkeypatch.setattr(chat_route, "get_agent", lambda: FakeAgent())
    with TestClient(app) as c:
        r = c.post("/api/chat", json={
            "message": "hi",
            "rewrite_query": True,
            "compress_history": True,
            "history_token_budget": 400,
            "max_subqueries": 2,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["answer"] == "a:hi"
        assert body["subqueries"] == ["sq"]
    assert captured["rewrite_query"] is True
    assert captured["compress_history"] is True
    assert captured["history_token_budget"] == 400
    assert captured["max_subqueries"] == 2
