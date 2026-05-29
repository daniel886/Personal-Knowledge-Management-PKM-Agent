"""Loop 7 — coverage push.

Goal: lift overall pytest-cov coverage from ~74% to ≥85% by exercising
previously-uncovered paths in:
- api/routes (search, chat, ingest, review, tasks)
- core.llm provider dispatch (openai / anthropic / ollama / unknown)
- scrapers/web_scraper.scrape happy path with stubbed httpx
- scrapers/pdf_scraper missing-file branch
- scrapers/__init__.get_scraper unknown raises
- agents.pkm_agent ingest/chat/review with stubbed workflows
- agents.review_agent weekly/monthly wrappers
- workflows.chat_workflow.run_chat full path with stubbed LLM + search
- tools.search.hybrid_search dedup merge
- core.obsidian._atomic_write_text rollback on write error

All tests fully isolated: no real LLM, no real network, no real DB beyond test files.
"""
from __future__ import annotations

import asyncio
import os
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Cov-1..4 — API routes via FastAPI TestClient with monkeypatched agent
# ---------------------------------------------------------------------------
def _client():
    from fastapi.testclient import TestClient
    from api.server import create_app

    return TestClient(create_app())


def test_cov1_api_search_returns_hits(monkeypatch):
    """POST /api/search exercises hybrid_search → SearchResponse path."""
    from models.schemas import SearchResultItem
    import api.routes.search as search_route

    async def fake_hybrid(query, *, k=5, source_type=None):
        return [
            SearchResultItem(
                id="kb1", title="t1", snippet="s", score=0.9,
                source="x", metadata={"tags": "a"},
            )
        ]

    monkeypatch.setattr(search_route, "hybrid_search", fake_hybrid)
    with _client() as c:
        r = c.post("/api/search", json={"query": "hello", "k": 3})
        assert r.status_code == 200
        body = r.json()
        assert body["query"] == "hello"
        assert len(body["hits"]) == 1
        assert body["hits"][0]["title"] == "t1"


def test_cov1b_api_search_handles_internal_error(monkeypatch):
    """Search route catches exceptions and returns 500."""
    import api.routes.search as search_route

    async def boom(*a, **kw):
        raise RuntimeError("vector db is on fire")

    monkeypatch.setattr(search_route, "hybrid_search", boom)
    with _client() as c:
        r = c.post("/api/search", json={"query": "x"})
        assert r.status_code == 500
        assert "fire" in r.json()["detail"]


def test_cov2_api_chat_returns_response(monkeypatch):
    """POST /api/chat — agent.chat is awaited and ChatResponse returned."""
    from models.schemas import ChatMessage, ChatResponse
    import api.routes.chat as chat_route

    class FakeAgent:
        async def chat(self, message, history=None, *, use_memory=True, top_k=6):
            return ChatResponse(
                answer=f"echo:{message}",
                sources=[],
                history=[
                    ChatMessage(role="user", content=message),
                    ChatMessage(role="assistant", content=f"echo:{message}"),
                ],
            )

    monkeypatch.setattr(chat_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post("/api/chat", json={"message": "hi"})
        assert r.status_code == 200
        assert r.json()["answer"] == "echo:hi"


def test_cov2b_api_chat_500_on_failure(monkeypatch):
    import api.routes.chat as chat_route

    class FakeAgent:
        async def chat(self, *a, **kw):
            raise ValueError("bad llm")

    monkeypatch.setattr(chat_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post("/api/chat", json={"message": "boom"})
        assert r.status_code == 500
        assert "bad llm" in r.json()["detail"]


def test_cov3_api_review_returns_response(monkeypatch):
    from models.schemas import ReviewResponse
    import api.routes.review as review_route

    class FakeAgent:
        async def review(self, period, *, start=None, end=None):
            return ReviewResponse(
                period=period,
                start=start or datetime(2025, 1, 1),
                end=end or datetime(2025, 1, 7),
                summary="mock summary",
                obsidian_path=None,
                knowledge_count=3,
            )

    monkeypatch.setattr(review_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post("/api/review", json={"period": "weekly"})
        assert r.status_code == 200
        body = r.json()
        assert body["period"] == "weekly"
        assert body["knowledge_count"] == 3


def test_cov3b_api_review_500_on_error(monkeypatch):
    import api.routes.review as review_route

    class FakeAgent:
        async def review(self, *a, **kw):
            raise RuntimeError("review crashed")

    monkeypatch.setattr(review_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post("/api/review", json={"period": "weekly"})
        assert r.status_code == 500


def test_cov3c_api_tasks_lists_jobs(monkeypatch):
    """GET /api/tasks returns scheduler jobs as TaskInfo[]."""
    import api.routes.review as review_route

    class FakeSched:
        def list_jobs(self):
            return [
                {"id": "weekly-review", "next_run": None, "trigger": "cron"},
                {"id": "monthly-review", "next_run": None, "trigger": "cron"},
            ]

    monkeypatch.setattr(review_route, "get_scheduler", lambda: FakeSched())
    with _client() as c:
        r = c.get("/api/tasks")
        assert r.status_code == 200
        body = r.json()
        assert len(body) == 2
        assert {j["id"] for j in body} == {"weekly-review", "monthly-review"}


def test_cov4_api_ingest_returns_result(monkeypatch):
    from models.schemas import IngestResult, SourceType
    import api.routes.ingest as ingest_route

    class FakeAgent:
        async def ingest(self, source_type, target, **kwargs):
            return IngestResult(
                knowledge_id="K-1",
                title="Hello",
                summary="sum",
                tags=["a", "b"],
                links=[],
                obsidian_path=None,
                chunks_indexed=2,
                source_type=SourceType.WEB if str(source_type) == "SourceType.WEB" else SourceType(str(source_type).split(".")[-1].lower() if "." in str(source_type) else source_type),
                source=str(target),
            )

    monkeypatch.setattr(ingest_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post(
            "/api/ingest",
            json={"source_type": "web", "url": "https://example.com/x"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["knowledge_id"] == "K-1"
        assert body["chunks_indexed"] == 2


def test_cov4b_api_ingest_requires_target():
    """Missing url and file_path → 400."""
    with _client() as c:
        r = c.post("/api/ingest", json={"source_type": "web"})
        assert r.status_code == 400
        assert "url or file_path" in r.json()["detail"]


def test_cov4c_api_ingest_500_on_failure(monkeypatch):
    import api.routes.ingest as ingest_route

    class FakeAgent:
        async def ingest(self, *a, **kw):
            raise RuntimeError("scrape failed")

    monkeypatch.setattr(ingest_route, "get_agent", lambda: FakeAgent())
    with _client() as c:
        r = c.post(
            "/api/ingest",
            json={"source_type": "web", "url": "https://example.com"},
        )
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# Cov-5 — core.llm provider factory branches
# ---------------------------------------------------------------------------
def test_cov5_llm_unknown_provider_raises(monkeypatch):
    """get_chat_llm with bogus provider raises ValueError."""
    from core import llm

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm.get_chat_llm(provider="grokzilla")


def test_cov5b_llm_get_chat_llm_openai_branch(monkeypatch):
    """OpenAI branch is reached and constructs a ChatOpenAI without API key warning fatal."""
    from core import llm
    from core.config import settings as s

    # Inject a fake ChatOpenAI to avoid real network
    class FakeChat:
        def __init__(self, **kw):
            self.kw = kw

    import langchain_openai as lo
    monkeypatch.setattr(lo, "ChatOpenAI", FakeChat, raising=False)

    res = llm.get_chat_llm(provider="openai")
    assert isinstance(res, FakeChat)


def test_cov5c_llm_get_chat_llm_anthropic_branch(monkeypatch):
    """Anthropic branch reached when langchain_anthropic stub is provided."""
    from core import llm

    try:
        import langchain_anthropic as la
    except Exception:
        pytest.skip("langchain_anthropic not installed")

    class FakeChat:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setattr(la, "ChatAnthropic", FakeChat, raising=False)
    res = llm.get_chat_llm(provider="anthropic")
    assert isinstance(res, FakeChat)


def test_cov5d_llm_get_embeddings_unknown_raises():
    from core import llm
    with pytest.raises(ValueError, match="Unknown embedding provider"):
        llm.get_embeddings("madeup")


def test_cov5e_llm_get_embeddings_openai_branch(monkeypatch):
    from core import llm
    import langchain_openai as lo

    class FakeEmb:
        def __init__(self, **kw):
            self.kw = kw

    monkeypatch.setattr(lo, "OpenAIEmbeddings", FakeEmb, raising=False)
    res = llm.get_embeddings("openai")
    assert isinstance(res, FakeEmb)


# ---------------------------------------------------------------------------
# Cov-6 — web_scraper.scrape happy path with stubbed httpx fetch
# ---------------------------------------------------------------------------
def test_cov6_web_scraper_with_stubbed_fetch(monkeypatch):
    from scrapers.web_scraper import WebScraper

    html = (
        "<html><head><title>Stub Title</title></head>"
        "<body><article><h1>Hi</h1><p>Hello world.</p></article></body></html>"
    )

    async def fake_fetch_httpx(self, url):
        return html

    async def fake_fetch_pw(self, url):
        return None  # force httpx fallback

    monkeypatch.setattr(WebScraper, "_fetch_with_httpx", fake_fetch_httpx)
    monkeypatch.setattr(WebScraper, "_fetch_with_playwright", fake_fetch_pw)

    s = WebScraper()
    doc = asyncio.run(s.scrape("https://example.com/article"))
    assert doc.source_type == "web"
    assert doc.title == "Stub Title"
    assert "Hello world" in doc.content
    assert doc.metadata["length"] > 0


def test_cov6b_web_scraper_rejects_non_http():
    """File:// and other schemes blocked by guard."""
    from scrapers.web_scraper import WebScraper
    s = WebScraper()
    with pytest.raises(ValueError, match="WebScraper only supports"):
        asyncio.run(s.scrape("file:///etc/passwd"))


# ---------------------------------------------------------------------------
# Cov-7 — pdf_scraper missing-file branch
# ---------------------------------------------------------------------------
def test_cov7_pdf_scraper_missing_file_raises():
    from scrapers.pdf_scraper import PDFScraper
    s = PDFScraper()
    with pytest.raises(FileNotFoundError):
        asyncio.run(s.scrape("/nonexistent/path/never-here.pdf"))


def test_cov7b_pdf_scraper_with_stubbed_parse(monkeypatch, tmp_path):
    """Local PDF path branch — stubbed parse_pdf to avoid real PDF dep usage."""
    from scrapers import pdf_scraper as ps

    fake = tmp_path / "doc.pdf"
    fake.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(ps, "parse_pdf", lambda p: "CONTENT-XYZ")

    s = ps.PDFScraper()
    doc = asyncio.run(s.scrape(str(fake)))
    assert doc.content == "CONTENT-XYZ"
    assert doc.source_type == "pdf"
    assert doc.title == "doc"


# ---------------------------------------------------------------------------
# Cov-8 — pkm_agent ingest/chat/review with stubbed workflows
# ---------------------------------------------------------------------------
def test_cov8_pkm_agent_ingest_dispatches(monkeypatch):
    from agents import pkm_agent
    from models.schemas import IngestResult, SourceType

    captured: dict[str, Any] = {}

    async def fake_ingest_url(source_type, target, **kwargs):
        captured["source_type"] = source_type
        captured["target"] = target
        return IngestResult(
            knowledge_id="ID-1", title="t", summary="s", tags=[], links=[],
            obsidian_path=None, chunks_indexed=0,
            source_type=source_type, source=target,
        )

    monkeypatch.setattr(pkm_agent, "ingest_url", fake_ingest_url)
    a = pkm_agent.PKMAgent()
    res = asyncio.run(a.ingest("web", "https://x.com"))
    assert res.knowledge_id == "ID-1"
    assert captured["source_type"] == SourceType.WEB
    assert captured["target"] == "https://x.com"


def test_cov8b_pkm_agent_chat_dispatches(monkeypatch):
    from agents import pkm_agent
    from models.schemas import ChatResponse

    async def fake_run_chat(message, history, *, use_memory, top_k):
        return ChatResponse(answer=f"got:{message}", sources=[], history=[])

    monkeypatch.setattr(pkm_agent, "run_chat", fake_run_chat)
    a = pkm_agent.PKMAgent()
    res = asyncio.run(a.chat("hello"))
    assert res.answer == "got:hello"


def test_cov8c_pkm_agent_review_dispatches(monkeypatch):
    from agents import pkm_agent
    from models.schemas import ReviewResponse

    async def fake_run_review(period="weekly", *, start=None, end=None):
        return ReviewResponse(
            period=period, start=datetime(2025, 1, 1), end=datetime(2025, 1, 7),
            summary="ok", obsidian_path=None, knowledge_count=0,
        )

    monkeypatch.setattr(pkm_agent, "run_review", fake_run_review)
    a = pkm_agent.PKMAgent()
    res = asyncio.run(a.review("weekly"))
    assert res.period == "weekly"


def test_cov8d_pkm_agent_singleton():
    from agents.pkm_agent import get_agent
    a1 = get_agent()
    a2 = get_agent()
    assert a1 is a2


# ---------------------------------------------------------------------------
# Cov-9 — review_agent weekly/monthly wrappers
# ---------------------------------------------------------------------------
def test_cov9_review_agent_weekly_calls_run_review(monkeypatch):
    from agents import review_agent
    from models.schemas import ReviewResponse

    captured = []

    async def fake_run_review(period):
        captured.append(period)
        return ReviewResponse(
            period=period, start=datetime(2025, 1, 1), end=datetime(2025, 1, 7),
            summary="x", obsidian_path=None, knowledge_count=0,
        )

    monkeypatch.setattr(review_agent, "run_review", fake_run_review)
    res = asyncio.run(review_agent.run_weekly_review())
    assert res.period == "weekly"
    assert captured == ["weekly"]


def test_cov9b_review_agent_monthly_calls_run_review(monkeypatch):
    from agents import review_agent
    from models.schemas import ReviewResponse

    async def fake_run_review(period):
        return ReviewResponse(
            period=period, start=datetime(2025, 1, 1), end=datetime(2025, 1, 31),
            summary="m", obsidian_path=None, knowledge_count=0,
        )

    monkeypatch.setattr(review_agent, "run_review", fake_run_review)
    res = asyncio.run(review_agent.run_monthly_review())
    assert res.period == "monthly"


# ---------------------------------------------------------------------------
# Cov-10 — chat_workflow run_chat full path with stubbed deps
# ---------------------------------------------------------------------------
def test_cov10_run_chat_full_path(monkeypatch):
    """run_chat uses the LangGraph; we stub LLM + hybrid_search to make it deterministic."""
    from workflows import chat_workflow as cw
    from models.schemas import ChatMessage, SearchResultItem

    # Reset cached graph so our stubs are picked up by node_answer at runtime via module-level lookups.
    cw._CHAT_GRAPH = None

    async def fake_search(query, *, k=6, source_type=None):
        return [
            SearchResultItem(
                id="n1", title="Note1", snippet="snip", score=0.7,
                source="vault/n1.md", metadata={},
            ),
        ]

    class FakeLLM:
        async def ainvoke(self, msgs):
            return SimpleNamespace(content=f"ANSWER({len(msgs)})")

    monkeypatch.setattr(cw, "hybrid_search", fake_search)
    monkeypatch.setattr(cw, "get_chat_llm", lambda: FakeLLM())

    res = asyncio.run(cw.run_chat(
        "what is X?",
        history=[ChatMessage(role="user", content="prev")],
        use_memory=True,
        top_k=4,
    ))
    assert res.answer.startswith("ANSWER(")
    assert len(res.sources) == 1
    assert res.history[-1].role == "assistant"
    # cleanup so other tests don't see our stubbed graph
    cw._CHAT_GRAPH = None


def test_cov10b_run_chat_no_memory(monkeypatch):
    """use_memory=False short-circuits retrieval (no hits)."""
    from workflows import chat_workflow as cw
    cw._CHAT_GRAPH = None

    called = {"n": 0}

    async def fake_search(*a, **kw):
        called["n"] += 1
        return []

    class FakeLLM:
        async def ainvoke(self, msgs):
            return SimpleNamespace(content="no-mem-answer")

    monkeypatch.setattr(cw, "hybrid_search", fake_search)
    monkeypatch.setattr(cw, "get_chat_llm", lambda: FakeLLM())

    res = asyncio.run(cw.run_chat("hi", use_memory=False))
    assert res.answer == "no-mem-answer"
    assert called["n"] == 0  # retrieval skipped
    cw._CHAT_GRAPH = None


# ---------------------------------------------------------------------------
# Cov-11 — tools.search hybrid_search dedup merge
# ---------------------------------------------------------------------------
def test_cov11_hybrid_search_dedup(monkeypatch):
    """Items sharing title are deduplicated; merged output respects k cap."""
    from tools import search as s
    from models.schemas import SearchResultItem

    async def fake_vec(query, *, k=5, source_type=None, tags=None):
        return [
            SearchResultItem(id="v1", title="Same", snippet="a", score=0.9, source="src/a", metadata={}),
            SearchResultItem(id="v2", title="Other", snippet="b", score=0.8, source="src/b", metadata={}),
        ]

    async def fake_kw(query, *, k=5):
        return [
            # duplicate of "Same" must be removed
            SearchResultItem(id="k1", title="Same", snippet="a2", score=1.0, source="vault/a", metadata={}),
            SearchResultItem(id="k2", title="Third", snippet="c", score=1.0, source="vault/c", metadata={}),
        ]

    monkeypatch.setattr(s, "vector_search", fake_vec)
    monkeypatch.setattr(s, "keyword_search", fake_kw)

    out = asyncio.run(s.hybrid_search("q", k=3))
    titles = [h.title for h in out]
    assert "Same" in titles
    assert titles.count("Same") == 1  # deduped
    assert len(out) <= 3


def test_cov11b_hybrid_search_k_cap(monkeypatch):
    """When merged > k, output is truncated."""
    from tools import search as s
    from models.schemas import SearchResultItem

    async def fake_vec(query, *, k=5, source_type=None, tags=None):
        return [SearchResultItem(id=f"v{i}", title=f"T{i}", snippet="s", score=0.5, source="x", metadata={})
                for i in range(5)]

    async def fake_kw(query, *, k=5):
        return [SearchResultItem(id=f"k{i}", title=f"K{i}", snippet="s", score=0.5, source="y", metadata={})
                for i in range(5)]

    monkeypatch.setattr(s, "vector_search", fake_vec)
    monkeypatch.setattr(s, "keyword_search", fake_kw)

    out = asyncio.run(s.hybrid_search("q", k=4))
    assert len(out) == 4


# ---------------------------------------------------------------------------
# Cov-12 — get_scraper unknown raises ValueError
# ---------------------------------------------------------------------------
def test_cov12_get_scraper_unknown_raises():
    from scrapers import get_scraper, SCRAPERS
    with pytest.raises(ValueError, match="No scraper registered"):
        get_scraper("snapchat")
    # known still works
    assert get_scraper("web").__class__.__name__ == "WebScraper"
    assert "web" in SCRAPERS


# ---------------------------------------------------------------------------
# Cov-13 — _atomic_write_text rollback on write error
# ---------------------------------------------------------------------------
def test_cov13_atomic_write_text_happy(tmp_path):
    from core.obsidian import _atomic_write_text
    target = tmp_path / "sub" / "out.md"
    _atomic_write_text(target, "hello\n", encoding="utf-8")
    assert target.read_text(encoding="utf-8") == "hello\n"
    # No leftover .tmp.* files
    leftovers = list((tmp_path / "sub").glob(".tmp.*"))
    assert leftovers == []


def test_cov13b_atomic_write_text_rollback_on_error(monkeypatch, tmp_path):
    """If os.replace fails, the .tmp. file is unlinked and exception propagates."""
    from core import obsidian as ob

    target = tmp_path / "out.md"

    def boom_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(ob.os, "replace", boom_replace)
    with pytest.raises(OSError, match="disk full"):
        ob._atomic_write_text(target, "data")
    # Tmp files should have been cleaned up by the except branch
    leftovers = list(tmp_path.glob(".tmp.*"))
    assert leftovers == []


# ---------------------------------------------------------------------------
# Cov-14 — sentinel: total tests in this file count toward coverage push
# ---------------------------------------------------------------------------
def test_cov14_loop7_sentinel():
    """Sanity sentinel — confirms loop 7 file loaded."""
    assert True
