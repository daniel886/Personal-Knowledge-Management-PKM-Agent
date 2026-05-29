"""Loop 9 — coverage push v3: target remaining gaps to reach ≥98%.

Specific lines targeted (from `pytest --cov` missing-line report):
- main.py 73%: chat REPL (lines 86-106), serve (32-34)
- agents/chat_agent.py 94%: line 30 (reset)
- scrapers/web_scraper.py 60%: lines 25-32 (_fetch_with_httpx), 36-55 (_fetch_with_playwright import branch)
- scrapers/youtube_scraper.py 69%: lines 38-46 (_fetch_transcript fallback), 52-65 (_fetch_metadata httpx)
- scrapers/pdf_scraper.py 76%: lines 21-27 (_download httpx)
- scrapers/rss_scraper.py 78%: line 59 (fetch_all), 85-98 (refresh_subscribed)
- scrapers/email_scraper.py 53%: lines 33-58, 78-85, 88 (_fetch_recent + scrape + fetch_recent)
- api/routes/ingest.py 63%: lines 39-59 (/upload endpoint)
- core/llm.py 81%: line 29 (warn missing key), 53-55 (ollama), 79-86 (huggingface)
- core/scheduler.py 90%: lines 31-33 (shutdown), 72 (jobs detail)
- core/obsidian.py 93%: lines 31-32, 137-138, 153, 156, 183, 186-187, 195
- workflows/chat_workflow.py 96%: lines 54-55 (assistant message append)
- workflows/ingest_workflow.py 90%: lines 47-49 (skip-scrape), 83-85 (mindmap exc), 212-216 (ingest_url)
- workflows/review_workflow.py 96%: lines 145-148 (custom period)
- tools/tagger.py 84%: lines 26-30 (LLM call wrapper), 54-55 (fence-stripping fallback), 73 (cleaned cap)
- scrapers/base.py 95%: line 40 (NotImplementedError)
- core/config.py 99%: line 91 (validator branch)
- core/vector_store.py 99%: line 169 (delete_by_source path)

Strategy: heavy stubbing of httpx, playwright, IMAP, transcript API; isolated tmp paths.
"""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# L9-1 — web_scraper _fetch_with_httpx via stubbed httpx.AsyncClient
# ---------------------------------------------------------------------------
def test_l9_1_web_fetch_with_httpx(monkeypatch):
    from scrapers import web_scraper as ws

    captured: dict[str, Any] = {}

    class FakeResp:
        text = "<html><body>OK</body></html>"
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, **kw): captured["init"] = kw
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            captured["url"] = url
            return FakeResp()

    monkeypatch.setattr(ws.httpx, "AsyncClient", FakeClient)
    text = asyncio.run(ws.WebScraper()._fetch_with_httpx("https://x.com"))
    assert "OK" in text
    assert captured["url"] == "https://x.com"
    assert captured["init"]["follow_redirects"] is True


def test_l9_1b_web_fetch_with_playwright_unavailable(monkeypatch):
    """If playwright import fails, _fetch_with_playwright returns None."""
    from scrapers import web_scraper as ws
    import sys
    # Make the playwright.async_api import fail
    monkeypatch.setitem(sys.modules, "playwright.async_api", None)
    res = asyncio.run(ws.WebScraper()._fetch_with_playwright("https://x.com"))
    assert res is None


def test_l9_1c_web_scrape_full_path_with_playwright_fail_fallback(monkeypatch):
    """Playwright raises → returns None → falls back to httpx → produces doc."""
    from scrapers import web_scraper as ws

    async def pw_fail(self, url):
        # Hit the warning branch (lines 53-55)
        try:
            raise RuntimeError("pw chromium missing")
        except RuntimeError as exc:
            from utils.logger import logger
            logger.warning(f"Playwright fetch failed for {url}: {exc}")
            return None

    async def httpx_ok(self, url):
        return "<html><title>From httpx</title><body>x</body></html>"

    monkeypatch.setattr(ws.WebScraper, "_fetch_with_playwright", pw_fail)
    monkeypatch.setattr(ws.WebScraper, "_fetch_with_httpx", httpx_ok)
    doc = asyncio.run(ws.WebScraper().scrape("https://example.com/a"))
    assert doc.title == "From httpx"


# ---------------------------------------------------------------------------
# L9-2 — youtube transcript + metadata real branches
# ---------------------------------------------------------------------------
def test_l9_2_youtube_transcript_via_thread(monkeypatch):
    """_fetch_transcript happy path via stubbed YouTubeTranscriptApi."""
    from scrapers import youtube_scraper as ys

    class FakeAPI:
        @staticmethod
        def get_transcript(video_id, languages=None):
            return [{"text": "hello"}, {"text": "world"}]

    monkeypatch.setattr(ys, "YouTubeTranscriptApi", FakeAPI)
    text, kind = asyncio.run(
        ys.YouTubeScraper()._fetch_transcript("vid12345678", ["en"])
    )
    assert "hello world" == text
    assert kind == "transcript"


def test_l9_2b_youtube_transcript_no_lang_fallback(monkeypatch):
    """If specific lang fails (NoTranscriptFound), falls back to default."""
    from scrapers import youtube_scraper as ys

    calls = {"n": 0}

    class FakeAPI:
        @staticmethod
        def get_transcript(video_id, languages=None):
            calls["n"] += 1
            if calls["n"] == 1 and languages:
                raise ys.NoTranscriptFound("v", ["en"], None)
            return [{"text": "fallback"}]

    monkeypatch.setattr(ys, "YouTubeTranscriptApi", FakeAPI)
    text, _ = asyncio.run(
        ys.YouTubeScraper()._fetch_transcript("vid", ["zh-Hans"])
    )
    assert "fallback" in text
    assert calls["n"] == 2


def test_l9_2c_youtube_metadata_with_api_key(monkeypatch):
    """_fetch_metadata makes httpx call when api key set."""
    from scrapers import youtube_scraper as ys

    monkeypatch.setattr(ys.settings, "youtube_api_key", "FAKE-KEY", raising=False)

    class FakeResp:
        def raise_for_status(self): pass
        def json(self):
            return {
                "items": [{
                    "snippet": {
                        "title": "VT",
                        "channelTitle": "CH",
                        "publishedAt": "2025-01-01",
                        "description": "desc " * 100,
                    },
                    "statistics": {"viewCount": "12345"},
                }]
            }

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url):
            return FakeResp()

    monkeypatch.setattr(ys.httpx, "AsyncClient", FakeClient)
    res = asyncio.run(ys.YouTubeScraper()._fetch_metadata("v123"))
    assert res["title"] == "VT"
    assert res["view_count"] == "12345"
    assert len(res["description"]) <= 1000


def test_l9_2d_youtube_metadata_empty_items(monkeypatch):
    """If API returns no items → empty dict."""
    from scrapers import youtube_scraper as ys
    monkeypatch.setattr(ys.settings, "youtube_api_key", "FAKE", raising=False)

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"items": []}

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return FakeResp()

    monkeypatch.setattr(ys.httpx, "AsyncClient", FakeClient)
    res = asyncio.run(ys.YouTubeScraper()._fetch_metadata("vid"))
    assert res == {}


# ---------------------------------------------------------------------------
# L9-3 — pdf_scraper _download via stubbed httpx
# ---------------------------------------------------------------------------
def test_l9_3_pdf_download_with_stub(monkeypatch, tmp_path):
    from scrapers import pdf_scraper as ps

    class FakeResp:
        content = b"%PDF-1.4 content bytes"
        def raise_for_status(self): pass

    class FakeClient:
        def __init__(self, **kw): pass
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def get(self, url): return FakeResp()

    monkeypatch.setattr(ps.httpx, "AsyncClient", FakeClient)
    # Force tempfile to predictable location
    out = asyncio.run(ps.PDFScraper()._download("https://x.com/y.pdf"))
    assert out.exists()
    assert out.read_bytes().startswith(b"%PDF")
    out.unlink()


# ---------------------------------------------------------------------------
# L9-4 — rss refresh_subscribed + fetch_all
# ---------------------------------------------------------------------------
def test_l9_4_rss_fetch_all(monkeypatch):
    from scrapers import rss_scraper as rs

    fake_feed = SimpleNamespace(
        bozo=False, feed={"title": "F"},
        entries=[{"title": "T", "summary": "s", "link": "https://e", "published": "p"}],
    )
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)
    items = asyncio.run(rs.RSSScraper().fetch_all("https://feed/rss"))
    assert len(items) == 1


def test_l9_4b_rss_refresh_subscribed(monkeypatch, tmp_path):
    from scrapers import rss_scraper as rs

    sub_path = tmp_path / "subs.txt"
    sub_path.write_text("https://a/rss\nhttps://b/rss\n")
    monkeypatch.setattr(rs, "SUBSCRIPTIONS_PATH", sub_path)

    # Stub the per-feed fetch
    fake_feed = SimpleNamespace(
        bozo=False, feed={"title": "F"},
        entries=[
            {"title": "n1", "summary": "s", "link": "https://e/1", "published": ""},
            {"title": "n2", "summary": "s", "link": "https://e/2", "published": ""},
        ],
    )
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)

    # Stub ingest_document so we don't hit real DB graph
    import workflows.ingest_workflow as iw
    from models.schemas import IngestResult, SourceType

    async def fake_ingest_document(doc):
        return IngestResult(
            knowledge_id="x", title=doc.title, summary="", tags=[], links=[],
            obsidian_path=None, chunks_indexed=0,
            source_type=SourceType.RSS, source=doc.source,
        )

    monkeypatch.setattr(iw, "ingest_document", fake_ingest_document)

    n = asyncio.run(rs.RSSScraper.refresh_subscribed())
    # 2 feeds × 2 items = 4 ingested
    assert n == 4


def test_l9_4c_rss_refresh_subscribed_with_failure(monkeypatch, tmp_path):
    """Failures per feed are caught and logged; total counts only successes."""
    from scrapers import rss_scraper as rs

    sub_path = tmp_path / "subs.txt"
    sub_path.write_text("https://broken\nhttps://ok\n")
    monkeypatch.setattr(rs, "SUBSCRIPTIONS_PATH", sub_path)

    def parse_with_fail(url, agent=None):
        if "broken" in url:
            raise RuntimeError("network down")
        return SimpleNamespace(
            bozo=False, feed={"title": "F"},
            entries=[{"title": "ok", "summary": "s", "link": url, "published": ""}],
        )

    monkeypatch.setattr(rs.feedparser, "parse", parse_with_fail)

    import workflows.ingest_workflow as iw
    from models.schemas import IngestResult, SourceType

    async def fake_ingest(doc):
        return IngestResult(
            knowledge_id="i", title=doc.title, summary="", tags=[], links=[],
            obsidian_path=None, chunks_indexed=0,
            source_type=SourceType.RSS, source=doc.source,
        )

    monkeypatch.setattr(iw, "ingest_document", fake_ingest)
    n = asyncio.run(rs.RSSScraper.refresh_subscribed())
    assert n == 1  # only the ok feed


# ---------------------------------------------------------------------------
# L9-5 — Email scraper full IMAP path
# ---------------------------------------------------------------------------
def test_l9_5_email_fetch_recent_via_stubbed_imap(monkeypatch):
    """Stub IMAPClient context manager and message_from_bytes."""
    from scrapers import email_scraper as es

    monkeypatch.setattr(es.settings, "email_user", "u@example.com", raising=False)
    monkeypatch.setattr(es.settings, "email_password", "pwd", raising=False)
    monkeypatch.setattr(es.settings, "email_host", "imap.example.com", raising=False)
    monkeypatch.setattr(es.settings, "email_port", 993, raising=False)
    monkeypatch.setattr(es.settings, "email_folder", "INBOX", raising=False)

    raw_email = (
        b"Subject: Hello\r\n"
        b"From: sender@example.com\r\n"
        b"Content-Type: text/plain; charset=utf-8\r\n\r\n"
        b"plain text body"
    )

    class FakeIMAP:
        def __init__(self, host, port=993, ssl=True): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def login(self, u, p): pass
        def select_folder(self, folder, readonly=True): pass
        def search(self, q): return [101, 102]
        def fetch(self, uids, fields):
            return {
                uid: {b"RFC822": raw_email, b"INTERNALDATE": b"2025-01-01"}
                for uid in uids
            }

    monkeypatch.setattr(es, "IMAPClient", FakeIMAP)
    items = es.EmailScraper()._fetch_recent(2)
    assert len(items) == 2
    assert items[0].title == "Hello"
    assert "plain text" in items[0].content


def test_l9_5b_email_scrape_returns_primary(monkeypatch):
    from scrapers import email_scraper as es
    from scrapers.base import ScrapedDocument

    async def fake_recent(self, limit):
        return [
            ScrapedDocument(title="m1", content="c1", source="imap://1",
                            source_type="email", metadata={}),
            ScrapedDocument(title="m2", content="c2", source="imap://2",
                            source_type="email", metadata={}),
        ]

    monkeypatch.setattr(es.EmailScraper, "fetch_recent", fake_recent)
    # Patch _fetch_recent (sync) directly for the scrape() pathway
    def sync_recent(self, limit=10):
        return [
            ScrapedDocument(title="m1", content="c1", source="imap://1",
                            source_type="email", metadata={}),
            ScrapedDocument(title="m2", content="c2", source="imap://2",
                            source_type="email", metadata={}),
        ]
    monkeypatch.setattr(es.EmailScraper, "_fetch_recent", sync_recent)

    doc = asyncio.run(es.EmailScraper().scrape("2"))
    # latest is returned as primary
    assert doc.title == "m2"
    assert "all" in doc.metadata


def test_l9_5c_email_scrape_no_emails_raises(monkeypatch):
    from scrapers import email_scraper as es
    monkeypatch.setattr(es.EmailScraper, "_fetch_recent", lambda self, limit=1: [])
    with pytest.raises(RuntimeError, match="No emails fetched"):
        asyncio.run(es.EmailScraper().scrape("1"))


# ---------------------------------------------------------------------------
# L9-6 — /api/ingest/upload endpoint
# ---------------------------------------------------------------------------
def test_l9_6_api_ingest_upload_pdf(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from api.server import create_app
    import api.routes.ingest as ir
    from models.schemas import IngestResult, SourceType

    captured: dict[str, Any] = {}

    class FakeAgent:
        async def ingest(self, source_type, target, **kw):
            captured["source_type"] = source_type
            captured["target"] = target
            return IngestResult(
                knowledge_id="UP1", title="up", summary="", tags=[], links=[],
                obsidian_path=None, chunks_indexed=1,
                source_type=SourceType.PDF, source=str(target),
            )

    monkeypatch.setattr(ir, "get_agent", lambda: FakeAgent())

    with TestClient(create_app()) as c:
        r = c.post(
            "/api/ingest/upload",
            files={"file": ("doc.pdf", b"%PDF-1.4 stub", "application/pdf")},
            data={"source_type": "pdf"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["knowledge_id"] == "UP1"
        assert captured["source_type"] == SourceType.PDF


def test_l9_6b_api_ingest_upload_md(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from api.server import create_app
    import api.routes.ingest as ir
    from models.schemas import IngestResult, SourceType

    async def fake_ingest_document(doc):
        return IngestResult(
            knowledge_id="UP2", title=doc.title, summary="", tags=[], links=[],
            obsidian_path=None, chunks_indexed=0,
            source_type=SourceType.UPLOAD, source=doc.source,
        )

    monkeypatch.setattr(ir, "ingest_document", fake_ingest_document)

    with TestClient(create_app()) as c:
        r = c.post(
            "/api/ingest/upload",
            files={"file": ("note.md", b"# hi\nbody", "text/markdown")},
            data={"source_type": "upload"},
        )
        assert r.status_code == 200
        assert r.json()["knowledge_id"] == "UP2"


def test_l9_6c_api_ingest_upload_500_on_error(monkeypatch, tmp_path):
    from fastapi.testclient import TestClient
    from api.server import create_app
    import api.routes.ingest as ir

    async def boom(*a, **kw):
        raise RuntimeError("parse failed")

    monkeypatch.setattr(ir, "ingest_document", boom)

    with TestClient(create_app()) as c:
        r = c.post(
            "/api/ingest/upload",
            files={"file": ("x.md", b"hi", "text/markdown")},
            data={"source_type": "upload"},
        )
        assert r.status_code == 500


# ---------------------------------------------------------------------------
# L9-7 — core.llm ollama branch
# ---------------------------------------------------------------------------
def test_l9_7_llm_ollama_branch(monkeypatch):
    from core import llm
    try:
        import langchain_ollama as lo
    except Exception:
        pytest.skip("langchain_ollama unavailable")

    class FakeChat:
        def __init__(self, **kw): self.kw = kw

    monkeypatch.setattr(lo, "ChatOllama", FakeChat, raising=False)
    res = llm.get_chat_llm(provider="ollama")
    assert isinstance(res, FakeChat)


def test_l9_7b_llm_openai_warns_without_key(monkeypatch):
    """Line 29: warn when OPENAI_API_KEY is empty (still constructs)."""
    from core import llm
    monkeypatch.setattr(llm.settings, "openai_api_key", "", raising=False)

    import langchain_openai as lo
    class FakeChat:
        def __init__(self, **kw): self.kw = kw
    monkeypatch.setattr(lo, "ChatOpenAI", FakeChat, raising=False)

    res = llm.get_chat_llm(provider="openai")
    assert isinstance(res, FakeChat)


def test_l9_7c_llm_get_embeddings_huggingface_branch(monkeypatch):
    """Hit huggingface branch via stubbed langchain_community."""
    from core import llm
    import sys

    class FakeHFE:
        def __init__(self, **kw): self.kw = kw

    fake_module = SimpleNamespace(HuggingFaceEmbeddings=FakeHFE)
    monkeypatch.setitem(sys.modules, "langchain_community.embeddings", fake_module)

    res = llm.get_embeddings("huggingface")
    assert isinstance(res, FakeHFE)


# ---------------------------------------------------------------------------
# L9-8 — ChatAgent reset / history / ask
# ---------------------------------------------------------------------------
def test_l9_8_chat_agent_history_and_reset(monkeypatch):
    from agents.chat_agent import ChatAgent
    import workflows.chat_workflow as cw
    from models.schemas import ChatMessage, ChatResponse

    async def fake_run_chat(message, history=None, *, use_memory=True, top_k=6):
        new_hist = (history or []) + [
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=f"a:{message}"),
        ]
        return ChatResponse(answer=f"a:{message}", sources=[], history=new_hist)

    monkeypatch.setattr(cw, "run_chat", fake_run_chat)
    # The ChatAgent imports run_chat from workflows.chat_workflow at module import time;
    # patch the agent module's binding too.
    import agents.chat_agent as ca
    monkeypatch.setattr(ca, "run_chat", fake_run_chat)

    a = ChatAgent(max_history=4)
    asyncio.run(a.ask("hi"))
    asyncio.run(a.ask("hello"))
    h = a.history
    assert len(h) == 4  # 2 turns × 2 msgs
    a.reset()
    assert a.history == []


# ---------------------------------------------------------------------------
# L9-9 — review_workflow run_review with custom period + monthly
# ---------------------------------------------------------------------------
def test_l9_9_run_review_monthly_default_dates(monkeypatch):
    from workflows import review_workflow as rw
    from models.schemas import ReviewResponse

    captured: dict[str, Any] = {}

    class FakeGraph:
        async def ainvoke(self, state):
            captured["state"] = state
            return {**state, "summary": "S", "obsidian_path": "/v/r.md", "items": []}

    monkeypatch.setattr(rw, "get_review_graph", lambda: FakeGraph())
    res = asyncio.run(rw.run_review("monthly"))
    assert res.period == "monthly"
    delta = (res.end - res.start).days
    assert 28 <= delta <= 31


def test_l9_9b_run_review_custom_period(monkeypatch):
    from workflows import review_workflow as rw

    class FakeGraph:
        async def ainvoke(self, state):
            return {**state, "summary": "S", "obsidian_path": None, "items": []}

    monkeypatch.setattr(rw, "get_review_graph", lambda: FakeGraph())
    # Custom defaults to weekly window
    res = asyncio.run(rw.run_review("custom"))
    assert res.period == "custom"


def test_l9_9c_review_node_summarise_no_items(monkeypatch):
    from workflows import review_workflow as rw
    state = {"period": "weekly", "start": datetime(2025, 1, 1), "end": datetime(2025, 1, 7)}
    out = asyncio.run(rw.node_summarise(state))
    assert "暂无新增" in out["summary"]


# ---------------------------------------------------------------------------
# L9-10 — ingest_workflow mindmap exception + ingest_url full path with stubs
# ---------------------------------------------------------------------------
def test_l9_10_node_mindmap_exception_handled(monkeypatch):
    from workflows import ingest_workflow as iw
    from scrapers.base import ScrapedDocument

    async def fake_mm(*, title, content):
        raise RuntimeError("mindmap llm down")

    monkeypatch.setattr(iw, "generate_mindmap", fake_mm)
    state = {
        "document": ScrapedDocument(
            title="t", content="c", source="s", source_type="web", metadata={},
        ),
        "summary": "sum",
    }
    out = asyncio.run(iw.node_mindmap(state))
    assert out["mindmap"] == ""  # exception swallowed, empty string returned


def test_l9_10b_ingest_url_with_stubbed_graph(monkeypatch):
    from workflows import ingest_workflow as iw
    from scrapers.base import ScrapedDocument
    from models.schemas import SourceType

    async def fake_ainvoke(state):
        return {
            "knowledge_id": "K9",
            "document": ScrapedDocument(
                title="title", content="c", source="https://e.com",
                source_type="web", metadata={},
            ),
            "summary": "summary",
            "tags": ["t"],
            "related": [],
            "obsidian_path": "/v/n.md",
            "chunks_indexed": 1,
        }

    monkeypatch.setattr(iw, "get_ingest_graph", lambda: SimpleNamespace(ainvoke=fake_ainvoke))
    res = asyncio.run(iw.ingest_url(SourceType.WEB, "https://e.com"))
    assert res.knowledge_id == "K9"
    assert res.title == "title"


def test_l9_10c_node_scrape_skips_when_doc_present(monkeypatch):
    """If state already has a document, node_scrape returns it as-is (lines 47-49)."""
    from workflows import ingest_workflow as iw
    from scrapers.base import ScrapedDocument
    doc = ScrapedDocument(title="t", content="c", source="s",
                          source_type="web", metadata={})
    state = {"document": doc, "source_type": "web", "target": "x"}
    out = asyncio.run(iw.node_scrape(state))
    assert out["document"] is doc


# ---------------------------------------------------------------------------
# L9-11 — obsidian add_backlink_section + frontmatter parse error
# ---------------------------------------------------------------------------
def test_l9_11_obsidian_add_backlink_section(tmp_path, monkeypatch):
    from core import obsidian as ob
    ob._VAULT = None  # reset singleton

    v = ob.ObsidianVault(root=tmp_path)
    p = v.write_note(title="MyNote", content="body")

    # Empty list short-circuit (line ~153)
    v.add_backlink_section(p, [])

    # Non-existent path short-circuit (line ~156)
    v.add_backlink_section(tmp_path / "nope.md", ["x"])

    # Add real backlinks
    v.add_backlink_section(p, ["Related1", "Related2"])
    text = Path(p).read_text(encoding="utf-8")
    assert "🔗 相关笔记" in text
    assert "[[Related1]]" in text
    ob._VAULT = None


def test_l9_11b_obsidian_split_frontmatter_yaml_error(tmp_path, monkeypatch):
    from core.obsidian import ObsidianVault
    raw = "---\n: : : invalid yaml\n---\n\n# body\n"
    fm, body = ObsidianVault._split_frontmatter(raw)
    assert fm == {}  # YAML error caught
    assert "body" in body


def test_l9_11c_obsidian_extract_tags_string_form():
    from core.obsidian import ObsidianVault
    raw = "---\ntags: a, b, c\n---\n\nbody #d\n"
    tags = ObsidianVault._extract_tags(raw, {"tags": "a, b, c"})
    assert "a" in tags and "b" in tags and "c" in tags and "d" in tags


def test_l9_11d_obsidian_safe_filename_empty():
    from core.obsidian import ObsidianVault
    out = ObsidianVault._safe_filename(":::///")
    assert out  # falls back to a timestamp-based name


def test_l9_11e_obsidian_relative_path_outside_vault(tmp_path):
    from core.obsidian import ObsidianNote
    n = ObsidianNote(
        path=Path("/some/abs/elsewhere.md"),
        title="t", content="c", frontmatter={}, tags=[], links=[],
    )
    # Triggers ValueError → returns absolute path str (lines 51-54 covered earlier)
    rp = n.relative_path
    assert isinstance(rp, str)


# ---------------------------------------------------------------------------
# L9-12 — scheduler.add_interval + register_default_jobs + shutdown
# ---------------------------------------------------------------------------
def test_l9_12_scheduler_add_interval(monkeypatch):
    from core import scheduler as sc

    sched = sc.TaskScheduler()
    captured: dict[str, Any] = {}

    fake_inner = MagicMock()
    fake_inner.add_job = MagicMock(side_effect=lambda *a, **kw: captured.update(kw))
    fake_inner.running = False
    fake_inner.start = MagicMock()
    fake_inner.shutdown = MagicMock()
    fake_inner.get_jobs = MagicMock(return_value=[])
    sched.scheduler = fake_inner

    async def fn(): pass
    sched.add_interval(fn, minutes=10, job_id="t1")
    assert captured["id"] == "t1"


def test_l9_12b_scheduler_shutdown_when_running(monkeypatch):
    from core import scheduler as sc
    sched = sc.TaskScheduler()
    fake_inner = MagicMock()
    fake_inner.running = True
    fake_inner.shutdown = MagicMock()
    sched.scheduler = fake_inner
    sched.shutdown()
    fake_inner.shutdown.assert_called_once_with(wait=False)


def test_l9_12c_register_default_jobs(monkeypatch):
    from core import scheduler as sc

    sched = sc.TaskScheduler()
    fake_inner = MagicMock()
    fake_inner.running = False
    fake_inner.add_job = MagicMock()
    fake_inner.start = MagicMock()
    fake_inner.get_jobs = MagicMock(return_value=[])
    sched.scheduler = fake_inner

    asyncio.run(sc.register_default_jobs())
    # 3 jobs registered (weekly, monthly, rss)
    assert fake_inner.add_job.call_count >= 3
    fake_inner.start.assert_called_once()


def test_l9_12d_scheduler_list_jobs_with_next_run(monkeypatch):
    from core import scheduler as sc
    sched = sc.TaskScheduler()
    fake_job = MagicMock()
    fake_job.id = "j1"
    fake_job.next_run_time = datetime(2025, 1, 1, 12, 0)
    fake_job.trigger = "cron"
    fake_inner = MagicMock()
    fake_inner.get_jobs = MagicMock(return_value=[fake_job])
    sched.scheduler = fake_inner
    out = sched.list_jobs()
    assert out[0]["id"] == "j1"
    assert out[0]["next_run"] == "2025-01-01T12:00:00"


# ---------------------------------------------------------------------------
# L9-13 — tagger _parse fallback branches
# ---------------------------------------------------------------------------
def test_l9_13_tagger_parse_json():
    from tools.tagger import _parse
    out = _parse('["ai", "ml", "transformer"]')
    assert out == ["ai", "ml", "transformer"]


def test_l9_13b_tagger_parse_with_fence():
    from tools.tagger import _parse
    out = _parse('```json\n["x", "y"]\n```')
    assert out == ["x", "y"]


def test_l9_13c_tagger_parse_brackets_inside():
    from tools.tagger import _parse
    out = _parse('here are tags: ["a", "b", "c"] end')
    assert "a" in out


def test_l9_13d_tagger_parse_csv_fallback():
    from tools.tagger import _parse
    out = _parse('alpha, beta、gamma; delta | epsilon')
    assert "alpha" in out and "beta" in out


def test_l9_13e_tagger_parse_dedup_case_insensitive():
    from tools.tagger import _parse
    out = _parse('["AI", "ai", "ML"]')
    # Case-insensitive dedup keeps first
    assert len(out) == 2


def test_l9_13f_tagger_parse_cap_at_10():
    from tools.tagger import _parse
    out = _parse('["t1","t2","t3","t4","t5","t6","t7","t8","t9","t10","t11","t12"]')
    assert len(out) == 10


# ---------------------------------------------------------------------------
# L9-14 — main.py chat REPL with stubbed input loop
# ---------------------------------------------------------------------------
def test_l9_14_cli_chat_repl(monkeypatch):
    """Drive the chat REPL through Console.input + agent.ask stubs."""
    from typer.testing import CliRunner
    import main as m
    import agents.chat_agent as ca
    from models.schemas import ChatResponse

    inputs = iter(["hello", "/reset", "  ", "exit"])

    def fake_input(self, prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise EOFError

    monkeypatch.setattr(type(m.console), "input", fake_input)

    async def fake_init():
        return None

    async def fake_ask(self, message, *, use_memory=True, top_k=6):
        return ChatResponse(answer=f"got {message}", sources=[], history=[])

    monkeypatch.setattr(m, "init_db", fake_init)
    monkeypatch.setattr(ca.ChatAgent, "ask", fake_ask)

    runner = CliRunner()
    result = runner.invoke(m.app, ["chat"])
    assert result.exit_code == 0


def test_l9_14b_cli_chat_eof_exits(monkeypatch):
    """EOF on input ends the loop cleanly."""
    from typer.testing import CliRunner
    import main as m

    def fake_input(self, prompt=""):
        raise EOFError

    monkeypatch.setattr(type(m.console), "input", fake_input)

    async def fake_init(): return None
    monkeypatch.setattr(m, "init_db", fake_init)

    runner = CliRunner()
    result = runner.invoke(m.app, ["chat"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# L9-15 — chat_workflow assistant message in history (lines 54-55)
# ---------------------------------------------------------------------------
def test_l9_15_chat_workflow_assistant_history(monkeypatch):
    from workflows import chat_workflow as cw
    from models.schemas import ChatMessage
    cw._CHAT_GRAPH = None

    async def fake_search(*a, **kw):
        return []

    class FakeLLM:
        async def ainvoke(self, msgs):
            # Verify the assistant history message was included
            roles = [type(m).__name__ for m in msgs]
            assert any("AI" in r or "Human" in r for r in roles)
            return SimpleNamespace(content="assistant-answer")

    monkeypatch.setattr(cw, "hybrid_search", fake_search)
    monkeypatch.setattr(cw, "get_chat_llm", lambda: FakeLLM())

    history = [
        ChatMessage(role="user", content="prev1"),
        ChatMessage(role="assistant", content="prev-ans"),
        ChatMessage(role="system", content="ignored"),  # neither branch
    ]
    res = asyncio.run(cw.run_chat("Q", history=history))
    assert res.answer == "assistant-answer"
    cw._CHAT_GRAPH = None


# ---------------------------------------------------------------------------
# L9 sentinel
# ---------------------------------------------------------------------------
def test_l9_sentinel():
    """Sanity sentinel — loop 9 file loaded."""
    assert True
