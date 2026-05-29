"""Loop 8 — coverage push v2: target CLI + remaining scrapers + parsers + search filters.

Goal: lift overall coverage from ~90% to ≥95% by exercising:
- main.py Typer CLI: ingest / search / review / init-db / rss-add
- scrapers/rss_scraper: _parse, scrape, fetch_all, list/add subscriptions, refresh
- scrapers/youtube_scraper: id extraction, scrape happy path, no-transcript fallback
- scrapers/email_scraper: missing creds, _decode_header, _extract_body multipart
- scrapers/notion_scraper: missing-key warn, _block_to_text branches, _fetch_page stubbed
- scrapers/wechat_scraper: full scrape with stubbed _web fetcher
- scrapers/pdf_scraper: download branch with stubbed httpx
- utils/parsers: parse_docx, parse_pdf, parse_file dispatch + unsupported
- tools/search: vector_search source_type filter, tags filter, keyword_search

All tests fully isolated: no real network, no real LLM, no real DB beyond test sqlite.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# L8-1 — main.py CLI commands via Typer CliRunner
# ---------------------------------------------------------------------------
def test_l8_1_cli_ingest(monkeypatch):
    """python main.py ingest web https://example.com — agent.ingest invoked."""
    from typer.testing import CliRunner
    import main as m
    from models.schemas import IngestResult, SourceType

    captured: dict[str, Any] = {}

    class FakeAgent:
        async def ingest(self, source_type, target, **kw):
            captured["source_type"] = source_type
            captured["target"] = target
            return IngestResult(
                knowledge_id="K1", title="title", summary="# summary",
                tags=["a", "b"], links=["L"], obsidian_path="/v/n.md",
                chunks_indexed=2, source_type=source_type, source=target,
            )

    async def fake_init():
        return None

    monkeypatch.setattr(m, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(m, "init_db", fake_init)

    runner = CliRunner()
    result = runner.invoke(m.app, ["ingest", "web", "https://example.com/x"])
    assert result.exit_code == 0
    assert captured["source_type"] == SourceType.WEB
    assert "title" in result.stdout


def test_l8_1b_cli_search(monkeypatch):
    """python main.py search 'q' --k 3 — hybrid_search invoked."""
    from typer.testing import CliRunner
    import main as m
    from models.schemas import SearchResultItem
    import tools.search as ts

    async def fake_init():
        return None

    async def fake_hybrid(query, *, k=5, source_type=None):
        return [
            SearchResultItem(
                id="x", title="Hello", snippet="content", score=0.5,
                source="path", metadata={},
            )
        ]

    monkeypatch.setattr(m, "init_db", fake_init)
    monkeypatch.setattr(ts, "hybrid_search", fake_hybrid)

    runner = CliRunner()
    result = runner.invoke(m.app, ["search", "test query", "--k", "3"])
    assert result.exit_code == 0
    assert "Hello" in result.stdout


def test_l8_1c_cli_review(monkeypatch):
    """python main.py review weekly — agent.review invoked."""
    from datetime import datetime
    from typer.testing import CliRunner
    import main as m
    from models.schemas import ReviewResponse

    class FakeAgent:
        async def review(self, period):
            return ReviewResponse(
                period=period, start=datetime(2025, 1, 1), end=datetime(2025, 1, 7),
                summary="# weekly\n\ncontent",
                obsidian_path="/v/review.md", knowledge_count=5,
            )

    async def fake_init():
        return None

    monkeypatch.setattr(m, "get_agent", lambda: FakeAgent())
    monkeypatch.setattr(m, "init_db", fake_init)

    runner = CliRunner()
    result = runner.invoke(m.app, ["review", "weekly"])
    assert result.exit_code == 0
    assert "/v/review.md" in result.stdout


def test_l8_1d_cli_init_db(monkeypatch):
    from typer.testing import CliRunner
    import main as m

    called = {"n": 0}

    async def fake_init():
        called["n"] += 1

    monkeypatch.setattr(m, "init_db", fake_init)
    runner = CliRunner()
    result = runner.invoke(m.app, ["init-db"])
    assert result.exit_code == 0
    assert called["n"] == 1
    assert "database ready" in result.stdout


def test_l8_1e_cli_rss_add(monkeypatch, tmp_path):
    from typer.testing import CliRunner
    import main as m
    from scrapers import rss_scraper

    sub_path = tmp_path / "subs.txt"
    monkeypatch.setattr(rss_scraper, "SUBSCRIPTIONS_PATH", sub_path)

    runner = CliRunner()
    result = runner.invoke(m.app, ["rss-add", "https://example.com/feed.xml"])
    assert result.exit_code == 0
    assert sub_path.exists()
    assert "https://example.com/feed.xml" in sub_path.read_text()


def test_l8_1f_cli_no_args_help():
    from typer.testing import CliRunner
    import main as m
    runner = CliRunner()
    result = runner.invoke(m.app, [])
    # No-args prints help and exits with 0 or 2 (help-mode exit codes vary)
    assert "ingest" in result.stdout or "ingest" in (result.output or "")


# ---------------------------------------------------------------------------
# L8-2 — RSS scraper
# ---------------------------------------------------------------------------
def test_l8_2_rss_parse_with_stubbed_feedparser(monkeypatch):
    from scrapers import rss_scraper as rs

    class Entry(dict):
        """feedparser-style entry: supports both .key and ['key']."""
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError as e:
                raise AttributeError(k) from e

    fake_feed = SimpleNamespace(
        bozo=False,
        bozo_exception=None,
        feed={"title": "Tech Feed"},
        entries=[
            Entry(title="Item1", summary="summary1", link="https://e.com/1",
                  published="2025-01-01"),
            Entry(title="Item2", description="desc2", link="https://e.com/2",
                  published="2025-01-02",
                  content=[{"value": "<p>full content</p>"}]),
        ],
    )
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)

    s = rs.RSSScraper()
    items = s._parse("https://feed.com/rss")
    assert len(items) == 2
    assert items[0].title == "Item1"
    assert items[0].metadata["feed_title"] == "Tech Feed"
    assert "full content" in items[1].content


def test_l8_2b_rss_parse_bozo_no_entries(monkeypatch):
    from scrapers import rss_scraper as rs
    fake_feed = SimpleNamespace(
        bozo=True, bozo_exception=Exception("xml broken"),
        feed={}, entries=[],
    )
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)
    items = rs.RSSScraper()._parse("https://bad/rss")
    assert items == []


def test_l8_2c_rss_scrape_no_entries_raises(monkeypatch):
    from scrapers import rss_scraper as rs
    fake_feed = SimpleNamespace(bozo=False, feed={}, entries=[])
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)
    with pytest.raises(RuntimeError, match="No entries parsed"):
        asyncio.run(rs.RSSScraper().scrape("https://empty/rss"))


def test_l8_2d_rss_subscriptions_roundtrip(monkeypatch, tmp_path):
    from scrapers import rss_scraper as rs
    sub_path = tmp_path / "subs.txt"
    monkeypatch.setattr(rs, "SUBSCRIPTIONS_PATH", sub_path)
    assert rs.RSSScraper.list_subscriptions() == []
    rs.RSSScraper.add_subscription("https://a.com/rss")
    rs.RSSScraper.add_subscription("https://a.com/rss")  # dup
    rs.RSSScraper.add_subscription("https://b.com/rss")
    subs = rs.RSSScraper.list_subscriptions()
    assert subs == ["https://a.com/rss", "https://b.com/rss"]


def test_l8_2e_rss_scrape_happy_with_stub(monkeypatch):
    from scrapers import rss_scraper as rs
    fake_feed = SimpleNamespace(
        bozo=False,
        feed={"title": "F"},
        entries=[{"title": "T1", "summary": "s", "link": "https://e/1", "published": "x"}],
    )
    monkeypatch.setattr(rs.feedparser, "parse", lambda url, agent=None: fake_feed)
    doc = asyncio.run(rs.RSSScraper().scrape("https://feed/rss"))
    assert doc.title == "T1"
    assert "all" in doc.metadata
    assert len(doc.metadata["all"]) == 1


# ---------------------------------------------------------------------------
# L8-3 — YouTube scraper
# ---------------------------------------------------------------------------
def test_l8_3_youtube_extract_video_id():
    from scrapers.youtube_scraper import YouTubeScraper
    assert YouTubeScraper.extract_video_id("https://youtu.be/abc12345678") == "abc12345678"
    assert YouTubeScraper.extract_video_id(
        "https://www.youtube.com/watch?v=ABCDEFGHIJK"
    ) == "ABCDEFGHIJK"
    assert YouTubeScraper.extract_video_id(
        "https://www.youtube.com/embed/ZZZ12345678"
    ) == "ZZZ12345678"


def test_l8_3b_youtube_extract_video_id_invalid():
    from scrapers.youtube_scraper import YouTubeScraper
    with pytest.raises(ValueError, match="Could not extract"):
        YouTubeScraper.extract_video_id("https://example.com/no-id-here")


def test_l8_3c_youtube_scrape_happy(monkeypatch):
    from scrapers import youtube_scraper as ys

    async def fake_transcript(self, video_id, languages):
        return ("hello world transcript", "transcript")

    async def fake_meta(self, video_id):
        return {"title": "Real Title", "description": "real desc",
                "channel": "Ch", "published_at": "2025"}

    monkeypatch.setattr(ys.YouTubeScraper, "_fetch_transcript", fake_transcript)
    monkeypatch.setattr(ys.YouTubeScraper, "_fetch_metadata", fake_meta)

    doc = asyncio.run(ys.YouTubeScraper().scrape("https://youtu.be/abc12345678"))
    assert doc.title == "Real Title"
    assert "hello world" in doc.content
    assert "real desc" in doc.content
    assert doc.metadata["video_id"] == "abc12345678"


def test_l8_3d_youtube_scrape_no_transcript(monkeypatch):
    from scrapers import youtube_scraper as ys

    async def fail_transcript(self, video_id, languages):
        raise RuntimeError("no captions")

    async def fake_meta(self, video_id):
        return {}

    monkeypatch.setattr(ys.YouTubeScraper, "_fetch_transcript", fail_transcript)
    monkeypatch.setattr(ys.YouTubeScraper, "_fetch_metadata", fake_meta)

    doc = asyncio.run(ys.YouTubeScraper().scrape("https://youtu.be/abc12345678"))
    assert doc.title.startswith("YouTube ")
    assert "no transcript available" in doc.content


def test_l8_3e_youtube_metadata_no_api_key(monkeypatch):
    """_fetch_metadata returns {} when no api key."""
    from scrapers import youtube_scraper as ys
    from core.config import settings as s
    monkeypatch.setattr(s, "youtube_api_key", "", raising=False)
    res = asyncio.run(ys.YouTubeScraper()._fetch_metadata("abcdefghijk"))
    assert res == {}


# ---------------------------------------------------------------------------
# L8-4 — Email scraper
# ---------------------------------------------------------------------------
def test_l8_4_email_missing_creds_raises(monkeypatch):
    from scrapers import email_scraper as es
    monkeypatch.setattr(es.settings, "email_user", "", raising=False)
    monkeypatch.setattr(es.settings, "email_password", "", raising=False)
    with pytest.raises(RuntimeError, match="EMAIL_USER"):
        es.EmailScraper()._fetch_recent(1)


def test_l8_4b_email_decode_header():
    from scrapers.email_scraper import _decode_header
    assert _decode_header(None) == ""
    assert _decode_header("plain") == "plain"
    # base64 encoded "你好"
    enc = "=?utf-8?B?5L2g5aW9?="
    out = _decode_header(enc)
    assert "你好" in out


def test_l8_4c_email_extract_body_simple():
    from email.message import EmailMessage
    from scrapers.email_scraper import EmailScraper
    msg = EmailMessage()
    msg["Subject"] = "S"
    msg.set_content("plain body content")
    body = EmailScraper._extract_body(msg)
    assert "plain body content" in body


def test_l8_4d_email_extract_body_multipart():
    from email.message import EmailMessage
    from scrapers.email_scraper import EmailScraper
    msg = EmailMessage()
    msg["Subject"] = "S"
    msg.set_content("plain part text")
    msg.add_alternative("<p>html part</p>", subtype="html")
    body = EmailScraper._extract_body(msg)
    assert "plain part" in body


# ---------------------------------------------------------------------------
# L8-5 — Notion scraper
# ---------------------------------------------------------------------------
def test_l8_5_notion_missing_key_warn(monkeypatch):
    from scrapers import notion_scraper as ns
    monkeypatch.setattr(ns.settings, "notion_api_key", "", raising=False)
    s = ns.NotionScraper()
    assert s._client is None
    with pytest.raises(RuntimeError, match="NOTION_API_KEY"):
        s._ensure()


def test_l8_5b_notion_block_to_text_all_branches(monkeypatch):
    from scrapers import notion_scraper as ns
    monkeypatch.setattr(ns.settings, "notion_api_key", "", raising=False)
    s = ns.NotionScraper()

    def mk(btype, text, **extra):
        return {
            "type": btype,
            btype: {"rich_text": [{"plain_text": text}], **extra},
        }

    assert s._block_to_text(mk("heading_1", "H1")) == "# H1"
    assert s._block_to_text(mk("heading_2", "H2")) == "## H2"
    assert s._block_to_text(mk("heading_3", "H3")) == "### H3"
    assert s._block_to_text(mk("bulleted_list_item", "B")) == "- B"
    assert s._block_to_text(mk("numbered_list_item", "N")) == "1. N"
    assert s._block_to_text(mk("to_do", "T", checked=True)) == "- [x] T"
    assert s._block_to_text(mk("to_do", "T2", checked=False)) == "- [ ] T2"
    assert s._block_to_text(mk("code", "print()", language="python")).startswith("```python")
    assert s._block_to_text(mk("quote", "Q")) == "> Q"
    assert s._block_to_text(mk("paragraph", "plain")) == "plain"
    assert s._block_to_text({"type": "unknown", "unknown": {}}) == ""


def test_l8_5c_notion_fetch_page_stubbed(monkeypatch):
    from scrapers import notion_scraper as ns
    monkeypatch.setattr(ns.settings, "notion_api_key", "", raising=False)
    s = ns.NotionScraper()

    class FakeBlocks:
        def list(self, block_id, start_cursor=None):
            return {
                "results": [
                    {"type": "heading_1",
                     "heading_1": {"rich_text": [{"plain_text": "Title"}]}},
                    {"type": "paragraph",
                     "paragraph": {"rich_text": [{"plain_text": "body"}]}},
                ],
                "has_more": False,
            }

    class FakePages:
        def retrieve(self, page_id):
            return {
                "url": "https://notion.so/p/" + page_id,
                "properties": {
                    "Title": {
                        "type": "title",
                        "title": [{"plain_text": "My Page"}],
                    }
                },
            }

    s._client = SimpleNamespace(
        pages=FakePages(),
        blocks=SimpleNamespace(children=FakeBlocks()),
    )

    doc = s._fetch_page("abc123")
    assert doc.title == "My Page"
    assert "Title" in doc.content
    assert "body" in doc.content
    assert doc.source.startswith("notion://")


def test_l8_5d_notion_scrape_dashes_stripped(monkeypatch):
    from scrapers import notion_scraper as ns
    monkeypatch.setattr(ns.settings, "notion_api_key", "", raising=False)
    s = ns.NotionScraper()

    captured = {}

    def fake_fetch(page_id):
        captured["page_id"] = page_id
        from scrapers.base import ScrapedDocument
        return ScrapedDocument(title="t", content="c", source="notion://x",
                               source_type="notion", metadata={})

    monkeypatch.setattr(s, "_fetch_page", fake_fetch)
    asyncio.run(s.scrape("a-b-c-d"))
    assert captured["page_id"] == "abcd"  # dashes removed


# ---------------------------------------------------------------------------
# L8-6 — WeChat scraper with stubbed _web fetcher
# ---------------------------------------------------------------------------
def test_l8_6_wechat_scrape_with_stub(monkeypatch):
    from scrapers import wechat_scraper as ws
    from scrapers.web_scraper import WebScraper

    html = """
    <html><head><title>Backup Title</title></head>
    <body>
      <h2 id="activity-name">Real WeChat Title</h2>
      <a id="js_name">Authorized Author</a>
      <em id="publish_time">2025-05-01</em>
      <div id="js_content">
        <p>Hello WeChat content.</p>
      </div>
    </body></html>
    """

    async def fake_pw(self, url):
        return None

    async def fake_httpx(self, url):
        return html

    monkeypatch.setattr(WebScraper, "_fetch_with_playwright", fake_pw)
    monkeypatch.setattr(WebScraper, "_fetch_with_httpx", fake_httpx)

    doc = asyncio.run(ws.WeChatScraper().scrape("https://mp.weixin.qq.com/s/xyz"))
    assert doc.title == "Real WeChat Title"
    assert doc.metadata["author"] == "Authorized Author"
    assert doc.metadata["published"] == "2025-05-01"
    assert "Hello WeChat" in doc.content


def test_l8_6b_wechat_non_wechat_url_warn(monkeypatch):
    from scrapers import wechat_scraper as ws
    from scrapers.web_scraper import WebScraper

    html = "<html><body><div id='js_content'>fallback</div></body></html>"

    async def fake_pw(self, url):
        return None

    async def fake_httpx(self, url):
        return html

    monkeypatch.setattr(WebScraper, "_fetch_with_playwright", fake_pw)
    monkeypatch.setattr(WebScraper, "_fetch_with_httpx", fake_httpx)
    doc = asyncio.run(ws.WeChatScraper().scrape("https://example.com/article"))
    assert "fallback" in doc.content


# ---------------------------------------------------------------------------
# L8-7 — PDF scraper download branch
# ---------------------------------------------------------------------------
def test_l8_7_pdf_scraper_download_branch(monkeypatch, tmp_path):
    from scrapers import pdf_scraper as ps

    fake_path = tmp_path / "downloaded.pdf"
    fake_path.write_bytes(b"%PDF-1.4 stub")

    async def fake_download(self, url):
        return fake_path

    monkeypatch.setattr(ps.PDFScraper, "_download", fake_download)
    monkeypatch.setattr(ps, "parse_pdf", lambda p: "downloaded text")

    doc = asyncio.run(ps.PDFScraper().scrape("https://example.com/x.pdf"))
    assert doc.content == "downloaded text"
    assert doc.source == "https://example.com/x.pdf"
    assert doc.source_type == "pdf"


def test_l8_7b_pdf_scraper_with_title_kwarg(monkeypatch, tmp_path):
    from scrapers import pdf_scraper as ps
    pdf = tmp_path / "f.pdf"
    pdf.write_bytes(b"%PDF stub")
    monkeypatch.setattr(ps, "parse_pdf", lambda p: "TXT")
    doc = asyncio.run(ps.PDFScraper().scrape(str(pdf), title="Custom Title"))
    assert doc.title == "Custom Title"


# ---------------------------------------------------------------------------
# L8-8 — utils/parsers coverage
# ---------------------------------------------------------------------------
def test_l8_8_parse_file_unsupported(tmp_path):
    from utils.parsers import parse_file
    f = tmp_path / "x.weird"
    f.write_text("data")
    with pytest.raises(ValueError, match="Unsupported"):
        parse_file(f)


def test_l8_8b_parse_file_dispatches_md(tmp_path):
    from utils.parsers import parse_file
    f = tmp_path / "n.md"
    f.write_text("# hi\nbody", encoding="utf-8")
    assert "body" in parse_file(f)


def test_l8_8c_parse_file_dispatches_txt(tmp_path):
    from utils.parsers import parse_file
    f = tmp_path / "n.txt"
    f.write_text("plain text content", encoding="utf-8")
    assert "plain text" in parse_file(f)


def test_l8_8d_parse_pdf_with_stubbed_pdfplumber(monkeypatch, tmp_path):
    """parse_pdf wraps pdfplumber.open — we stub the context manager."""
    import utils.parsers as up

    pdf_path = tmp_path / "x.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 stub")

    class FakePage:
        def extract_text(self):
            return "page1 text"

    class FakePDF:
        pages = [FakePage(), FakePage()]
        def __enter__(self): return self
        def __exit__(self, *a): return False

    fake_pdfplumber = SimpleNamespace(open=lambda p: FakePDF())

    import sys
    monkeypatch.setitem(sys.modules, "pdfplumber", fake_pdfplumber)
    text = up.parse_pdf(pdf_path)
    assert "page1" in text


def test_l8_8e_parse_docx_with_stubbed_python_docx(monkeypatch, tmp_path):
    """parse_docx wraps python-docx — stub it via module injection."""
    import sys
    import utils.parsers as up

    class FakePara:
        def __init__(self, text): self.text = text

    class FakeCell:
        def __init__(self, text): self.text = text

    class FakeRow:
        def __init__(self, cells): self.cells = cells

    class FakeTable:
        def __init__(self, rows): self.rows = rows

    class FakeDoc:
        paragraphs = [FakePara("para A"), FakePara(""), FakePara("para B")]
        tables = [FakeTable([FakeRow([FakeCell("c1"), FakeCell("c2")])])]

    fake_docx = SimpleNamespace(Document=lambda p: FakeDoc())
    monkeypatch.setitem(sys.modules, "docx", fake_docx)

    out = up.parse_docx(tmp_path / "x.docx")
    assert "para A" in out
    assert "para B" in out
    assert "c1 | c2" in out


# ---------------------------------------------------------------------------
# L8-9 — tools/search filter branches (real vector store, real obsidian)
# ---------------------------------------------------------------------------
def test_l8_9_vector_search_zero_k_short_circuit():
    from tools.search import vector_search
    res = asyncio.run(vector_search("anything", k=0))
    assert res == []


def test_l8_9b_keyword_search_zero_k_short_circuit():
    from tools.search import keyword_search
    res = asyncio.run(keyword_search("anything", k=0))
    assert res == []


def test_l8_9c_vector_search_with_source_type_filter(monkeypatch):
    """Source-type filter path - stub vector store search."""
    from core import vector_store as vs
    from tools import search as ts
    from core.vector_store import SearchHit

    captured: dict[str, Any] = {}

    class FakeVS:
        async def search(self, query, *, k=5, where=None):
            captured["where"] = where
            return [
                SearchHit(
                    id="h1", content="content", score=0.9,
                    metadata={"title": "T", "source": "s", "tags": "ai,ml"},
                )
            ]

    monkeypatch.setattr(ts, "get_vector_store", lambda: FakeVS())
    res = asyncio.run(ts.vector_search("q", k=3, source_type="web"))
    assert captured["where"] == {"source_type": "web"}
    assert len(res) == 1


def test_l8_9d_vector_search_tags_filter(monkeypatch):
    """Tags filter - only items where wanted ⊆ stored tags survive."""
    from tools import search as ts
    from core.vector_store import SearchHit

    class FakeVS:
        async def search(self, query, *, k=5, where=None):
            return [
                SearchHit(id="a", content="A", score=0.9,
                          metadata={"title": "A", "tags": "ai,ml"}),
                SearchHit(id="b", content="B", score=0.8,
                          metadata={"title": "B", "tags": "cooking"}),
            ]

    monkeypatch.setattr(ts, "get_vector_store", lambda: FakeVS())
    res = asyncio.run(ts.vector_search("q", k=5, tags=["ai"]))
    titles = [r.title for r in res]
    assert "A" in titles
    assert "B" not in titles


def test_l8_9e_keyword_search_returns_items(monkeypatch, tmp_path):
    """Keyword search reads from the vault; stub vault to return controlled notes."""
    from tools import search as ts
    from core.obsidian import ObsidianNote

    class FakeVault:
        def search_notes(self, keyword):
            return [
                ObsidianNote(
                    path=tmp_path / "n.md",
                    title="MyNote",
                    content="hello\nworld",
                    frontmatter={},
                    tags=["t1"],
                    links=[],
                ),
            ]

    monkeypatch.setattr(ts, "get_vault", lambda: FakeVault())
    res = asyncio.run(ts.keyword_search("hello", k=3))
    assert len(res) == 1
    assert res[0].title == "MyNote"


# ---------------------------------------------------------------------------
# L8-10 sentinel
# ---------------------------------------------------------------------------
def test_l8_10_loop8_sentinel():
    """Sanity sentinel — loop 8 file loaded."""
    assert True
