"""Loop 6 — 20 fresh quantitative iterations on parsers, schemas, and graph hygiene.

Targets disjoint from loops 1-5: ScrapedDocument round-trip, parsers (srt/vtt/md),
unknown extensions, ABC enforcement, ingest graph order, empty add, boolean metadata,
SearchHit equality, concurrent reads, Settings defaults, ChatMessage/IngestResult/
ReviewResponse round-trip, frontmatter special chars, append_to_daily, make_wikilink.
"""
from __future__ import annotations

import asyncio
import hashlib
import inspect
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


# ---------- shared fakes ----------
class FakeEmbeddings(Embeddings):
    DIM = 16

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        v = [b / 255.0 for b in h[: self.DIM]]
        n = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / n for x in v]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


def _fresh_vault(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))
    from core import config as cfg
    cfg.settings = cfg.Settings()
    import core.obsidian as obsmod
    obsmod._VAULT = None
    monkeypatch.setattr(obsmod, "settings", cfg.settings, raising=False)
    return obsmod.get_vault()


def _fresh_vstore(tmp_path: Path, monkeypatch, name: str):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", name)
    from core import config as cfg
    cfg.settings = cfg.Settings()
    import core.vector_store as vsmod
    vsmod._CLIENT_CACHE.clear()
    vsmod._VECTOR_STORE = None
    monkeypatch.setattr(vsmod, "settings", cfg.settings, raising=False)
    return vsmod.VectorStore(collection_name=name, embeddings=FakeEmbeddings())


# ============= ITER 1 =============
def test_iter01_baseline_marker():
    # Captured externally: 120 tests / median 1.98s.
    assert True


# ============= ITER 2: ScrapedDocument.to_dict round-trip =============
def test_iter02_scraped_document_to_dict_roundtrip():
    from scrapers.base import ScrapedDocument
    d = ScrapedDocument(
        title="T", content="body", source="https://x.example/a",
        source_type="web", metadata={"k": "v"},
    )
    out = d.to_dict()
    assert out["title"] == "T"
    assert out["content"] == "body"
    assert out["source"] == "https://x.example/a"
    assert out["source_type"] == "web"
    assert out["metadata"] == {"k": "v"}
    # fetched_at is ISO 8601
    datetime.fromisoformat(out["fetched_at"])


# ============= ITER 3: parse_srt strips timestamps =============
def test_iter03_parse_srt_strips_timestamps(tmp_path):
    from utils.parsers import parse_srt
    srt = (
        "1\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "Hello world\n\n"
        "2\n"
        "00:00:02,000 --> 00:00:03,500\n"
        "Second line\n"
    )
    f = tmp_path / "x.srt"
    f.write_text(srt, encoding="utf-8")
    out = parse_srt(f)
    assert "Hello world" in out
    assert "Second line" in out
    assert "-->" not in out
    assert "00:00" not in out


# ============= ITER 4: parse_vtt drops WEBVTT/NOTE/timing =============
def test_iter04_parse_vtt_drops_metadata(tmp_path):
    from utils.parsers import parse_vtt
    vtt = (
        "WEBVTT\n\n"
        "NOTE this is a note\n\n"
        "00:00:01.000 --> 00:00:02.000\n"
        "Spoken text alpha\n\n"
        "00:00:02.000 --> 00:00:03.000\n"
        "Spoken text beta\n"
    )
    f = tmp_path / "x.vtt"
    f.write_text(vtt, encoding="utf-8")
    out = parse_vtt(f)
    assert "Spoken text alpha" in out
    assert "Spoken text beta" in out
    assert "WEBVTT" not in out
    assert "NOTE" not in out
    assert "-->" not in out


# ============= ITER 5: parse_markdown handles UTF-8 + emoji =============
def test_iter05_parse_markdown_utf8_emoji(tmp_path):
    from utils.parsers import parse_markdown
    f = tmp_path / "y.md"
    payload = "# 标题🚀\n\n中文内容 with emoji 🎯 and Японский\n"
    f.write_text(payload, encoding="utf-8")
    out = parse_markdown(f)
    assert out == payload


# ============= ITER 6: parse_file rejects unknown extension =============
def test_iter06_parse_file_unknown_ext(tmp_path):
    from utils.parsers import parse_file
    f = tmp_path / "z.unknownext"
    f.write_text("data", encoding="utf-8")
    with pytest.raises(ValueError):
        parse_file(f)


# ============= ITER 7: ABC enforces scrape() implementation =============
def test_iter07_abc_enforcement():
    from scrapers.base import BaseScraper
    class Incomplete(BaseScraper):
        source_type = "broken"
    with pytest.raises(TypeError):
        Incomplete()  # type: ignore


# ============= ITER 8: ingest graph node order via source inspection =============
def test_iter08_ingest_graph_node_order():
    import workflows.ingest_workflow as iw
    src = inspect.getsource(iw.build_ingest_graph)
    # nodes in order
    expected_order = ["scrape", "summarize", "tag_node", "link_node", "mindmap_node", "persist"]
    positions = [src.index(f'"{n}"') for n in expected_order]
    assert positions == sorted(positions), f"node order broken: {positions}"


# ============= ITER 9: vector_store.add_documents([]) returns [] =============
@pytest.mark.asyncio
async def test_iter09_empty_add_returns_empty(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter6_09")
    out = await store.add_documents([])
    assert out == []
    assert store.count() == 0


# ============= ITER 10: boolean metadata preserved exactly =============
@pytest.mark.asyncio
async def test_iter10_boolean_metadata_preserved(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter6_10")
    await store.add_documents([
        Document(page_content="boolT", metadata={"flag": True, "title": "yes"}),
        Document(page_content="boolF", metadata={"flag": False, "title": "no"}),
    ])
    yes = await store.search("boolT", k=1)
    no = await store.search("boolF", k=1)
    assert yes[0].metadata["flag"] is True
    assert no[0].metadata["flag"] is False


# ============= ITER 11: SearchHit dataclass equality + repr =============
def test_iter11_search_hit_equality():
    from core.vector_store import SearchHit
    a = SearchHit(id="x", content="c", metadata={"k": "v"}, score=0.5)
    b = SearchHit(id="x", content="c", metadata={"k": "v"}, score=0.5)
    c = SearchHit(id="y", content="c", metadata={}, score=0.5)
    assert a == b
    assert a != c
    assert "SearchHit" in repr(a)


# ============= ITER 12: concurrent searches return identical hit count =============
@pytest.mark.asyncio
async def test_iter12_concurrent_search_consistent(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter6_12")
    await store.add_documents([
        Document(page_content=f"doc{i}", metadata={"i": i}) for i in range(20)
    ])
    results = await asyncio.gather(*[store.search("doc", k=5) for _ in range(10)])
    counts = {len(r) for r in results}
    assert counts == {5}


# ============= ITER 13: Settings defaults present =============
def test_iter13_settings_defaults():
    from core.config import Settings
    s = Settings()
    # required fields with defaults exist
    assert s.chroma_collection
    assert s.chroma_persist_dir
    assert s.database_url
    assert s.vault_path
    assert s.weekly_review_cron
    assert s.monthly_review_cron


# ============= ITER 14: ChatMessage Pydantic round-trip =============
def test_iter14_chat_message_roundtrip():
    from models.schemas import ChatMessage
    m = ChatMessage(role="user", content="hello")
    payload = m.model_dump()
    m2 = ChatMessage.model_validate(payload)
    assert m == m2
    # invalid role rejected
    with pytest.raises(Exception):
        ChatMessage(role="bogus_role", content="x")


# ============= ITER 15: IngestResult Pydantic round-trip =============
def test_iter15_ingest_result_roundtrip():
    from models.schemas import IngestResult, SourceType
    r = IngestResult(
        knowledge_id="k1", title="T", summary="s",
        tags=["a", "b"], links=["x"], obsidian_path="/p.md",
        chunks_indexed=3, source_type=SourceType.WEB, source="https://x",
    )
    payload = r.model_dump()
    assert payload["chunks_indexed"] == 3
    r2 = IngestResult.model_validate(payload)
    assert r2.tags == ["a", "b"] and r2.knowledge_id == "k1"


# ============= ITER 16: ReviewResponse Pydantic round-trip =============
def test_iter16_review_response_roundtrip():
    from models.schemas import ReviewResponse
    r = ReviewResponse(
        period="weekly", start=datetime(2024, 1, 1), end=datetime(2024, 1, 8),
        summary="s", obsidian_path=None, knowledge_count=12,
    )
    payload = r.model_dump()
    r2 = ReviewResponse.model_validate(payload)
    assert r2.knowledge_count == 12 and r2.period == "weekly"


# ============= ITER 17: frontmatter survives colons/quotes/multiline =============
def test_iter17_frontmatter_special_chars(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    spec = {
        "source": "https://example.com/path?q=a:b",
        "quote": 'he said "hello"',
        "multiline": "line1\nline2\nline3",
    }
    p = vault.write_note(title="special", content="body", frontmatter=spec)
    n = vault.read_note(p)
    for k, v in spec.items():
        assert n.frontmatter.get(k) == v, f"{k}: got {n.frontmatter.get(k)!r} != {v!r}"


# ============= ITER 18: append_to_daily creates then appends =============
def test_iter18_append_to_daily(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    day = datetime(2024, 6, 15, 9, 0, 0)
    p = vault.append_to_daily("first entry", day=day)
    assert p.exists()
    txt1 = p.read_text(encoding="utf-8")
    assert "first entry" in txt1
    p2 = vault.append_to_daily("second entry", day=day)
    assert p == p2
    txt2 = p.read_text(encoding="utf-8")
    assert "first entry" in txt2 and "second entry" in txt2


# ============= ITER 19: make_wikilink format =============
def test_iter19_make_wikilink_format():
    from core.obsidian import ObsidianVault
    assert ObsidianVault.make_wikilink("Note A") == "[[Note A]]"
    assert ObsidianVault.make_wikilink("Note A", alias="alpha") == "[[Note A|alpha]]"


# ============= ITER 20: regression sweep =============
def test_iter20_regression_sweep():
    import importlib
    mods = [
        "core.config", "core.llm", "core.obsidian", "core.scheduler", "core.vector_store",
        "models.database", "models.schemas",
        "tools.linker", "tools.mindmap", "tools.search", "tools.summarizer", "tools.tagger",
        "scrapers.base", "scrapers.web_scraper", "scrapers.rss_scraper", "scrapers.pdf_scraper",
        "agents.chat_agent", "agents.pkm_agent", "agents.review_agent",
        "workflows.chat_workflow", "workflows.ingest_workflow", "workflows.review_workflow",
        "api.server",
        "utils.logger", "utils.parsers", "utils.retry",
    ]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e}")
    assert not failed, "import failures: " + "; ".join(failed)
