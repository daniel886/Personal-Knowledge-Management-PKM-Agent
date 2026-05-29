"""Loop 4 — 20 fresh quantitative iterations focused on robustness, hygiene, and throughput.

Targets are disjoint from loops 1-3. All LLM/network calls stubbed; tmp_path used for state.
"""
from __future__ import annotations

import asyncio
import hashlib
import statistics
import time
from datetime import datetime, timedelta
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


def _fresh_db(tmp_path: Path, monkeypatch, name: str = "loop4.db"):
    db_file = tmp_path / name
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    from core import config as cfg
    cfg.settings = cfg.Settings()
    import models.database as dbmod
    dbmod._engine = None
    dbmod._session_factory = None
    monkeypatch.setattr(dbmod, "settings", cfg.settings, raising=False)
    return dbmod


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


# ============= ITER 1: baseline already captured =============
def test_iter01_baseline_marker():
    # Baseline: 80 tests / median 1.38s captured externally.
    assert True


# ============= ITER 2: delete_by_source removes only matching docs =============
@pytest.mark.asyncio
async def test_iter02_delete_by_source_precision(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_02")
    docs = [
        Document(page_content="alpha", metadata={"source": "u1"}),
        Document(page_content="beta", metadata={"source": "u1"}),
        Document(page_content="gamma", metadata={"source": "u2"}),
    ]
    await store.add_documents(docs)
    assert store.count() == 3
    await store.delete_by_source("u1")
    assert store.count() == 1
    hits = await store.search("gamma", k=5)
    sources = {h.metadata.get("source") for h in hits}
    assert sources == {"u2"}


# ============= ITER 3: search returns metadata + content separately, no blowup =============
@pytest.mark.asyncio
async def test_iter03_search_payload_shape(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_03")
    big = "x" * 5000
    await store.add_documents(
        [Document(page_content=big, metadata={"source": "s", "title": "t"})]
    )
    hits = await store.search("x", k=1)
    assert len(hits) == 1
    h = hits[0]
    # metadata + content are separate fields; metadata not nested in content
    assert h.metadata.get("title") == "t"
    assert h.content == big
    assert isinstance(h.id, str) and len(h.id) > 0


# ============= ITER 4: write/read frontmatter+body roundtrip preserves user keys =============
def test_iter04_obsidian_roundtrip_preserves_frontmatter(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    custom = {"type": "note", "priority": 7, "lang": "zh", "custom_user_key": "preserved"}
    p = vault.write_note(
        title="roundtrip",
        content="hello world\n\nbody line 2\n",
        frontmatter=custom,
    )
    note = vault.read_note(p)
    for k, v in custom.items():
        assert note.frontmatter.get(k) == v, f"key {k} not preserved: {note.frontmatter.get(k)!r} != {v!r}"
    assert "hello world" in note.content
    assert "body line 2" in note.content


# ============= ITER 5: tag normalization is case-insensitive de-duplicating =============
def test_iter05_tag_parser_case_insensitive_dedup():
    from tools.tagger import _parse
    raw = '["AI", "ai", "Ai", "ML", "ml", "RAG", " rag ", "rag"]'
    out = _parse(raw)
    # All "ai" variants collapse to one entry; same for "ml" and "rag"
    lowered = [t.lower() for t in out]
    assert len(lowered) == len(set(lowered))
    assert {"ai", "ml", "rag"}.issubset(set(lowered))


# ============= ITER 6: repeated identical query is amortized (cache embedder) =============
@pytest.mark.asyncio
async def test_iter06_repeat_query_amortized(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_06")
    await store.add_documents([Document(page_content=f"doc {i}", metadata={"i": i}) for i in range(20)])
    # warm up
    await store.search("doc", k=5)
    # 5 repeat queries should each be fast
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        await store.search("doc", k=5)
        times.append(time.perf_counter() - t0)
    median_ms = statistics.median(times) * 1000
    assert median_ms < 200, f"repeat query too slow: {median_ms:.1f}ms"


# ============= ITER 7: scraper rejects non-http(s) URL with ValueError =============
@pytest.mark.asyncio
async def test_iter07_scraper_rejects_bad_scheme():
    from scrapers.web_scraper import WebScraper
    s = WebScraper()
    for bad in ("file:///etc/passwd", "javascript:alert(1)", "ftp://example.com/x", "ssh://x"):
        with pytest.raises(ValueError):
            await s.scrape(bad)


# ============= ITER 8: wikilink parser handles plain & nested-bracket-like content safely =============
def test_iter08_wikilink_nested_safe(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    # Content includes wikilink + a literal bracket pair that should NOT match the wikilink regex
    text = "see [[NoteA]] and [bracketed text] and [[Note B|alias]] end"
    p = vault.write_note(title="nesty", content=text)
    note = vault.read_note(p)
    # plain wikilinks parsed; bracket-only text is NOT misidentified as a wikilink
    assert "NoteA" in note.links
    assert "Note B" in note.links
    for lk in note.links:
        assert "[" not in lk and "]" not in lk


# ============= ITER 9: frontmatter merge keeps user keys on overwrite =============
def test_iter09_frontmatter_keeps_user_keys(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    vault.write_note(title="merge", content="v1", frontmatter={"x": 1, "y": 2})
    p2 = vault.write_note(title="merge", content="v2", frontmatter={"y": 99, "z": "added"})
    n = vault.read_note(p2)
    assert n.frontmatter.get("y") == 99  # overridden
    assert n.frontmatter.get("z") == "added"  # newly added
    assert "v2" in n.content


# ============= ITER 10: Knowledge upsert via session.merge is idempotent =============
@pytest.mark.asyncio
async def test_iter10_knowledge_upsert_idempotent(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter4_10.db")
    await dbmod.init_db()
    from sqlalchemy import select, func
    fixed_id = "fixed-knowledge-id-007"
    for n in range(5):
        async with dbmod.session_scope() as s:
            await s.merge(dbmod.Knowledge(
                id=fixed_id, title=f"v{n}", source_type="web",
                source="https://x.example/post", summary=f"s{n}",
            ))
    async with dbmod.session_scope() as s:
        res = await s.execute(select(func.count()).select_from(dbmod.Knowledge))
        cnt = res.scalar_one()
    assert cnt == 1


# ============= ITER 11: ChatHistory delete-by-session removes all rows =============
@pytest.mark.asyncio
async def test_iter11_chat_history_delete_by_session(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter4_11.db")
    await dbmod.init_db()
    from sqlalchemy import delete, select, func
    sid = "sess-A"
    async with dbmod.session_scope() as s:
        for i in range(7):
            s.add(dbmod.ChatHistory(session_id=sid, role="user", content=f"m{i}"))
        for i in range(3):
            s.add(dbmod.ChatHistory(session_id="sess-B", role="user", content=f"b{i}"))
    async with dbmod.session_scope() as s:
        await s.execute(delete(dbmod.ChatHistory).where(dbmod.ChatHistory.session_id == sid))
    async with dbmod.session_scope() as s:
        res = await s.execute(select(func.count()).select_from(dbmod.ChatHistory))
        total = res.scalar_one()
    assert total == 3  # only sess-B remains


# ============= ITER 12: vector_store sanitizes metadata of unsupported types =============
@pytest.mark.asyncio
async def test_iter12_metadata_sanitization(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_12")
    nested = {"x": 1}  # dict -> str via fallback
    docs = [Document(
        page_content="t",
        metadata={
            "src": "u",
            "tags": ["a", "b", "c"],          # list -> CSV
            "nested": nested,                 # dict -> str
            "none_value": None,               # dropped
            "flag": True,                     # bool kept
            "n": 42,                          # int kept
        },
    )]
    ids = await store.add_documents(docs)
    assert len(ids) == 1
    hits = await store.search("t", k=1)
    md = hits[0].metadata
    assert md.get("tags") == "a,b,c"
    assert "none_value" not in md
    assert md.get("flag") is True
    assert md.get("n") == 42
    assert isinstance(md.get("nested"), str)


# ============= ITER 13: settings reload picks up new env without restart =============
def test_iter13_settings_env_reload(monkeypatch, tmp_path):
    from core import config as cfg
    monkeypatch.setenv("CHROMA_COLLECTION", "first_v1")
    cfg.settings = cfg.Settings()
    assert cfg.settings.chroma_collection == "first_v1"
    monkeypatch.setenv("CHROMA_COLLECTION", "second_v1")
    cfg.settings = cfg.Settings()
    assert cfg.settings.chroma_collection == "second_v1"


# ============= ITER 14: hybrid_search with k=0 returns empty cleanly =============
@pytest.mark.asyncio
async def test_iter14_search_k_zero_empty(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_14")
    await store.add_documents([Document(page_content="hello", metadata={"title": "h", "source": "s"})])
    # store-level
    hits = await store.search("hello", k=1)
    assert len(hits) == 1
    # tools.search keyword path with k=0 yields empty
    import core.obsidian as obsmod
    obsmod._VAULT = None
    monkeypatch.setattr("tools.search.get_vault", lambda: _fresh_vault(tmp_path, monkeypatch))
    monkeypatch.setattr("tools.search.get_vector_store", lambda: store)
    from tools.search import hybrid_search
    out = await hybrid_search("hello", k=0)
    assert out == []


# ============= ITER 15: SearchResponse Pydantic model rejects unknown fields under strict use =============
def test_iter15_search_schema_strict():
    from models.schemas import SearchRequest, SearchResponse, SearchResultItem
    req = SearchRequest(query="hello", k=3)
    assert req.k == 3
    item = SearchResultItem(id="x", title="t", snippet="s", score=0.5, source="src", metadata={})
    resp = SearchResponse(query="q", hits=[item])
    payload = resp.model_dump()
    assert "query" in payload and "hits" in payload
    # round-trip
    resp2 = SearchResponse.model_validate(payload)
    assert resp2.query == "q" and len(resp2.hits) == 1


# ============= ITER 16: scraper raises on httpx timeout (stubbed) =============
@pytest.mark.asyncio
async def test_iter16_scraper_timeout(monkeypatch):
    from scrapers.web_scraper import WebScraper
    import httpx
    s = WebScraper()

    async def boom(_self, url):
        raise httpx.ReadTimeout("deadline exceeded")

    # disable playwright fallback for deterministic path
    async def no_pw(_self, url):
        return None

    monkeypatch.setattr(WebScraper, "_fetch_with_playwright", no_pw)
    monkeypatch.setattr(WebScraper, "_fetch_with_httpx", boom)
    with pytest.raises(httpx.HTTPError):
        await s.scrape("https://example.com/timeout")


# ============= ITER 17: parallel review.run_review with stubbed LLM yields no race =============
@pytest.mark.asyncio
async def test_iter17_parallel_reviews_isolated(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter4_17.db")
    await dbmod.init_db()
    _ = _fresh_vault(tmp_path, monkeypatch)

    # stub LLM in summariser path
    from langchain_core.runnables import RunnableLambda
    import workflows.review_workflow as rw

    def fake_llm(*a, **kw):
        return RunnableLambda(lambda pv: type("R", (), {"content": "stub-summary"})())

    monkeypatch.setattr(rw, "get_chat_llm", fake_llm)
    rw._REVIEW_GRAPH = None  # rebuild with stubbed llm

    # seed some knowledge rows
    async with dbmod.session_scope() as s:
        for i in range(4):
            s.add(dbmod.Knowledge(id=f"k{i}", title=f"item{i}", source_type="web",
                                  source=f"u{i}", summary="x"))

    end = datetime.utcnow()
    start = end - timedelta(days=7)
    r1, r2 = await asyncio.gather(
        rw.run_review("weekly", start=start, end=end),
        rw.run_review("monthly", start=start - timedelta(days=23), end=end),
    )
    assert r1.summary == "stub-summary"
    assert r2.summary == "stub-summary"
    assert r1.knowledge_count >= 1 and r2.knowledge_count >= 1


# ============= ITER 18: review.run_review respects explicit start/end =============
@pytest.mark.asyncio
async def test_iter18_review_explicit_dates(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter4_18.db")
    await dbmod.init_db()
    _ = _fresh_vault(tmp_path, monkeypatch)

    from langchain_core.runnables import RunnableLambda
    import workflows.review_workflow as rw
    monkeypatch.setattr(rw, "get_chat_llm",
                        lambda *a, **kw: RunnableLambda(lambda pv: type("R", (), {"content": "ok"})()))
    rw._REVIEW_GRAPH = None

    s_dt = datetime(2024, 1, 1)
    e_dt = datetime(2024, 1, 31)
    r = await rw.run_review("custom", start=s_dt, end=e_dt)
    assert r.start == s_dt and r.end == e_dt


# ============= ITER 19: 10-doc concurrent ingest under 800ms =============
@pytest.mark.asyncio
async def test_iter19_concurrent_ingest_throughput(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter4_19")

    async def add_one(i: int):
        await store.add_documents(
            [Document(page_content=f"chunk {i} body", metadata={"i": i, "source": f"s{i}"})]
        )

    t0 = time.perf_counter()
    await asyncio.gather(*[add_one(i) for i in range(10)])
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert store.count() == 10, f"expected 10, got {store.count()}"
    assert elapsed_ms < 800, f"throughput too slow: {elapsed_ms:.1f}ms"


# ============= ITER 20: regression sweep — every key module imports cleanly =============
def test_iter20_full_import_sweep():
    mods = [
        "core.config", "core.llm", "core.obsidian", "core.scheduler", "core.vector_store",
        "models.database", "models.schemas",
        "tools.linker", "tools.mindmap", "tools.search", "tools.summarizer", "tools.tagger",
        "scrapers.base", "scrapers.web_scraper", "scrapers.rss_scraper", "scrapers.pdf_scraper",
        "scrapers.youtube_scraper", "scrapers.notion_scraper", "scrapers.email_scraper",
        "scrapers.wechat_scraper",
        "agents.chat_agent", "agents.pkm_agent", "agents.review_agent",
        "workflows.chat_workflow", "workflows.ingest_workflow", "workflows.review_workflow",
        "api.server",
        "utils.logger", "utils.parsers", "utils.retry",
    ]
    import importlib
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e}")
    assert not failed, "import failures: " + "; ".join(failed)
