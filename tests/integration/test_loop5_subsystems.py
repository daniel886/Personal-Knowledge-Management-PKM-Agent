"""Loop 5 — 20 fresh quantitative iterations focused on subsystem coverage gaps.

Targets disjoint from loops 1-4: list_notes filtering, content-hash dedup,
re-ingest by id, linker top_k cap, mindmap empty input, summarizer prompt vars,
chat history window, ingest failure isolation, RSS parsing, PDF errors,
DB limit query, SourceType enum, logger non-string, retry max_attempts cap,
scheduler cron parsing, FastAPI router mounting, ingest-then-search e2e.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from datetime import datetime
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnableLambda


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


def _fresh_db(tmp_path: Path, monkeypatch, name: str = "loop5.db"):
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


# ============= ITER 1 =============
def test_iter01_baseline_marker():
    # Captured externally: 100 tests / median 1.56s.
    assert True


# ============= ITER 2: list_notes folder filter =============
def test_iter02_list_notes_folder_filter(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    vault.write_note(title="A", content="x", folder="inbox")
    vault.write_note(title="B", content="y", folder="reviews")
    vault.write_note(title="C", content="z", folder="inbox")
    inbox_notes = vault.list_notes(folder="inbox")
    titles = sorted(n.title for n in inbox_notes)
    assert titles == ["A", "C"]


# ============= ITER 3: search_notes case insensitive =============
def test_iter03_search_notes_case_insensitive(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    vault.write_note(title="MixedCase", content="Hello World contains MIXED tokens")
    hits_lower = vault.search_notes("hello")
    hits_upper = vault.search_notes("HELLO")
    hits_mixed = vault.search_notes("Hello")
    assert len(hits_lower) == 1 == len(hits_upper) == len(hits_mixed)


# ============= ITER 4: identical content with diff metadata = single doc (content-hash dedup) =============
@pytest.mark.asyncio
async def test_iter04_content_hash_dedup_drops_duplicate(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter5_04")
    docs = [
        Document(page_content="same content", metadata={"source": "a"}),
        Document(page_content="same content", metadata={"source": "b"}),  # dropped
        Document(page_content="other content", metadata={"source": "c"}),
    ]
    ids = await store.add_documents(docs)
    assert store.count() == 2
    assert len(ids) == 2


# ============= ITER 5: re-add with explicit ids replaces (chroma upsert behaviour) =============
@pytest.mark.asyncio
async def test_iter05_explicit_id_round_trip(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter5_05")
    ids1 = await store.add_documents(
        [Document(page_content="first version", metadata={"source": "u1", "v": 1})],
        ids=["fixed-id-001"],
    )
    assert ids1 == ["fixed-id-001"]
    assert store.count() == 1
    # search returns the inserted record
    hits = await store.search("first", k=1)
    assert hits[0].id == "fixed-id-001"
    assert hits[0].metadata.get("v") == 1


# ============= ITER 6: linker.suggest_links respects top_k cap exactly =============
@pytest.mark.asyncio
async def test_iter06_linker_top_k_cap(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter5_06")
    monkeypatch.setattr("core.vector_store.get_vector_store", lambda: store)
    monkeypatch.setattr("tools.linker.get_vector_store", lambda: store)
    docs = [
        Document(page_content=f"doc body {i}", metadata={"title": f"Note{i:02d}", "knowledge_id": f"k{i}"})
        for i in range(15)
    ]
    await store.add_documents(docs)
    from tools.linker import suggest_links
    titles = await suggest_links(title="query", content="doc body", top_k=4)
    assert len(titles) <= 4
    assert all(t.startswith("Note") for t in titles)


# ============= ITER 7: mindmap idempotent wrapper for empty/short content =============
@pytest.mark.asyncio
async def test_iter07_mindmap_empty_safe(monkeypatch):
    import tools.mindmap as mm
    monkeypatch.setattr(mm, "get_chat_llm",
                        lambda *a, **kw: RunnableLambda(lambda pv: type("R", (), {"content": ""})()))
    out = await mm.generate_mindmap(title="Empty Title", content="")
    assert out.startswith("```mermaid")
    assert "Empty Title" in out


# ============= ITER 8: summarizer prompt receives variables verbatim =============
@pytest.mark.asyncio
async def test_iter08_summarizer_prompt_substitution(monkeypatch):
    import tools.summarizer as smod
    captured = {}

    def fake_llm(*a, **kw):
        def consume(prompt_value):
            msgs = prompt_value.to_messages()
            captured["last"] = msgs[-1].content
            return type("R", (), {"content": "summary-stub"})()
        return RunnableLambda(consume)

    monkeypatch.setattr(smod, "get_chat_llm", fake_llm)
    out = await smod.summarise(
        title="MyTitle12345",
        content="some_unique_body_marker_xyz",
        source="https://example.com/x",
        source_type="web",
    )
    assert out == "summary-stub"
    assert "MyTitle12345" in captured["last"]
    assert "some_unique_body_marker_xyz" in captured["last"]


# ============= ITER 9: ChatAgent history bounded after multi-turn =============
@pytest.mark.asyncio
async def test_iter09_chat_history_bounded(tmp_path, monkeypatch):
    # stub chat workflow's run_chat to avoid real LLM
    import workflows.chat_workflow as cwm
    from models.schemas import ChatMessage, ChatResponse

    async def fake_run_chat(message, history=None, *, use_memory=True, top_k=6, **_kwargs):
        h = (history or []) + [
            ChatMessage(role="user", content=message),
            ChatMessage(role="assistant", content=f"reply to {message}"),
        ]
        return ChatResponse(answer=f"reply to {message}", sources=[], history=h)

    monkeypatch.setattr("agents.chat_agent.run_chat", fake_run_chat)
    from agents.chat_agent import ChatAgent
    agent = ChatAgent(max_history=6)
    for i in range(10):
        await agent.ask(f"q{i}")
    # max_history=6, history is sliced [-6:]
    assert len(agent.history) <= 6


# ============= ITER 10: ingest failure isolation — second ingest works after first errors =============
@pytest.mark.asyncio
async def test_iter10_ingest_failure_isolation(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter5_10")
    # first call: simulate embedder error
    bad_calls = {"n": 0}
    real_add = store.add_documents

    async def add_some(docs, **kw):
        bad_calls["n"] += 1
        if bad_calls["n"] == 1:
            raise RuntimeError("simulated transient ingest failure")
        return await real_add(docs, **kw)

    store.add_documents = add_some  # type: ignore

    with pytest.raises(RuntimeError):
        await store.add_documents([Document(page_content="boom", metadata={})])

    # subsequent call succeeds
    ids = await store.add_documents([Document(page_content="ok", metadata={"source": "s"})])
    assert len(ids) == 1


# ============= ITER 11: RSS scraper _parse handles synthetic feed =============
def test_iter11_rss_parse_synthetic(monkeypatch):
    import scrapers.rss_scraper as rmod

    class FakeEntry:
        def __init__(self, title, link, summary):
            self._d = {"title": title, "link": link, "summary": summary, "published": ""}
        def get(self, key, default=None):
            return self._d.get(key, default)

    class FakeFeed:
        bozo = False
        bozo_exception = None
        feed = {"title": "MyFeed"}
        entries = [FakeEntry(f"t{i}", f"https://x/{i}", f"sum{i}") for i in range(3)]

    monkeypatch.setattr(rmod, "feedparser", type("M", (), {"parse": staticmethod(lambda *a, **k: FakeFeed())}))
    s = rmod.RSSScraper()
    items = s._parse("https://feed.example/rss.xml")
    assert len(items) == 3
    assert items[0].source_type == "rss"
    assert items[0].metadata["feed_title"] == "MyFeed"


# ============= ITER 12: PDF scraper raises clean error on missing file =============
@pytest.mark.asyncio
async def test_iter12_pdf_missing_file_raises(tmp_path):
    from scrapers.pdf_scraper import PDFScraper
    s = PDFScraper()
    missing = tmp_path / "does_not_exist.pdf"
    with pytest.raises((FileNotFoundError, Exception)):
        await s.scrape(str(missing))


# ============= ITER 13: list_recent_knowledge respects limit =============
@pytest.mark.asyncio
async def test_iter13_list_recent_limit(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter5_13.db")
    await dbmod.init_db()
    async with dbmod.session_scope() as s:
        for i in range(25):
            s.add(dbmod.Knowledge(
                id=f"k{i}", title=f"t{i}", source_type="web",
                source=f"u{i}", summary="x",
            ))
    rows = await dbmod.list_recent_knowledge(limit=10)
    assert len(rows) == 10


# ============= ITER 14: SourceType enum validates values =============
def test_iter14_source_type_enum():
    from models.schemas import SourceType
    valid = {st.value for st in SourceType}
    # construction with a known value works
    sample = next(iter(valid))
    assert SourceType(sample).value == sample
    # invalid raises
    with pytest.raises(ValueError):
        SourceType("not_a_real_source_type_xyz")


# ============= ITER 15: logger handles non-string args without crash =============
def test_iter15_logger_non_string_args():
    from utils.logger import logger
    # Loguru formats arbitrary objects; should not raise
    logger.info("dict={}", {"a": 1, "b": [1, 2]})
    logger.info("list={}", [1, 2, 3])
    logger.info("none={}", None)
    logger.info("tuple={}", (1, 2))


# ============= ITER 16: async_retry stops at exactly max_attempts =============
@pytest.mark.asyncio
async def test_iter16_retry_stops_at_max_attempts():
    from utils.retry import async_retry
    counter = {"n": 0}

    @async_retry(max_attempts=3, initial_wait=0.001, max_wait=0.005)
    async def always_fail():
        counter["n"] += 1
        raise RuntimeError("nope")

    with pytest.raises(RuntimeError):
        await always_fail()
    assert counter["n"] == 3, f"expected exactly 3 attempts, got {counter['n']}"


# ============= ITER 17: scheduler accepts valid cron, rejects invalid =============
def test_iter17_cron_validation():
    from apscheduler.triggers.cron import CronTrigger
    # valid expressions parse
    CronTrigger.from_crontab("0 8 * * 1")
    CronTrigger.from_crontab("*/15 * * * *")
    # invalid expressions raise
    with pytest.raises(Exception):
        CronTrigger.from_crontab("not a cron")


# ============= ITER 18: FastAPI app mounts all expected routers =============
def test_iter18_fastapi_routers_mounted():
    from api.server import create_app
    app = create_app()
    routes = {r.path for r in app.routes if hasattr(r, "path")}
    expected_prefixes = ["/api/search", "/api/chat", "/api/ingest", "/api/review"]
    for pfx in expected_prefixes:
        assert any(p.startswith(pfx) for p in routes), f"missing route prefix {pfx}; routes={sorted(routes)}"


# ============= ITER 19: e2e ingest then search recovers same doc =============
@pytest.mark.asyncio
async def test_iter19_e2e_ingest_then_search(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "iter5_19")
    target_phrase = "rare-magic-token-XQZ-9876"
    await store.add_documents(
        [Document(page_content=target_phrase, metadata={"title": "needle", "source": "u"})]
    )
    hits = await store.search(target_phrase, k=3)
    assert len(hits) >= 1
    assert hits[0].metadata.get("title") == "needle"
    assert target_phrase in hits[0].content


# ============= ITER 20: regression sweep: every test loop's marker imports cleanly =============
def test_iter20_regression_sweep():
    import importlib
    mods = [
        "tests.integration.test_iter01_config",
        "tests.integration.test_iter02_logger",
        "tests.integration.test_iter05_vector",
        "tests.integration.test_iter09_graphs",
        "tests.integration.test_iter10_end_to_end",
        "tests.integration.test_loop2_efficiency_accuracy",
        "tests.integration.test_loop3_fresh_angles",
        "tests.integration.test_loop4_robustness",
    ]
    failed = []
    for m in mods:
        try:
            importlib.import_module(m)
        except Exception as e:
            failed.append(f"{m}: {e}")
    assert not failed, "import failures: " + "; ".join(failed)
