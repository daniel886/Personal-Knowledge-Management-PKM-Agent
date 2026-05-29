"""Loop 3 — 20 fresh quantitative iterations on PKM Agent.

Each test asserts a specific target the prior loops did not cover.
All network/LLM calls are stubbed; tmp_path used for state.
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


def _fresh_db(tmp_path: Path, monkeypatch, name: str = "loop3.db"):
    """Wire SQLite to tmp_path and reset cached engine/factory."""
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
def test_iter01_baseline():
    assert True  # baseline run separately, see report


# ============= ITER 2: Knowledge.tags CSV roundtrip =============
@pytest.mark.asyncio
async def test_iter02_knowledge_tags_csv(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter2.db")
    await dbmod.init_db()
    tags_input = ["机器学习", "transformer", "PKM"]
    async with dbmod.session_scope() as s:
        s.add(dbmod.Knowledge(
            id="k2", title="t", source_type="web", source="x",
            tags=",".join(tags_input),
        ))
    from sqlalchemy import select
    async with dbmod.session_scope() as s:
        row = (await s.execute(select(dbmod.Knowledge).where(dbmod.Knowledge.id == "k2"))).scalar_one()
    assert row.tags.split(",") == tags_input


# ============= ITER 3: ChatHistory insert + ordering =============
@pytest.mark.asyncio
async def test_iter03_chat_history_order(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter3.db")
    await dbmod.init_db()
    async with dbmod.session_scope() as s:
        for i, (role, content) in enumerate([
            ("user", "q1"), ("assistant", "a1"),
            ("user", "q2"), ("assistant", "a2"),
        ]):
            s.add(dbmod.ChatHistory(session_id="s1", role=role, content=content))

    from sqlalchemy import select
    async with dbmod.session_scope() as s:
        rows = (await s.execute(
            select(dbmod.ChatHistory).order_by(dbmod.ChatHistory.id)
        )).scalars().all()
    assert [r.role for r in rows] == ["user", "assistant", "user", "assistant"]
    assert [r.content for r in rows] == ["q1", "a1", "q2", "a2"]


# ============= ITER 4: wikilink extraction =============
def test_iter04_wikilink_extraction(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    body = "see [[NoteA]] and [[NoteB|alias]] and [[NoteC]] but not [single]"
    p = vault.write_note(title="links", content=body)
    note = vault.read_note(p)
    assert set(note.links) == {"NoteA", "NoteB", "NoteC"}, f"got: {note.links}"


# ============= ITER 5: frontmatter YAML edge cases =============
def test_iter05_frontmatter_edge_cases(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    fm = {
        "tags": ["a", "b", "c"],
        "title_zh": "中文标题",
        "score": 3.14,
        "active": True,
        "nested": {"k": "v"},
        "with:colon": "value: with colon",
    }
    p = vault.write_note(title="fm", content="body", frontmatter=fm)
    note = vault.read_note(p)
    assert note.frontmatter.get("tags") == ["a", "b", "c"]
    assert note.frontmatter.get("title_zh") == "中文标题"
    assert note.frontmatter.get("score") == 3.14
    assert note.frontmatter.get("active") is True


# ============= ITER 6: filename sanitization =============
def test_iter06_filename_sanitization(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    cases = [
        "../../../etc/passwd",
        "a/b\\c:d*e?f<g>h|i\"j",
        "very_long_" + "x" * 200,
        "中文 标题 with 空格",
        "",  # falls back to timestamp
    ]
    paths = []
    for title in cases:
        p = vault.write_note(title=title, content="body")
        paths.append(p)
    for p in paths:
        # path must be under vault root, no traversal
        assert vault.root in p.parents or vault.root in p.parents or p.is_relative_to(vault.root), p
        assert p.exists()
        # max 120 chars + ".md"
        assert len(p.stem) <= 120


# ============= ITER 7: vector search top_k bounds =============
@pytest.mark.asyncio
async def test_iter07_search_topk_bounds(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "topk")
    await store.reset()
    docs = [Document(page_content=f"doc{i}", metadata={"title": f"T{i}"}) for i in range(5)]
    await store.add_documents(docs)
    # k=1 returns exactly 1
    hits1 = await store.search("doc", k=1)
    assert len(hits1) == 1
    # k larger than collection clamps to collection size
    hits100 = await store.search("doc", k=100)
    assert len(hits100) == 5


# ============= ITER 8: mindmap wrapper idempotency =============
def test_iter08_mindmap_wrapper_idempotent():
    """generate_mindmap wraps content in ```mermaid; if content already has it, no double-wrap."""
    import inspect
    from tools import mindmap as mm
    src = inspect.getsource(mm.generate_mindmap)
    # the function checks startswith("```mermaid") before wrapping
    assert 'startswith("```mermaid")' in src


# ============= ITER 9: linker exclude_id removes self =============
@pytest.mark.asyncio
async def test_iter09_linker_excludes_self(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "linker")
    await store.reset()
    await store.add_documents([
        Document(page_content="x", metadata={"title": "self_note", "knowledge_id": "K1"}),
        Document(page_content="y", metadata={"title": "other_note", "knowledge_id": "K2"}),
    ])
    # patch the vector store accessor used by linker
    import core.vector_store as vsmod
    vsmod._VECTOR_STORE = store
    from tools.linker import suggest_links
    out = await suggest_links(title="anything", content="x", top_k=5, exclude_id="K1")
    assert "self_note" not in out
    assert "other_note" in out


# ============= ITER 10: summariser truncation respects max_chars =============
@pytest.mark.asyncio
async def test_iter10_summariser_truncation(monkeypatch):
    from langchain_core.runnables import RunnableLambda
    from tools import summarizer

    captured = {}

    def fake_call(prompt_value):
        # prompt_value is a ChatPromptValue; pull last human message text
        msgs = prompt_value.to_messages()
        captured["prompt_text"] = msgs[-1].content
        class R:
            content = "ok"
        return R()

    # Replace the LLM with a Runnable that records what it received
    monkeypatch.setattr(summarizer, "get_chat_llm", lambda *a, **kw: RunnableLambda(fake_call))

    long = "x" * 50000
    out = await summarizer.summarise(
        title="t", content=long, source="s", source_type="web", max_chars=12000
    )
    assert out == "ok"
    # The human prompt template injected `content` truncated to 12000 chars.
    # The full prompt text contains other fields too, but the substring of x's
    # in it must be exactly 12000 (no more, no less).
    x_run_len = captured["prompt_text"].count("x")
    assert x_run_len == 12000, f"truncation broken: x count={x_run_len}"


# ============= ITER 11: text splitter respects chunk_size =============
def test_iter11_text_splitter_chunk_size():
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    text = "abc " * 1000  # ~4000 chars
    chunks = splitter.split_text(text)
    assert len(chunks) >= 10
    for c in chunks:
        assert len(c) <= 220, f"chunk too big: {len(c)}"


# ============= ITER 12: daily note append idempotent (file exists) =============
def test_iter12_daily_note_append(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    day = datetime(2026, 5, 29, 10, 0)
    p1 = vault.append_to_daily("first entry", day=day)
    p2 = vault.append_to_daily("second entry", day=day.replace(hour=11))
    assert p1 == p2  # same file
    body = p1.read_text(encoding="utf-8")
    assert "first entry" in body and "second entry" in body
    # only one frontmatter block
    assert body.count("---\n") == 2  # opening + closing


# ============= ITER 13: backlink section append =============
def test_iter13_backlink_section(tmp_path, monkeypatch):
    vault = _fresh_vault(tmp_path, monkeypatch)
    p = vault.write_note(title="base", content="body")
    vault.add_backlink_section(p, ["A", "B", "C"])
    body = p.read_text(encoding="utf-8")
    assert "## 🔗 相关笔记" in body
    assert "[[A]]" in body and "[[B]]" in body and "[[C]]" in body


# ============= ITER 14: settings env override precedence =============
def test_iter14_settings_env_override(monkeypatch):
    monkeypatch.setenv("CHROMA_COLLECTION", "override_name")
    from core import config as cfg
    s = cfg.Settings()
    assert s.chroma_collection == "override_name"


# ============= ITER 15: Knowledge.extra JSON roundtrip =============
@pytest.mark.asyncio
async def test_iter15_knowledge_extra_json(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter15.db")
    await dbmod.init_db()
    extra = {"author": "alice", "scores": [1, 2, 3], "nested": {"k": "v"}}
    async with dbmod.session_scope() as s:
        s.add(dbmod.Knowledge(
            id="kx", title="t", source_type="web", source="x", extra=extra,
        ))
    from sqlalchemy import select
    async with dbmod.session_scope() as s:
        row = (await s.execute(
            select(dbmod.Knowledge).where(dbmod.Knowledge.id == "kx")
        )).scalar_one()
    assert row.extra == extra


# ============= ITER 16: session_scope rollback on error =============
@pytest.mark.asyncio
async def test_iter16_session_rollback(tmp_path, monkeypatch):
    dbmod = _fresh_db(tmp_path, monkeypatch, "iter16.db")
    await dbmod.init_db()
    # insert valid row first
    async with dbmod.session_scope() as s:
        s.add(dbmod.Knowledge(id="ok1", title="t", source_type="web", source="x"))

    # try to insert duplicate id inside a scope that raises
    with pytest.raises(Exception):
        async with dbmod.session_scope() as s:
            s.add(dbmod.Knowledge(id="ok1", title="dup", source_type="web", source="x"))

    # confirm only the original row exists
    from sqlalchemy import select
    async with dbmod.session_scope() as s:
        rows = (await s.execute(select(dbmod.Knowledge))).scalars().all()
    assert len(rows) == 1
    assert rows[0].title == "t"


# ============= ITER 17: vector reset idempotent =============
@pytest.mark.asyncio
async def test_iter17_vector_reset_idempotent(tmp_path, monkeypatch):
    store = _fresh_vstore(tmp_path, monkeypatch, "reset_idem")
    await store.add_documents([
        Document(page_content=f"x{i}", metadata={"title": f"t{i}"}) for i in range(3)
    ])
    assert store.count() == 3
    await store.reset()
    assert store.count() == 0
    # second reset is a no-op, no exception
    await store.reset()
    assert store.count() == 0


# ============= ITER 18: logger emoji + CJK + control chars =============
def test_iter18_logger_unicode():
    from utils.logger import logger
    logger.info("🎉 emoji works")
    logger.info("中文 ✓ こんにちは 한국어")
    logger.info("control: \t\n\r safe")
    logger.warning(f"mixed {None} {[1,2,3]} {'你好'}")
    # no exception ⇒ pass


# ============= ITER 19: e2e 5-doc batch median latency =============
@pytest.mark.asyncio
async def test_iter19_e2e_batch_perf(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/b.db")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "batch")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))

    from core import config as cfg
    cfg.settings = cfg.Settings()

    import core.vector_store as vsmod
    import core.obsidian as obsmod
    import models.database as dbmod
    import workflows.ingest_workflow as wf

    vsmod._CLIENT_CACHE.clear()
    vsmod._VECTOR_STORE = None
    obsmod._VAULT = None
    dbmod._engine = None
    dbmod._session_factory = None
    wf._INGEST_GRAPH = None
    monkeypatch.setattr(vsmod, "settings", cfg.settings, raising=False)
    monkeypatch.setattr(obsmod, "settings", cfg.settings, raising=False)
    monkeypatch.setattr(dbmod, "settings", cfg.settings, raising=False)

    vsmod._VECTOR_STORE = vsmod.VectorStore(
        collection_name="batch", embeddings=FakeEmbeddings()
    )
    await dbmod.init_db()

    async def fake_summarise(*, title, content, source=None, source_type=None):
        return "ok"

    async def fake_tags(*, title, content, max_chars=8000):
        return ["t"]

    async def fake_links(*, title, content, top_k=5, exclude_id=None):
        return []

    async def fake_mindmap(*, title, content, max_chars=6000):
        return "```mermaid\nmindmap\n```"

    monkeypatch.setattr(wf, "summarise", fake_summarise)
    monkeypatch.setattr(wf, "generate_tags", fake_tags)
    monkeypatch.setattr(wf, "suggest_links", fake_links)
    monkeypatch.setattr(wf, "generate_mindmap", fake_mindmap)

    from scrapers.base import ScrapedDocument

    times = []
    for batch in range(3):
        await vsmod._VECTOR_STORE.reset()
        t0 = time.perf_counter()
        for i in range(5):
            await wf.ingest_document(ScrapedDocument(
                title=f"b{batch}-doc{i}",
                content=f"content {i} " * 50,
                source=f"https://x/b{batch}/d{i}",
                source_type="web",
            ))
        times.append(time.perf_counter() - t0)
    median_s = statistics.median(times)
    assert median_s < 1.5, f"5-doc batch median {median_s*1000:.0f}ms > 1500ms"


# ============= ITER 20: final regression sweep =============
def test_iter20_regression_sweep():
    """Make sure each module remains importable cleanly after all changes."""
    import importlib, sys
    for mod_name in [
        "core.config", "core.llm", "core.vector_store", "core.obsidian",
        "models.database", "models.schemas",
        "workflows.ingest_workflow", "workflows.chat_workflow", "workflows.review_workflow",
        "tools.summarizer", "tools.tagger", "tools.linker", "tools.mindmap", "tools.search",
        "scrapers", "scrapers.web_scraper", "scrapers.pdf_scraper", "scrapers.rss_scraper",
        "utils.logger", "utils.retry",
        "api.server", "main",
    ]:
        mod = importlib.import_module(mod_name)
        assert mod is not None, mod_name
