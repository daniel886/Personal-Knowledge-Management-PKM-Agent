"""Iter 10 — End-to-end mocked ingest workflow.

Run the LangGraph ingest pipeline with all network-bound tools (LLM,
embeddings) replaced by deterministic fakes. Verify:
- chunks land in the (fake-embedded) vector store
- an Obsidian markdown note is written
- a Knowledge row + KnowledgeChunk rows are persisted in SQLite
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    DIM = 16

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        vec = [b / 255.0 for b in h[: self.DIM]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.mark.asyncio
async def test_end_to_end_ingest(tmp_path: Path, monkeypatch):
    # 1) point all stateful directories at tmp_path BEFORE importing anything
    db_file = tmp_path / "iter10.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_file}")
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "iter10")
    monkeypatch.setenv("OBSIDIAN_VAULT_PATH", str(tmp_path / "vault"))

    # 2) refresh cached settings + module-level singletons so they pick up the new env
    from core import config as cfg

    cfg.settings = cfg.Settings()  # rebuild to pick up monkeypatched env

    import core.vector_store as vsmod
    import core.obsidian as obsmod
    import models.database as dbmod
    import workflows.ingest_workflow as wf

    vsmod._VECTOR_STORE = None
    obsmod._VAULT = None
    dbmod._engine = None
    dbmod._session_factory = None
    wf._INGEST_GRAPH = None

    # also ensure modules that captured `settings` reference the refreshed object
    monkeypatch.setattr(vsmod, "settings", cfg.settings, raising=False)
    monkeypatch.setattr(obsmod, "settings", cfg.settings, raising=False)
    monkeypatch.setattr(dbmod, "settings", cfg.settings, raising=False)

    # 3) inject FakeEmbeddings into the vector-store singleton
    vsmod._VECTOR_STORE = vsmod.VectorStore(
        collection_name="iter10", embeddings=FakeEmbeddings()
    )

    # 4) initialise the SQLite schema
    await dbmod.init_db()

    # 5) stub out LLM-bound tools (imported by name into the workflow)
    async def fake_summarise(*, title, content, source=None, source_type=None):
        return f"## TL;DR\n{title} 摘要 (mocked).\n\n## 关键知识点\n- 点1\n- 点2"

    async def fake_tags(*, title, content, max_chars=8000):
        return ["机器学习", "测试", "PKM"]

    async def fake_links(*, title, content, top_k=5, exclude_id=None):
        return ["相关笔记 A", "相关笔记 B"]

    async def fake_mindmap(*, title, content, max_chars=6000):
        return "```mermaid\nmindmap\n  root((" + title + "))\n```"

    monkeypatch.setattr(wf, "summarise", fake_summarise)
    monkeypatch.setattr(wf, "generate_tags", fake_tags)
    monkeypatch.setattr(wf, "suggest_links", fake_links)
    monkeypatch.setattr(wf, "generate_mindmap", fake_mindmap)

    # 6) build a synthetic scraped document and run the workflow
    from scrapers.base import ScrapedDocument
    from models.schemas import SourceType

    doc = ScrapedDocument(
        title="AI 与个人知识管理：端到端测试",
        content=("人工智能与个人知识管理的结合正在重塑学习方式。" * 30),
        source="https://example.com/iter10",
        source_type="web",
        metadata={"author": "tester"},
    )

    result = await wf.ingest_document(doc)

    # 7) assertions
    assert result.knowledge_id, "knowledge_id missing"
    assert result.chunks_indexed > 0, "no chunks indexed"
    assert result.tags == ["机器学习", "测试", "PKM"]
    assert result.links == ["相关笔记 A", "相关笔记 B"]
    assert result.summary.startswith("## TL;DR")
    assert result.source_type == SourceType.WEB

    # obsidian file exists on disk
    assert result.obsidian_path is not None
    note_path = Path(result.obsidian_path)
    assert note_path.exists(), f"obsidian note not written: {note_path}"
    body = note_path.read_text(encoding="utf-8")
    assert "AI 与个人知识管理" in body
    assert "智能摘要" in body
    assert "机器学习" in body or "tags:" in body

    # vector store has the indexed chunks
    assert vsmod._VECTOR_STORE.count() == result.chunks_indexed

    # SQLite row + chunk rows exist
    from sqlalchemy import select
    from models.database import Knowledge, KnowledgeChunk, session_scope

    async with session_scope() as session:
        rows = (await session.execute(select(Knowledge))).scalars().all()
        assert len(rows) == 1
        assert rows[0].title == doc.title
        assert rows[0].chunks_indexed == result.chunks_indexed

        chunk_rows = (await session.execute(select(KnowledgeChunk))).scalars().all()
        assert len(chunk_rows) == result.chunks_indexed

    # 8) retrieval round-trip on the vector store
    hits = await vsmod._VECTOR_STORE.search("个人知识管理", k=3)
    assert hits, "vector search returned no hits"
    titles = [h.metadata.get("title") for h in hits]
    assert doc.title in titles
