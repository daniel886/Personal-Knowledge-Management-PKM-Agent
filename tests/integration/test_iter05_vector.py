"""Iter 5 — VectorStore add/search/delete using a deterministic fake embedder."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """Hash-based deterministic embedder, 16-dim, no network."""

    DIM = 16

    def _embed(self, text: str) -> list[float]:
        h = hashlib.sha256(text.encode("utf-8")).digest()
        # take first 16 bytes; normalize to unit vector
        vec = [b / 255.0 for b in h[: self.DIM]]
        norm = sum(x * x for x in vec) ** 0.5 or 1.0
        return [x / norm for x in vec]

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.mark.asyncio
async def test_vector_store_add_search(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION", "iter5")

    from core.vector_store import VectorStore
    store = VectorStore(collection_name="iter5", embeddings=FakeEmbeddings())

    docs = [
        Document(page_content="人工智能与机器学习", metadata={
            "title": "AI 笔记", "source": "x://1", "source_type": "web", "tags": "ai,ml"
        }),
        Document(page_content="天气预报与厨房菜谱", metadata={
            "title": "其他", "source": "x://2", "source_type": "web", "tags": "misc"
        }),
    ]
    ids = await store.add_documents(docs)
    assert len(ids) == 2
    assert store.count() == 2

    hits = await store.search("机器学习", k=2)
    assert hits, "search returned no hits"
    titles = [h.metadata.get("title") for h in hits]
    assert "AI 笔记" in titles

    # delete by source
    await store.delete_by_source("x://1")
    assert store.count() == 1

    # reset
    await store.reset()
    assert store.count() == 0


@pytest.mark.asyncio
async def test_vector_metadata_sanitisation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma2"))
    monkeypatch.setenv("CHROMA_COLLECTION", "iter5b")

    from core.vector_store import VectorStore
    store = VectorStore(collection_name="iter5b", embeddings=FakeEmbeddings())

    doc = Document(
        page_content="x",
        metadata={
            "title": "T",
            "source": "s",
            "source_type": "web",
            "tags": ["a", "b"],         # list -> joined
            "nested": {"k": 1},          # dict -> str
            "none": None,                # dropped
        },
    )
    ids = await store.add_documents([doc])
    assert ids
