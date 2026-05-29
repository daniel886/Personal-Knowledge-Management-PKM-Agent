"""Chroma-backed vector store wrapper used across the project."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable
from uuid import uuid4

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from core.config import settings
from core.llm import get_embeddings
from utils.logger import logger

# module-level client cache keyed by persist_dir to avoid re-opening sqlite
_CLIENT_CACHE: dict[str, "chromadb.api.client.Client"] = {}


def _get_client(path: str) -> "chromadb.api.client.Client":
    if path not in _CLIENT_CACHE:
        _CLIENT_CACHE[path] = chromadb.PersistentClient(
            path=path,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _CLIENT_CACHE[path]


@dataclass
class SearchHit:
    id: str
    content: str
    metadata: dict[str, Any]
    score: float


class VectorStore:
    """Async-friendly facade around a persistent Chroma collection."""

    def __init__(
        self,
        collection_name: str | None = None,
        *,
        embeddings: Embeddings | None = None,
    ) -> None:
        self.collection_name = collection_name or settings.chroma_collection
        self._client = _get_client(settings.chroma_persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._embeddings = embeddings if embeddings is not None else get_embeddings()
        logger.info(
            f"VectorStore ready @ {settings.chroma_persist_dir} "
            f"collection='{self.collection_name}' size={self._collection.count()}"
        )

    # ---------------- ingest ----------------
    async def add_documents(
        self,
        documents: Iterable[Document],
        *,
        ids: list[str] | None = None,
        dedup: bool = True,
    ) -> list[str]:
        docs = list(documents)
        if not docs:
            return []
        if dedup:
            seen: set[str] = set()
            unique: list[Document] = []
            for d in docs:
                import hashlib

                key = hashlib.sha1(d.page_content.encode("utf-8")).hexdigest()
                if key in seen:
                    continue
                seen.add(key)
                unique.append(d)
            docs = unique
        ids = ids or [str(uuid4()) for _ in docs]
        texts = [d.page_content for d in docs]
        metadatas = [self._sanitize_metadata(d.metadata) for d in docs]

        embeddings = await asyncio.to_thread(self._embeddings.embed_documents, texts)
        await asyncio.to_thread(
            self._collection.add,
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        logger.info(f"Indexed {len(docs)} chunks into '{self.collection_name}'")
        return ids

    # ---------------- query ----------------
    async def search(
        self,
        query: str,
        *,
        k: int = 5,
        where: dict[str, Any] | None = None,
    ) -> list[SearchHit]:
        embedding = await asyncio.to_thread(self._embeddings.embed_query, query)
        result = await asyncio.to_thread(
            self._collection.query,
            query_embeddings=[embedding],
            n_results=k,
            where=where,
        )
        hits: list[SearchHit] = []
        ids = result.get("ids", [[]])[0]
        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        dists = result.get("distances", [[]])[0]
        for i, doc_id in enumerate(ids):
            score = 1.0 - float(dists[i]) if dists else 0.0
            hits.append(
                SearchHit(
                    id=doc_id,
                    content=docs[i],
                    metadata=metas[i] or {},
                    score=score,
                )
            )
        return hits

    # ---------------- maintenance ----------------
    async def delete_by_source(self, source: str) -> None:
        await asyncio.to_thread(self._collection.delete, where={"source": source})
        logger.info(f"Removed vectors for source={source}")

    async def reset(self) -> None:
        await asyncio.to_thread(self._client.delete_collection, self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.warning(f"Vector collection '{self.collection_name}' reset")

    def count(self) -> int:
        return self._collection.count()

    # ---------------- helpers ----------------
    @staticmethod
    def _sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
        """Chroma only accepts primitive values."""
        clean: dict[str, Any] = {}
        for k, v in (meta or {}).items():
            if v is None:
                continue
            if isinstance(v, (str, int, float, bool)):
                clean[k] = v
            elif isinstance(v, (list, tuple)):
                clean[k] = ",".join(map(str, v))
            else:
                clean[k] = str(v)
        return clean


_VECTOR_STORE: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _VECTOR_STORE
    if _VECTOR_STORE is None:
        _VECTOR_STORE = VectorStore()
    return _VECTOR_STORE
