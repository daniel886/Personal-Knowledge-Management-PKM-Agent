"""Search facade combining vector + keyword search."""
from __future__ import annotations

from typing import Any

from core.obsidian import get_vault
from core.vector_store import SearchHit, get_vector_store
from models.schemas import SearchResultItem
from utils.logger import logger


def _hit_to_item(hit: SearchHit) -> SearchResultItem:
    md = hit.metadata or {}
    return SearchResultItem(
        id=hit.id,
        title=md.get("title") or "(untitled)",
        snippet=hit.content[:300],
        score=round(hit.score, 4),
        source=md.get("source") or "",
        metadata=md,
    )


async def vector_search(
    query: str,
    *,
    k: int = 5,
    source_type: str | None = None,
    tags: list[str] | None = None,
) -> list[SearchResultItem]:
    if k <= 0:
        return []
    where: dict[str, Any] = {}
    if source_type:
        where["source_type"] = source_type
    hits = await get_vector_store().search(query, k=k, where=where or None)

    if tags:
        wanted = {t.lower() for t in tags}
        hits = [
            h for h in hits
            if wanted.issubset(set(str(h.metadata.get("tags", "")).lower().split(",")))
        ]
    return [_hit_to_item(h) for h in hits]


async def keyword_search(query: str, *, k: int = 5) -> list[SearchResultItem]:
    """Simple keyword search across the Obsidian vault."""
    if k <= 0:
        return []
    matches = get_vault().search_notes(query)[:k]
    out: list[SearchResultItem] = []
    for note in matches:
        out.append(
            SearchResultItem(
                id=note.relative_path,
                title=note.title,
                snippet=note.content[:300],
                score=1.0,
                source=note.relative_path,
                metadata={"tags": note.tags, "links": note.links},
            )
        )
    return out


async def hybrid_search(
    query: str,
    *,
    k: int = 6,
    source_type: str | None = None,
) -> list[SearchResultItem]:
    """Combine vector + keyword search and de-duplicate."""
    vec = await vector_search(query, k=k, source_type=source_type)
    kw = await keyword_search(query, k=k)
    seen: set[str] = set()
    merged: list[SearchResultItem] = []
    for item in [*vec, *kw]:
        key = item.title or item.id
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= k:
            break
    logger.debug(f"Hybrid search '{query}' -> {len(merged)} hits")
    return merged
