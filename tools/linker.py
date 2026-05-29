"""Find candidate Obsidian wiki-links for a new note via vector similarity."""
from __future__ import annotations

from core.vector_store import get_vector_store
from utils.logger import logger


async def suggest_links(
    *, title: str, content: str, top_k: int = 5, exclude_id: str | None = None
) -> list[str]:
    """Suggest titles of related notes by querying the vector store."""
    store = get_vector_store()
    query = f"{title}\n{content[:2000]}"
    hits = await store.search(query, k=top_k * 2)

    seen_titles: list[str] = []
    for h in hits:
        if exclude_id and h.metadata.get("knowledge_id") == exclude_id:
            continue
        t = h.metadata.get("title")
        if t and t not in seen_titles:
            seen_titles.append(t)
        if len(seen_titles) >= top_k:
            break
    logger.debug(f"Link suggestions for '{title}': {seen_titles}")
    return seen_titles
