"""Knowledge graph builder.

Constructs a node/edge graph from the SQLite metadata, with optional
vector-similarity edges fetched from the Chroma store.

Edge types:
- "tag":         two notes share >= min_tag_overlap tag(s)
- "wikilink":    a note's `extra.related` list points to another note id/title
- "similarity":  cosine similarity above threshold (requires vector store)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from sqlalchemy import select

from models.database import Knowledge, session_scope
from models.schemas import GraphEdge, GraphNode, GraphResponse, GraphScope
from utils.logger import logger


def _split_tags(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [t.strip() for t in raw.split(",") if t.strip()]


def _node_from_row(row: Knowledge) -> GraphNode:
    return GraphNode(
        id=row.id,
        title=row.title,
        source_type=row.source_type,
        tags=_split_tags(row.tags),
        created_at=row.created_at,
        group=row.source_type,
    )


def _tag_edges(
    nodes: list[GraphNode],
    *,
    min_overlap: int = 1,
) -> list[GraphEdge]:
    """Generate one edge per (node_a, node_b) pair sharing >= min_overlap tag(s).

    Weight = number of shared tags. Edge label = first shared tag (display hint).
    """
    by_tag: dict[str, list[str]] = defaultdict(list)
    for n in nodes:
        for t in n.tags:
            by_tag[t.lower()].append(n.id)

    pair_shared: dict[tuple[str, str], list[str]] = defaultdict(list)
    for tag, ids in by_tag.items():
        if len(ids) < 2:
            continue
        ids_sorted = sorted(set(ids))
        for i in range(len(ids_sorted)):
            for j in range(i + 1, len(ids_sorted)):
                pair = (ids_sorted[i], ids_sorted[j])
                pair_shared[pair].append(tag)

    edges: list[GraphEdge] = []
    for (a, b), shared in pair_shared.items():
        if len(shared) < min_overlap:
            continue
        edges.append(
            GraphEdge(
                source=a,
                target=b,
                weight=float(len(shared)),
                edge_type="tag",
                label=shared[0],
            )
        )
    return edges


def _wikilink_edges(rows: Iterable[Knowledge], id_set: set[str]) -> list[GraphEdge]:
    """Edges driven by `extra.related` (list of titles or ids in the ingest workflow)."""
    title_to_id = {}
    rows_list = list(rows)
    for r in rows_list:
        title_to_id[r.title.lower()] = r.id

    edges: list[GraphEdge] = []
    for r in rows_list:
        related = (r.extra or {}).get("related") or []
        if not isinstance(related, list):
            continue
        for target in related:
            if not isinstance(target, str) or not target.strip():
                continue
            target_id: str | None = None
            if target in id_set:
                target_id = target
            else:
                target_id = title_to_id.get(target.lower())
            if not target_id or target_id == r.id:
                continue
            edges.append(
                GraphEdge(
                    source=r.id,
                    target=target_id,
                    weight=1.0,
                    edge_type="wikilink",
                    label=target,
                )
            )
    return edges


async def _fetch_rows(
    *,
    scope: GraphScope,
    limit: int,
    tag: str | None = None,
) -> list[Knowledge]:
    async with session_scope() as s:
        stmt = select(Knowledge).order_by(Knowledge.created_at.desc())
        if scope == GraphScope.RECENT:
            stmt = stmt.limit(limit)
        elif scope == GraphScope.TAG and tag:
            # Use a LIKE filter on the comma-separated tags column
            stmt = stmt.where(Knowledge.tags.like(f"%{tag}%")).limit(limit)
        else:
            stmt = stmt.limit(limit)
        rows = await s.execute(stmt)
        return list(rows.scalars())


async def _similarity_edges(
    nodes: list[GraphNode],
    *,
    threshold: float,
    top_k_per_node: int = 5,
) -> list[GraphEdge]:
    """Vector-similarity edges using the existing VectorStore similarity_search_with_score.

    This is a best-effort enrichment: any error from the vector store is logged
    and an empty list is returned (graph still renders from tag/wikilink edges).
    """
    try:
        from core.vector_store import get_vector_store
    except Exception as exc:  # pragma: no cover
        logger.warning(f"VectorStore unavailable for similarity edges: {exc}")
        return []

    try:
        store = get_vector_store()
    except Exception as exc:
        logger.warning(f"VectorStore init failed: {exc}")
        return []

    id_set = {n.id for n in nodes}
    title_lookup = {n.id: n.title for n in nodes}
    seen: set[tuple[str, str]] = set()
    edges: list[GraphEdge] = []

    for n in nodes:
        query = title_lookup.get(n.id) or n.id
        try:
            hits = await store.similarity_search(query, k=top_k_per_node + 1)
        except Exception as exc:
            logger.debug(f"similarity_search failed for {n.id}: {exc}")
            continue
        for hit in hits or []:
            target_id = (hit.get("knowledge_id") if isinstance(hit, dict) else None) or (
                getattr(hit, "metadata", {}) or {}
            ).get("knowledge_id")
            score = (hit.get("score") if isinstance(hit, dict) else None) or (
                getattr(hit, "score", None)
            )
            if not target_id or target_id == n.id or target_id not in id_set:
                continue
            if score is None or score < threshold:
                continue
            pair = tuple(sorted((n.id, target_id)))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append(
                GraphEdge(
                    source=pair[0],
                    target=pair[1],
                    weight=float(score),
                    edge_type="similarity",
                    label=f"sim={score:.2f}",
                )
            )
    return edges


def _apply_weights(nodes: list[GraphNode], edges: list[GraphEdge]) -> None:
    """Compute node.weight as the count of incident edges (degree)."""
    degree: dict[str, int] = defaultdict(int)
    for e in edges:
        degree[e.source] += 1
        degree[e.target] += 1
    for n in nodes:
        n.weight = max(1, degree.get(n.id, 0))


async def build_graph(
    *,
    scope: GraphScope = GraphScope.ALL,
    limit: int = 200,
    tag: str | None = None,
    include_similarity: bool = False,
    similarity_threshold: float = 0.75,
    min_tag_overlap: int = 1,
) -> GraphResponse:
    """Construct the knowledge graph for the given scope."""
    rows = await _fetch_rows(scope=scope, limit=limit, tag=tag)
    nodes = [_node_from_row(r) for r in rows]
    id_set = {n.id for n in nodes}

    tag_edges = _tag_edges(nodes, min_overlap=min_tag_overlap)
    wiki_edges = _wikilink_edges(rows, id_set)
    sim_edges: list[GraphEdge] = []
    if include_similarity and nodes:
        sim_edges = await _similarity_edges(nodes, threshold=similarity_threshold)

    edges = tag_edges + wiki_edges + sim_edges
    _apply_weights(nodes, edges)

    stats = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "tag_edges": len(tag_edges),
        "wikilink_edges": len(wiki_edges),
        "similarity_edges": len(sim_edges),
        "scope": scope.value,
    }
    logger.info(
        f"Graph built: {stats['node_count']} nodes, "
        f"{stats['edge_count']} edges (scope={scope.value})"
    )
    return GraphResponse(
        scope=scope.value, nodes=nodes, edges=edges, stats=stats
    )
