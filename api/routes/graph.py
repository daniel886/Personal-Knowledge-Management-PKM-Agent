"""Knowledge-graph API route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from models.schemas import GraphResponse, GraphScope
from tools.graph_builder import build_graph
from utils.logger import logger

router = APIRouter(prefix="/api/graph", tags=["graph"])


@router.get("", response_model=GraphResponse)
async def get_graph(
    scope: GraphScope = Query(default=GraphScope.ALL),
    limit: int = Query(default=200, ge=1, le=1000),
    tag: str | None = Query(default=None, description="Filter when scope=tag"),
    include_similarity: bool = Query(default=False),
    similarity_threshold: float = Query(default=0.75, ge=0.0, le=1.0),
    min_tag_overlap: int = Query(default=1, ge=1),
) -> GraphResponse:
    """Return the knowledge graph as JSON for the frontend visualisation."""
    try:
        return await build_graph(
            scope=scope,
            limit=limit,
            tag=tag,
            include_similarity=include_similarity,
            similarity_threshold=similarity_threshold,
            min_tag_overlap=min_tag_overlap,
        )
    except Exception as exc:
        logger.exception("Graph build failed")
        raise HTTPException(status_code=500, detail=str(exc))
