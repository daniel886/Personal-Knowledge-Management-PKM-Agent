"""FastAPI search route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models.schemas import SearchRequest, SearchResponse
from tools.search import hybrid_search
from utils.logger import logger

router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("", response_model=SearchResponse)
async def search(req: SearchRequest) -> SearchResponse:
    try:
        hits = await hybrid_search(
            req.query,
            k=req.k,
            source_type=req.source_type.value if req.source_type else None,
        )
        return SearchResponse(query=req.query, hits=hits)
    except Exception as exc:
        logger.exception("Search failed")
        raise HTTPException(status_code=500, detail=str(exc))
