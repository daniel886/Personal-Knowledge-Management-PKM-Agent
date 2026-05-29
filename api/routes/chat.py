"""FastAPI chat route."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents import get_agent
from models.schemas import ChatRequest, ChatResponse
from utils.logger import logger

router = APIRouter(prefix="/api/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    try:
        return await get_agent().chat(
            req.message,
            history=req.history,
            use_memory=req.use_memory,
            top_k=req.top_k,
        )
    except Exception as exc:
        logger.exception("Chat failed")
        raise HTTPException(status_code=500, detail=str(exc))
