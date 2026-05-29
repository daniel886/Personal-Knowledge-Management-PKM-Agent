"""FastAPI review + scheduler routes."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from agents import get_agent
from core.scheduler import get_scheduler
from models.schemas import ReviewRequest, ReviewResponse, TaskInfo
from utils.logger import logger

router = APIRouter(prefix="/api", tags=["review"])


@router.post("/review", response_model=ReviewResponse)
async def review(req: ReviewRequest) -> ReviewResponse:
    try:
        return await get_agent().review(req.period, start=req.start, end=req.end)
    except Exception as exc:
        logger.exception("Review failed")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/tasks", response_model=list[TaskInfo])
async def list_tasks() -> list[TaskInfo]:
    return [TaskInfo(**j) for j in get_scheduler().list_jobs()]
