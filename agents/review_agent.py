"""Review agent providing scheduler-friendly entry points."""
from __future__ import annotations

from models.schemas import ReviewResponse
from utils.logger import logger
from workflows.review_workflow import run_review


async def run_weekly_review() -> ReviewResponse:
    logger.info("⏰ Triggering weekly review")
    return await run_review("weekly")


async def run_monthly_review() -> ReviewResponse:
    logger.info("⏰ Triggering monthly review")
    return await run_review("monthly")
