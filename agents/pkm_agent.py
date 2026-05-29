"""High-level PKM agent orchestrating all workflows."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from models.schemas import (
    ChatMessage,
    ChatResponse,
    IngestResult,
    ReviewResponse,
    SourceType,
)
from utils.logger import logger
from workflows.chat_workflow import run_chat
from workflows.ingest_workflow import ingest_url
from workflows.review_workflow import run_review


class PKMAgent:
    """Orchestrates ingest / chat / review on top of the LangGraph workflows."""

    async def ingest(
        self,
        source_type: SourceType | str,
        target: str,
        **kwargs: Any,
    ) -> IngestResult:
        if isinstance(source_type, str):
            source_type = SourceType(source_type)
        logger.info(f"PKMAgent.ingest source_type={source_type} target={target}")
        return await ingest_url(source_type, target, **kwargs)

    async def chat(
        self,
        message: str,
        history: list[ChatMessage] | None = None,
        *,
        use_memory: bool = True,
        top_k: int = 6,
        rewrite_query: bool = False,
        compress_history: bool = False,
        history_token_budget: int = 1200,
        max_subqueries: int = 3,
    ) -> ChatResponse:
        return await run_chat(
            message,
            history,
            use_memory=use_memory,
            top_k=top_k,
            rewrite_query=rewrite_query,
            compress_history=compress_history,
            history_token_budget=history_token_budget,
            max_subqueries=max_subqueries,
        )

    async def review(
        self,
        period: str = "weekly",
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> ReviewResponse:
        return await run_review(period=period, start=start, end=end)


_AGENT: PKMAgent | None = None


def get_agent() -> PKMAgent:
    global _AGENT
    if _AGENT is None:
        _AGENT = PKMAgent()
    return _AGENT
