"""Conversational chat agent (CLI / API entrypoint)."""
from __future__ import annotations

from models.schemas import ChatMessage, ChatResponse
from workflows.chat_workflow import run_chat


class ChatAgent:
    """Stateful chat session keeping its own short-term memory."""

    def __init__(self, max_history: int = 30) -> None:
        self._history: list[ChatMessage] = []
        self._max_history = max_history

    async def ask(
        self,
        message: str,
        *,
        use_memory: bool = True,
        top_k: int = 6,
        rewrite_query: bool = False,
        compress_history: bool = False,
        history_token_budget: int = 1200,
        max_subqueries: int = 3,
    ) -> ChatResponse:
        resp = await run_chat(
            message,
            history=self._history,
            use_memory=use_memory,
            top_k=top_k,
            rewrite_query=rewrite_query,
            compress_history=compress_history,
            history_token_budget=history_token_budget,
            max_subqueries=max_subqueries,
        )
        self._history = resp.history[-self._max_history:]
        return resp

    @property
    def history(self) -> list[ChatMessage]:
        return list(self._history)

    def reset(self) -> None:
        self._history.clear()
