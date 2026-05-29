"""RAG optimisation pipeline.

Two pluggable enhancements:

1. Query rewriting — expands a single user question into 2-3 retrieval-friendly
   sub-queries via an LLM. Improves recall on under-specified or multi-hop queries.

2. Context compression — when chat history exceeds a token budget, summarises
   the oldest messages into a single moving "system: previously…" message and
   keeps only the most recent messages verbatim. Avoids context-window overflow
   in long conversations.

Both functions are pure-async and fail gracefully: any LLM error returns the
original input unchanged so downstream retrieval/answering keep working.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_chat_llm
from models.schemas import ChatMessage, SearchResultItem
from tools.search import hybrid_search
from utils.logger import logger

# ---------------- Prompts ----------------

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名检索专家。基于用户的问题和最近对话历史，"
            "生成 {max_n} 个互补的中文检索查询（短句、关键词驱动），"
            "覆盖问题的不同侧面或潜在的同义表达。"
            '严格输出 JSON 数组，例如: ["查询A", "查询B", "查询C"]。'
            "不要解释，不要多余文字。",
        ),
        (
            "human",
            "最近对话:\n{history}\n\n当前问题:\n{question}",
        ),
    ]
)

COMPRESS_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是对话摘要助手。请将以下早期对话压缩为不超过 4 句中文要点，"
            "保留关键事实、用户偏好与未解决的问题。不要使用列表符号。",
        ),
        (
            "human",
            "{transcript}",
        ),
    ]
)


# ---------------- Helpers ----------------


def _approx_tokens(text: str) -> int:
    """Cheap CJK-aware token estimator: 1 token ≈ 1 CJK char or 4 ASCII chars."""
    if not text:
        return 0
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other = len(text) - cjk
    return cjk + max(1, other // 4)


def _history_tokens(history: Iterable[ChatMessage]) -> int:
    return sum(_approx_tokens(m.content) for m in history)


def _format_history(history: Iterable[ChatMessage], max_msgs: int = 6) -> str:
    items = list(history)[-max_msgs:]
    return "\n".join(f"{m.role}: {m.content}" for m in items) or "(empty)"


def _parse_subqueries(text: str, fallback: str, max_n: int) -> list[str]:
    """Parse the LLM JSON array; fall back to the original query on any error."""
    text = (text or "").strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        arr = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if m:
            try:
                arr = json.loads("[" + m.group(1) + "]")
            except json.JSONDecodeError:
                arr = []
        else:
            arr = []
    if not isinstance(arr, list):
        arr = []
    cleaned: list[str] = []
    seen: set[str] = set()
    for x in arr:
        s = str(x).strip().strip("\"'`")
        if not s or len(s) > 200:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(s)
        if len(cleaned) >= max_n:
            break
    if fallback and fallback.lower() not in seen:
        cleaned.insert(0, fallback)
        cleaned = cleaned[:max_n]
    return cleaned or [fallback]


# ---------------- Query rewriting ----------------


async def rewrite_query(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    max_n: int = 3,
) -> list[str]:
    """Return 1..max_n retrieval-friendly sub-queries (original always included).

    On any LLM failure this returns ``[question]`` so callers can always proceed.
    """
    if max_n <= 1:
        return [question]
    try:
        llm = get_chat_llm(temperature=0.0)
        chain = REWRITE_PROMPT | llm
        res = await chain.ainvoke(
            {
                "max_n": max_n,
                "question": question,
                "history": _format_history(history or []),
            }
        )
        text = res.content if hasattr(res, "content") else str(res)
        subs = _parse_subqueries(text, fallback=question, max_n=max_n)
        logger.info(f"Rewrote query into {len(subs)} sub-queries")
        return subs
    except Exception as exc:
        logger.warning(f"Query rewrite failed, falling back: {exc}")
        return [question]


# ---------------- Multi-query retrieval ----------------


async def retrieve_with_rewrite(
    question: str,
    *,
    history: list[ChatMessage] | None = None,
    k: int = 6,
    max_subqueries: int = 3,
    source_type: str | None = None,
) -> tuple[list[SearchResultItem], list[str]]:
    """Run hybrid_search for each rewritten sub-query and merge/dedupe results.

    Returns ``(merged_hits, subqueries)``. Order preserves the original query's
    hits first, then subsequent sub-queries' new hits.
    """
    subs = await rewrite_query(question, history=history, max_n=max_subqueries)
    seen_keys: set[str] = set()
    merged: list[SearchResultItem] = []
    for sq in subs:
        try:
            hits = await hybrid_search(sq, k=k, source_type=source_type)
        except Exception as exc:
            logger.debug(f"Sub-query '{sq}' failed: {exc}")
            continue
        for h in hits:
            key = h.id or h.title
            if key in seen_keys:
                continue
            seen_keys.add(key)
            merged.append(h)
            if len(merged) >= k:
                break
        if len(merged) >= k:
            break
    return merged, subs


# ---------------- Context compression ----------------


async def compress_history(
    history: list[ChatMessage],
    *,
    token_budget: int = 1200,
    keep_recent: int = 4,
) -> tuple[list[ChatMessage], str | None]:
    """Compress an overlong chat history.

    Strategy:
    - If total tokens <= budget: return history unchanged.
    - Else: keep the last ``keep_recent`` messages verbatim, summarise the rest
      with the LLM into a single SystemMessage prefixed with "previously: ".

    Returns ``(new_history, summary_or_None)``. On any LLM failure the
    *truncated* history is returned with summary=None so the conversation
    never breaks.
    """
    if not history:
        return [], None
    total = _history_tokens(history)
    if total <= token_budget:
        return list(history), None
    if len(history) <= keep_recent:
        return list(history), None
    head = history[:-keep_recent]
    tail = history[-keep_recent:]
    transcript = "\n".join(f"{m.role}: {m.content}" for m in head)
    summary: str | None = None
    try:
        llm = get_chat_llm(temperature=0.0)
        chain = COMPRESS_PROMPT | llm
        res = await chain.ainvoke({"transcript": transcript})
        summary = (res.content if hasattr(res, "content") else str(res)).strip()
    except Exception as exc:
        logger.warning(f"History compression failed, using truncation only: {exc}")
        summary = None

    new_history: list[ChatMessage] = []
    if summary:
        new_history.append(
            ChatMessage(role="system", content=f"previously: {summary}")
        )
    new_history.extend(tail)
    logger.info(
        f"Compressed history {len(history)}→{len(new_history)} msgs "
        f"(approx {total}→{_history_tokens(new_history)} tokens)"
    )
    return new_history, summary
