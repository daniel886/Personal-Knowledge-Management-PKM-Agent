"""LangGraph review workflow producing weekly / monthly knowledge reviews."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, StateGraph
from sqlalchemy import select

from core.config import settings
from core.llm import get_chat_llm
from core.obsidian import get_vault
from models.database import Knowledge, ReviewReport, session_scope
from models.schemas import ReviewResponse
from utils.logger import logger

REVIEW_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是用户的「学习教练 + 知识管理顾问」。请基于本周期内整理的知识条目，"
            "生成一份高质量的回顾报告，使用中文 Markdown，结构如下：\n"
            "1. **本期亮点**：3-5 条最重要的洞察。\n"
            "2. **主题聚类**：将相关条目分组并解释关联。\n"
            "3. **延伸阅读**：根据空白点推荐 3 个值得探索的方向。\n"
            "4. **行动计划**：给出本期可立即落实的 3 项行动。\n"
            "保持简洁、可执行，不复述全部条目。",
        ),
        (
            "human",
            "周期: {period}\n范围: {start} ~ {end}\n条目数: {count}\n\n条目摘要：\n{items}",
        ),
    ]
)


class ReviewState(TypedDict, total=False):
    period: Literal["weekly", "monthly", "custom"]
    start: datetime
    end: datetime
    items: list[Knowledge]
    summary: str
    obsidian_path: str


async def node_collect(state: ReviewState) -> ReviewState:
    async with session_scope() as session:
        rows = await session.execute(
            select(Knowledge).where(
                Knowledge.created_at >= state["start"],
                Knowledge.created_at <= state["end"],
            )
        )
        items = list(rows.scalars())
    logger.info(f"Review collected {len(items)} items")
    return {**state, "items": items}


async def node_summarise(state: ReviewState) -> ReviewState:
    items = state.get("items", [])
    if not items:
        return {**state, "summary": "_本期暂无新增知识条目。_"}
    bullet = "\n".join(
        f"- [{i.source_type}] **{i.title}** | tags: {i.tags or '-'} | "
        f"摘要: {(i.summary or '')[:240]}"
        for i in items
    )
    chain = REVIEW_PROMPT | get_chat_llm()
    res = await chain.ainvoke(
        {
            "period": state["period"],
            "start": state["start"].strftime("%Y-%m-%d"),
            "end": state["end"].strftime("%Y-%m-%d"),
            "count": len(items),
            "items": bullet,
        }
    )
    return {**state, "summary": res.content if hasattr(res, "content") else str(res)}


async def node_persist(state: ReviewState) -> ReviewState:
    summary = state["summary"]
    period = state["period"]
    start = state["start"]
    end = state["end"]
    title = f"{period.capitalize()} Review {start:%Y-%m-%d} ~ {end:%Y-%m-%d}"
    note_path = get_vault().write_note(
        title=title,
        content=summary,
        folder=settings.obsidian_review_folder,
        frontmatter={
            "type": "review",
            "period": period,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "count": len(state.get("items", [])),
        },
    )
    async with session_scope() as session:
        session.add(
            ReviewReport(
                period=period,
                start_at=start,
                end_at=end,
                summary=summary,
                obsidian_path=str(note_path),
            )
        )
    return {**state, "obsidian_path": str(note_path)}


def build_review_graph():
    g = StateGraph(ReviewState)
    g.add_node("collect", node_collect)
    g.add_node("summarise", node_summarise)
    g.add_node("persist", node_persist)
    g.set_entry_point("collect")
    g.add_edge("collect", "summarise")
    g.add_edge("summarise", "persist")
    g.add_edge("persist", END)
    return g.compile()


_REVIEW_GRAPH = None


def get_review_graph():
    global _REVIEW_GRAPH
    if _REVIEW_GRAPH is None:
        _REVIEW_GRAPH = build_review_graph()
    return _REVIEW_GRAPH


async def run_review(
    period: str = "weekly",
    *,
    start: datetime | None = None,
    end: datetime | None = None,
) -> ReviewResponse:
    end = end or datetime.utcnow()
    if start is None:
        if period == "weekly":
            start = end - timedelta(days=7)
        elif period == "monthly":
            start = end - timedelta(days=30)
        else:
            start = end - timedelta(days=7)

    state: ReviewState = {"period": period, "start": start, "end": end}
    result = await get_review_graph().ainvoke(state)
    return ReviewResponse(
        period=period,
        start=start,
        end=end,
        summary=result["summary"],
        obsidian_path=result.get("obsidian_path"),
        knowledge_count=len(result.get("items", [])),
    )
