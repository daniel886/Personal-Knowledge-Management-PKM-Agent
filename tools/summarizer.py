"""LLM-driven summarisation tool."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_chat_llm
from utils.logger import logger

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一个资深的个人知识管理助理。请基于用户提供的内容，以中文产出 4 部分输出：\n"
            "1) **TL;DR**：3 句话以内的精炼总结。\n"
            "2) **关键知识点**：4-8 条 Markdown 列表，覆盖核心概念、数据、洞察。\n"
            "3) **延伸思考**：2-3 条值得深入研究的问题。\n"
            "4) **行动项**：可立即执行的 1-3 条建议。\n"
            "全部输出使用 Markdown，不要带前后注释，保持精炼。",
        ),
        (
            "human",
            "标题：{title}\n来源类型：{source_type}\n来源：{source}\n\n内容：\n{content}",
        ),
    ]
)


async def summarise(
    *,
    title: str,
    content: str,
    source: str,
    source_type: str,
    max_chars: int = 12000,
) -> str:
    """Produce a Markdown summary suitable for an Obsidian note."""
    if not content.strip():
        return "_(no content provided)_"
    truncated = content[:max_chars]
    if len(content) > max_chars:
        logger.info(f"Content truncated for summary: {len(content)} -> {max_chars}")
    llm = get_chat_llm()
    chain = SUMMARY_PROMPT | llm
    res = await chain.ainvoke(
        {
            "title": title,
            "source_type": source_type,
            "source": source,
            "content": truncated,
        }
    )
    return res.content if hasattr(res, "content") else str(res)
