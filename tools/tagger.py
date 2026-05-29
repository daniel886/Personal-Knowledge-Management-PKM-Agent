"""LLM-based tag generation."""
from __future__ import annotations

import json
import re

from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_chat_llm
from utils.logger import logger

TAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是一名信息架构师。请基于内容提取 5-10 个高信息密度的中英文标签，"
            "输出严格 JSON 数组，无其他文字。例如: [\"机器学习\", \"transformer\", \"论文\"]。"
            "标签遵循: 简短 (≤6 字)、不重复、不要 # 号、可使用驼峰或下划线。",
        ),
        ("human", "标题：{title}\n\n内容：\n{content}"),
    ]
)


async def generate_tags(*, title: str, content: str, max_chars: int = 8000) -> list[str]:
    llm = get_chat_llm(temperature=0.0)
    chain = TAG_PROMPT | llm
    res = await chain.ainvoke({"title": title, "content": content[:max_chars]})
    text = res.content if hasattr(res, "content") else str(res)
    return _parse(text)


def _parse(text: str) -> list[str]:
    text = text.strip()
    # strip ```json ... ``` fences if present
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    raw: list[str] = []
    # try direct JSON
    try:
        arr = json.loads(text)
        if isinstance(arr, list):
            raw = [str(t) for t in arr if t]
    except json.JSONDecodeError:
        pass
    # fallback: extract bracketed JSON
    if not raw:
        m = re.search(r"\[(.*?)\]", text, re.DOTALL)
        if m:
            try:
                arr = json.loads("[" + m.group(1) + "]")
                raw = [str(t) for t in arr if t]
            except json.JSONDecodeError:
                pass
    # final fallback: split by separators
    if not raw:
        logger.warning(f"Tag parsing fallback for: {text[:120]}")
        raw = re.split(r"[,\n、;|]", text)

    cleaned: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = str(t).strip().strip("\"'`").lstrip("#").strip()
        if not t or len(t) > 24:
            continue
        key = t.lower()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(t)
        if len(cleaned) >= 10:
            break
    return cleaned
