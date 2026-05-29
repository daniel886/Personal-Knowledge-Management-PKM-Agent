"""Generate Markdown / Mermaid mind-maps from a piece of content."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from core.llm import get_chat_llm

MINDMAP_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是结构化思维专家。请将给定内容整理为 Mermaid mindmap 格式，"
            "根节点使用标题，下分 3-5 个一级节点，每个一级节点 2-4 个子节点。"
            "严格输出可直接渲染的 Mermaid 代码块（包含 ```mermaid 包裹）。",
        ),
        ("human", "标题：{title}\n\n内容：\n{content}"),
    ]
)


async def generate_mindmap(*, title: str, content: str, max_chars: int = 6000) -> str:
    """Return a Mermaid-formatted mind-map string."""
    llm = get_chat_llm(temperature=0.3)
    chain = MINDMAP_PROMPT | llm
    res = await chain.ainvoke({"title": title, "content": content[:max_chars]})
    text = res.content if hasattr(res, "content") else str(res)
    text = text.strip()
    if not text.startswith("```mermaid"):
        text = f"```mermaid\nmindmap\n  root(({title}))\n```" if "mindmap" not in text else f"```mermaid\n{text}\n```"
    return text
