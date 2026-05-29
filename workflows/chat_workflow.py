"""LangGraph chat workflow with retrieval augmented generation.

Optional RAG enhancements (toggled via ChatState flags):
- rewrite_query:    expand user question into multiple sub-queries (better recall)
- compress_history: summarise old turns when conversation grows (avoid overflow)
"""
from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from core.llm import get_chat_llm
from models.schemas import ChatMessage, ChatResponse, SearchResultItem
from tools.rag_pipeline import compress_history, retrieve_with_rewrite
from tools.search import hybrid_search
from utils.logger import logger

SYSTEM_PROMPT = """你是用户的个人知识管理助理 (PKM Agent)。
请遵循：
1. 严格基于<context>中检索到的资料给出回答；当资料不足时直接说明并提出问题。
2. 用中文回答，简洁、有条理，必要时使用 Markdown。
3. 在答案末尾用 "📚 参考资料" 列出引用 (使用 [n] 数字编号 + 标题)。
4. 不杜撰资料；若引用，请确保来源出现在 context 中。
"""


class ChatState(TypedDict, total=False):
    message: str
    history: list[ChatMessage]
    use_memory: bool
    top_k: int
    hits: list[SearchResultItem]
    answer: str
    # RAG optimisation
    rewrite_query: bool
    compress_history: bool
    history_token_budget: int
    max_subqueries: int
    subqueries: list[str]
    compressed_history_summary: str | None


async def node_compress(state: ChatState) -> ChatState:
    """Optionally compress overlong history into a summary + recent tail."""
    if not state.get("compress_history"):
        return state
    history = state.get("history", []) or []
    new_history, summary = await compress_history(
        history,
        token_budget=state.get("history_token_budget", 1200),
    )
    return {
        **state,
        "history": new_history,
        "compressed_history_summary": summary,
    }


async def node_retrieve(state: ChatState) -> ChatState:
    if not state.get("use_memory", True):
        return {**state, "hits": [], "subqueries": []}
    if state.get("rewrite_query"):
        hits, subs = await retrieve_with_rewrite(
            state["message"],
            history=state.get("history", []),
            k=state.get("top_k", 6),
            max_subqueries=state.get("max_subqueries", 3),
        )
        return {**state, "hits": hits, "subqueries": subs}
    hits = await hybrid_search(state["message"], k=state.get("top_k", 6))
    return {**state, "hits": hits, "subqueries": [state["message"]]}


async def node_answer(state: ChatState) -> ChatState:
    history = state.get("history", [])
    hits = state.get("hits", [])
    context_block = "\n\n".join(
        f"[{i+1}] {h.title}\n来源: {h.source}\n内容: {h.snippet}"
        for i, h in enumerate(hits)
    ) or "（暂无检索结果）"

    msgs: list = [
        SystemMessage(content=SYSTEM_PROMPT),
        SystemMessage(content=f"<context>\n{context_block}\n</context>"),
    ]
    for h in history[-12:]:
        if h.role == "user":
            msgs.append(HumanMessage(content=h.content))
        elif h.role == "assistant":
            msgs.append(AIMessage(content=h.content))
        elif h.role == "system":
            msgs.append(SystemMessage(content=h.content))
    msgs.append(HumanMessage(content=state["message"]))

    llm = get_chat_llm()
    res = await llm.ainvoke(msgs)
    answer = res.content if hasattr(res, "content") else str(res)
    logger.debug(f"Chat answer length={len(answer)}")
    return {**state, "answer": answer}


def build_chat_graph():
    g = StateGraph(ChatState)
    g.add_node("compress_node", node_compress)
    g.add_node("retrieve", node_retrieve)
    g.add_node("answer_node", node_answer)
    g.set_entry_point("compress_node")
    g.add_edge("compress_node", "retrieve")
    g.add_edge("retrieve", "answer_node")
    g.add_edge("answer_node", END)
    return g.compile()


_CHAT_GRAPH = None


def get_chat_graph():
    global _CHAT_GRAPH
    if _CHAT_GRAPH is None:
        _CHAT_GRAPH = build_chat_graph()
    return _CHAT_GRAPH


async def run_chat(
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
    state: ChatState = {
        "message": message,
        "history": history or [],
        "use_memory": use_memory,
        "top_k": top_k,
        "rewrite_query": rewrite_query,
        "compress_history": compress_history,
        "history_token_budget": history_token_budget,
        "max_subqueries": max_subqueries,
    }
    result = await get_chat_graph().ainvoke(state)
    new_history = (history or []) + [
        ChatMessage(role="user", content=message),
        ChatMessage(role="assistant", content=result["answer"]),
    ]
    return ChatResponse(
        answer=result["answer"],
        sources=result.get("hits", []),
        history=new_history,
        subqueries=result.get("subqueries", []),
        compressed_history_summary=result.get("compressed_history_summary"),
    )
