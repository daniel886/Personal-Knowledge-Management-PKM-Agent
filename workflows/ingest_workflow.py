"""LangGraph ingest workflow.

Pipeline: scrape -> summarise -> tag -> link -> mindmap -> persist (DB + Vector + Obsidian).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, TypedDict
from uuid import uuid4

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

from core.obsidian import get_vault
from core.vector_store import get_vector_store
from models.database import Knowledge, KnowledgeChunk, session_scope
from models.schemas import IngestResult, SourceType
from scrapers import get_scraper
from scrapers.base import ScrapedDocument
from tools.linker import suggest_links
from tools.mindmap import generate_mindmap
from tools.summarizer import summarise
from tools.tagger import generate_tags
from utils.logger import logger


class IngestState(TypedDict, total=False):
    knowledge_id: str
    source_type: str
    target: str
    document: ScrapedDocument
    summary: str
    tags: list[str]
    related: list[str]
    mindmap: str
    obsidian_path: str
    chunks_indexed: int


# ---------- node functions ----------


async def node_scrape(state: IngestState) -> IngestState:
    if "document" in state and state["document"]:
        return state
    scraper = get_scraper(state["source_type"])
    doc = await scraper.scrape(state["target"])
    return {**state, "document": doc}


async def node_summarise(state: IngestState) -> IngestState:
    doc = state["document"]
    summary = await summarise(
        title=doc.title,
        content=doc.content,
        source=doc.source,
        source_type=doc.source_type,
    )
    return {**state, "summary": summary}


async def node_tag(state: IngestState) -> IngestState:
    doc = state["document"]
    tags = await generate_tags(title=doc.title, content=doc.content)
    return {**state, "tags": tags}


async def node_link(state: IngestState) -> IngestState:
    doc = state["document"]
    related = await suggest_links(
        title=doc.title,
        content=doc.content,
        exclude_id=state.get("knowledge_id"),
    )
    return {**state, "related": related}


async def node_mindmap(state: IngestState) -> IngestState:
    doc = state["document"]
    try:
        mm = await generate_mindmap(title=doc.title, content=state.get("summary") or doc.content)
    except Exception as exc:
        logger.warning(f"Mindmap generation failed: {exc}")
        mm = ""
    return {**state, "mindmap": mm}
async def node_persist(state: IngestState) -> IngestState:
    doc = state["document"]
    summary = state.get("summary", "")
    tags = state.get("tags", [])
    related = state.get("related", [])
    mindmap = state.get("mindmap", "")
    knowledge_id = state.get("knowledge_id") or str(uuid4())

    # 1) Vector indexing
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
    chunks = splitter.split_text(doc.content) if doc.content else []
    metadata_base = {
        "knowledge_id": knowledge_id,
        "title": doc.title,
        "source": doc.source,
        "source_type": doc.source_type,
        "tags": ",".join(tags),
    }
    docs = [
        Document(page_content=chunk, metadata={**metadata_base, "chunk": i})
        for i, chunk in enumerate(chunks)
    ]
    chunk_ids = await get_vector_store().add_documents(docs) if docs else []

    # 2) Obsidian note
    body_parts = [
        f"> 来源: [{doc.source_type}]({doc.source})\n",
        "## ✨ 智能摘要\n\n" + (summary or "_(empty)_"),
    ]
    if mindmap:
        body_parts.append("## 🧠 思维导图\n\n" + mindmap)
    if related:
        body_parts.append(
            "## 🔗 相关笔记\n\n" + "\n".join(f"- [[{r}]]" for r in related)
        )
    if doc.content:
        snippet = doc.content[:4000]
        body_parts.append("## 📝 原始正文（节选）\n\n" + snippet)

    obsidian_path = get_vault().write_note(
        title=doc.title,
        content="\n\n".join(body_parts),
        frontmatter={
            "id": knowledge_id,
            "type": doc.source_type,
            "source": doc.source,
            "tags": tags,
            "related": related,
            "ingested_at": datetime.utcnow().isoformat(),
        },
    )

    # 3) SQLite metadata
    async with session_scope() as session:
        k = Knowledge(
            id=knowledge_id,
            title=doc.title,
            source_type=doc.source_type,
            source=doc.source,
            summary=summary,
            tags=",".join(tags),
            obsidian_path=str(obsidian_path),
            extra=doc.metadata,
            chunks_indexed=len(chunk_ids),
        )
        session.add(k)
        for i, content in enumerate(chunks):
            session.add(
                KnowledgeChunk(
                    id=f"{knowledge_id}:{i}",
                    knowledge_id=knowledge_id,
                    chunk_index=i,
                    content=content,
                )
            )

    logger.info(
        f"Ingested '{doc.title}' id={knowledge_id} chunks={len(chunk_ids)} -> {obsidian_path}"
    )
    return {
        **state,
        "knowledge_id": knowledge_id,
        "chunks_indexed": len(chunk_ids),
        "obsidian_path": str(obsidian_path),
    }


# ---------- graph build ----------


def build_ingest_graph():
    graph = StateGraph(IngestState)
    graph.add_node("scrape", node_scrape)
    graph.add_node("summarize", node_summarise)
    graph.add_node("tag_node", node_tag)
    graph.add_node("link_node", node_link)
    graph.add_node("mindmap_node", node_mindmap)
    graph.add_node("persist", node_persist)

    graph.set_entry_point("scrape")
    graph.add_edge("scrape", "summarize")
    graph.add_edge("summarize", "tag_node")
    graph.add_edge("tag_node", "link_node")
    graph.add_edge("link_node", "mindmap_node")
    graph.add_edge("mindmap_node", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


_INGEST_GRAPH = None


def get_ingest_graph():
    global _INGEST_GRAPH
    if _INGEST_GRAPH is None:
        _INGEST_GRAPH = build_ingest_graph()
    return _INGEST_GRAPH


# ---------- public API ----------


async def ingest_url(source_type: SourceType, target: str, **kwargs: Any) -> IngestResult:
    """Run the ingest workflow for a URL/path."""
    graph = get_ingest_graph()
    state: IngestState = {"source_type": source_type.value, "target": target}
    result = await graph.ainvoke(state)
    doc: ScrapedDocument = result["document"]
    return IngestResult(
        knowledge_id=result["knowledge_id"],
        title=doc.title,
        summary=result.get("summary", ""),
        tags=result.get("tags", []),
        links=result.get("related", []),
        obsidian_path=result.get("obsidian_path"),
        chunks_indexed=result.get("chunks_indexed", 0),
        source_type=SourceType(doc.source_type),
        source=doc.source,
    )


async def ingest_document(doc: ScrapedDocument) -> IngestResult:
    """Run the ingest workflow given an already-scraped document."""
    graph = get_ingest_graph()
    state: IngestState = {
        "source_type": doc.source_type,
        "target": doc.source,
        "document": doc,
    }
    result = await graph.ainvoke(state)
    return IngestResult(
        knowledge_id=result["knowledge_id"],
        title=doc.title,
        summary=result.get("summary", ""),
        tags=result.get("tags", []),
        links=result.get("related", []),
        obsidian_path=result.get("obsidian_path"),
        chunks_indexed=result.get("chunks_indexed", 0),
        source_type=SourceType(doc.source_type),
        source=doc.source,
    )
