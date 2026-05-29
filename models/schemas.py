"""Pydantic v2 schemas used across the API and agents."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, HttpUrl


class SourceType(str, Enum):
    WEB = "web"
    PDF = "pdf"
    YOUTUBE = "youtube"
    WECHAT = "wechat"
    EMAIL = "email"
    NOTION = "notion"
    RSS = "rss"
    UPLOAD = "upload"
    MARKDOWN = "markdown"


class IngestRequest(BaseModel):
    source_type: SourceType
    url: HttpUrl | None = None
    file_path: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)
    auto_save_obsidian: bool = True


class IngestResult(BaseModel):
    knowledge_id: str
    title: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    links: list[str] = Field(default_factory=list)
    obsidian_path: str | None = None
    chunks_indexed: int = 0
    source_type: SourceType
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    source_type: SourceType | None = None
    tags: list[str] | None = None


class SearchResultItem(BaseModel):
    id: str
    title: str
    snippet: str
    score: float
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchResultItem]


class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = Field(default_factory=list)
    use_memory: bool = True
    top_k: int = 6
    # RAG optimisation toggles
    rewrite_query: bool = False
    compress_history: bool = False
    history_token_budget: int = Field(default=1200, ge=200, le=8000)
    max_subqueries: int = Field(default=3, ge=1, le=5)


class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResultItem] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list)
    # RAG diagnostics (optional)
    subqueries: list[str] = Field(default_factory=list)
    compressed_history_summary: str | None = None


class ReviewRequest(BaseModel):
    period: Literal["weekly", "monthly", "custom"] = "weekly"
    start: datetime | None = None
    end: datetime | None = None


class ReviewResponse(BaseModel):
    period: str
    start: datetime
    end: datetime
    summary: str
    obsidian_path: str | None = None
    knowledge_count: int


class TaskInfo(BaseModel):
    id: str
    next_run: datetime | None
    trigger: str


# ---------------- Knowledge Graph ----------------


class GraphNode(BaseModel):
    """A single node in the knowledge graph."""

    id: str
    title: str
    source_type: str
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None
    # Visual hints for the frontend
    weight: int = 1  # node size proxy (number of connections)
    group: str | None = None  # optional cluster label (defaults to source_type)


class GraphEdge(BaseModel):
    """A weighted edge between two graph nodes."""

    source: str
    target: str
    weight: float = 1.0
    edge_type: Literal["tag", "wikilink", "similarity"] = "tag"
    label: str | None = None  # e.g. the shared tag name


class GraphScope(str, Enum):
    ALL = "all"
    RECENT = "recent"
    TAG = "tag"


class GraphRequest(BaseModel):
    scope: GraphScope = GraphScope.ALL
    limit: int = Field(default=200, ge=1, le=1000)
    tag: str | None = None  # only used when scope == tag
    include_similarity: bool = False
    similarity_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    min_tag_overlap: int = Field(default=1, ge=1)


class GraphResponse(BaseModel):
    scope: str
    nodes: list[GraphNode]
    edges: list[GraphEdge]
    stats: dict[str, Any] = Field(default_factory=dict)
