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


class ChatResponse(BaseModel):
    answer: str
    sources: list[SearchResultItem] = Field(default_factory=list)
    history: list[ChatMessage] = Field(default_factory=list)


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
