from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=5000)
    session_id: str | None = Field(default=None, min_length=3, max_length=64)


class SourceItem(BaseModel):
    index: int
    title: str
    source: str
    score: float
    document_id: str | None = None
    preview: str | None = None


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    intent: str
    agent: str
    answer: str
    sources: list[dict[str, Any]]
    route_confidence: float
    quality_score: float
    needs_human: bool
    cache_hit: bool


class DocumentCreate(BaseModel):
    title: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=20, max_length=500_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchRequest(BaseModel):
    topic: str = Field(min_length=3, max_length=500)
    session_id: str | None = Field(default=None, min_length=3, max_length=64)


class ResearchResponse(BaseModel):
    task_id: str
    status: str
    topic: str
    report: str
    sources: list[dict[str, Any]]
    quality_score: float
    revision_count: int

