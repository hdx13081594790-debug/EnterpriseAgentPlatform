"""HTTP 边界的请求/响应模型。

数据流：客户端 JSON → Pydantic 校验/规范化 → 路由 → Service；
Service 字典 → response_model 过滤与序列化 → 客户端 JSON。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """聊天入口；未传 session_id 时由 API 自动生成。"""

    message: str = Field(min_length=1, max_length=5000)
    session_id: str | None = Field(default=None, min_length=3, max_length=64)


class SourceItem(BaseModel):
    """标准来源结构，保留文档、分数和预览以便追溯。"""

    index: int
    title: str
    source: str
    score: float
    document_id: str | None = None
    preview: str | None = None


class ChatResponse(BaseModel):
    """统一返回路由、Agent、回答、来源、质量和缓存信息。"""

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
    """知识导入请求；metadata 用于未来的租户/类别过滤。"""

    title: str = Field(min_length=1, max_length=255)
    source: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=20, max_length=500_000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ResearchRequest(BaseModel):
    """同步研究任务请求。"""

    topic: str = Field(min_length=3, max_length=500)
    session_id: str | None = Field(default=None, min_length=3, max_length=64)


class ResearchResponse(BaseModel):
    """研究图结束后的报告、来源与评审信息。"""

    task_id: str
    status: str
    topic: str
    report: str
    sources: list[dict[str, Any]]
    quality_score: float
    revision_count: int
