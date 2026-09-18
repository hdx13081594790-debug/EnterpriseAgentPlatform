"""REST API 路由层。

数据流：HTTP/Pydantic → ApplicationContainer 中的工作流/服务 → Repository/Cache →
响应模型。路由只负责协议转换和状态码，不承载 Agent 业务逻辑。
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    DocumentCreate,
    ResearchRequest,
    ResearchResponse,
)
from app.db.repositories import ConversationRepository
from app.services.container import ApplicationContainer

router = APIRouter()


def container_from(request: Request) -> ApplicationContainer:
    """从 FastAPI 生命周期状态获取唯一的应用容器。"""

    return request.app.state.container


@router.get("/health", tags=["system"])
def health(request: Request) -> dict[str, Any]:
    """聚合数据库、缓存、模型模式和知识库规模。"""

    container = container_from(request)
    # 数据流：health endpoint → DB SELECT 1 / Redis PING → 聚合状态 JSON。
    database_ok = container.database.health()
    cache_ok = container.cache.health()
    return {
        "status": "ok" if database_ok else "degraded",
        "database": "up" if database_ok else "down",
        "cache": {
            "backend": container.cache.backend_name,
            "status": "up" if cache_ok else "down",
        },
        "llm_provider": container.settings.llm_provider,
        "knowledge": container.rag.stats() if database_ok else {},
    }


@router.post("/api/v1/chat", response_model=ChatResponse, tags=["agents"])
def chat(payload: ChatRequest, request: Request) -> dict[str, Any]:
    """把一条用户消息交给客服 Supervisor 图。"""

    container = container_from(request)
    # session_id 串起多轮历史；request_id 串起本次调用的所有 AgentTrace。
    session_id = payload.session_id or f"session-{uuid.uuid4().hex[:12]}"
    request_id = f"req-{uuid.uuid4().hex}"
    # 数据流：ChatRequest → CustomerServiceWorkflow → ChatResponse。
    return container.customer_service.handle(
        session_id=session_id,
        message=payload.message,
        request_id=request_id,
    )


@router.get("/api/v1/sessions/{session_id}/messages", tags=["agents"])
def conversation_history(session_id: str, request: Request) -> dict[str, Any]:
    """从 MySQL 事实源读取会话，而不是只返回可能过期的 Redis 副本。"""

    container = container_from(request)
    with container.database.session_factory() as db:
        messages = ConversationRepository(db).history(session_id, limit=50)
    return {"session_id": session_id, "messages": messages}


@router.post(
    "/api/v1/knowledge/documents",
    status_code=status.HTTP_201_CREATED,
    tags=["rag"],
)
def add_document(payload: DocumentCreate, request: Request) -> dict[str, Any]:
    """导入知识：分块、向量化、持久化，并失效相关缓存。"""

    container = container_from(request)
    # 数据流：DocumentCreate → RAGService.ingest → MySQL → cache invalidation。
    return container.rag.ingest(
        title=payload.title,
        source=payload.source,
        content=payload.content,
        metadata=payload.metadata,
    )


@router.get("/api/v1/knowledge/stats", tags=["rag"])
def knowledge_stats(request: Request) -> dict[str, int]:
    """返回文档和分块计数。"""

    return container_from(request).rag.stats()


@router.post("/api/v1/research", response_model=ResearchResponse, tags=["research"])
def create_research(payload: ResearchRequest, request: Request) -> dict[str, Any]:
    """同步运行研究子图并持久化结果。"""

    container = container_from(request)
    session_id = payload.session_id or f"research-{uuid.uuid4().hex[:12]}"
    # 数据流：topic → Planner → Retriever → Analyst → Writer → Reviewer → MySQL。
    return container.research.run(topic=payload.topic, session_id=session_id)


@router.get("/api/v1/research/{task_id}", tags=["research"])
def get_research(task_id: str, request: Request) -> dict[str, Any]:
    """按 task_id 读取已持久化的研究任务。"""

    result = container_from(request).research.get_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="研究任务不存在")
    return result
