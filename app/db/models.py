"""数据库表的 ORM 映射。

总体数据流：API/Agent → Repository → 这些 ORM 实体 → MySQL；读取方向相反。
JSON 字段用于保存元数据、特征、来源和演示向量，避免把灵活结构拆成过多表。
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utc_now() -> datetime:
    """统一生成带时区的 UTC 时间，避免多服务器时区不一致。"""

    return datetime.now(UTC)


def new_id() -> str:
    """生成跨数据库兼容的 UUID 字符串主键。"""

    return str(uuid.uuid4())


class Base(DeclarativeBase):
    """所有 ORM 模型共享的 SQLAlchemy 声明式基类。"""

    pass


class KnowledgeDocument(Base):
    """RAG 原始文档；``content_hash`` 保证同内容幂等导入。"""

    __tablename__ = "knowledge_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    source: Mapped[str] = mapped_column(String(500), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )

    # 数据流：一篇原文 → 多个 KnowledgeChunk；删除原文时级联清理分块。
    chunks: Mapped[list[KnowledgeChunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class KnowledgeChunk(Base):
    """RAG 文本块、演示向量及其在原文中的顺序。"""

    __tablename__ = "knowledge_chunks"
    __table_args__ = (Index("ix_knowledge_chunks_document_index", "document_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("knowledge_documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(JSON, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    document: Mapped[KnowledgeDocument] = relationship(back_populates="chunks")


class Product(Base):
    """商品事实数据，由 Product Agent 按关键词和预算查询。"""

    __tablename__ = "products"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    name: Mapped[str] = mapped_column(String(120), unique=True, nullable=False, index=True)
    category: Mapped[str] = mapped_column(String(80), nullable=False, index=True)
    price: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    features: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    stock: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    rating: Mapped[float] = mapped_column(Float, nullable=False, default=0)


class Order(Base):
    """订单与物流事实数据，由 Order Agent 根据订单号精确查询。"""

    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    product_name: Mapped[str] = mapped_column(String(120), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(10, 2), nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    carrier: Mapped[str | None] = mapped_column(String(80))
    tracking_number: Mapped[str | None] = mapped_column(String(80))
    estimated_delivery: Mapped[str | None] = mapped_column(String(40))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class ChatMessage(Base):
    """完整会话记录；Redis 仅缓存其中最近的热数据。"""

    __tablename__ = "chat_messages"
    __table_args__ = (Index("ix_chat_session_created", "session_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    agent_name: Mapped[str | None] = mapped_column(String(80))
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )


class ResearchTask(Base):
    """研究工作流的任务状态、报告、来源与质量分。"""

    __tablename__ = "research_tasks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    session_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    topic: Mapped[str] = mapped_column(String(500), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    report: Mapped[str | None] = mapped_column(Text)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    quality_score: Mapped[float] = mapped_column(Float, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )


class AgentTrace(Base):
    """客服图每个节点的轻量执行轨迹，用于定位路由与延迟问题。"""

    __tablename__ = "agent_traces"
    __table_args__ = (Index("ix_trace_request_created", "request_id", "created_at"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    node_name: Mapped[str] = mapped_column(String(80), nullable=False)
    input_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    output_summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
