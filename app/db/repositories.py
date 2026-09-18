from __future__ import annotations

import hashlib
from collections.abc import Iterable
from decimal import Decimal
from typing import Any

from sqlalchemy import Select, func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import (
    AgentTrace,
    ChatMessage,
    KnowledgeChunk,
    KnowledgeDocument,
    Order,
    Product,
    ResearchTask,
)


class KnowledgeRepository:
    def __init__(self, db: Session):
        self.db = db

    def add_document(
        self,
        *,
        title: str,
        source: str,
        content: str,
        metadata: dict[str, Any],
        chunks: Iterable[tuple[str, list[float], int]],
    ) -> tuple[KnowledgeDocument, bool]:
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        existing = self.db.scalar(
            select(KnowledgeDocument).where(KnowledgeDocument.content_hash == content_hash)
        )
        if existing:
            return existing, False

        document = KnowledgeDocument(
            title=title,
            source=source,
            content=content,
            content_hash=content_hash,
            metadata_json=metadata,
        )
        self.db.add(document)
        self.db.flush()
        for index, (chunk_text, embedding, token_count) in enumerate(chunks):
            self.db.add(
                KnowledgeChunk(
                    document_id=document.id,
                    chunk_index=index,
                    content=chunk_text,
                    embedding=embedding,
                    token_count=token_count,
                )
            )
        self.db.commit()
        self.db.refresh(document)
        return document, True

    def all_chunks(self, limit: int = 5000) -> list[KnowledgeChunk]:
        statement: Select = (
            select(KnowledgeChunk)
            .options(joinedload(KnowledgeChunk.document))
            .order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)
            .limit(limit)
        )
        return list(self.db.scalars(statement).unique().all())

    def count_documents(self) -> int:
        return int(self.db.scalar(select(func.count(KnowledgeDocument.id))) or 0)

    def count_chunks(self) -> int:
        return int(self.db.scalar(select(func.count(KnowledgeChunk.id))) or 0)


class CommerceRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_order(self, order_id: str) -> Order | None:
        return self.db.get(Order, order_id.upper())

    def search_products(
        self,
        keyword: str | None = None,
        budget: Decimal | None = None,
        limit: int = 5,
    ) -> list[Product]:
        statement = select(Product)
        if keyword:
            fuzzy = f"%{keyword}%"
            statement = statement.where(
                or_(Product.name.like(fuzzy), Product.category.like(fuzzy))
            )
        if budget is not None:
            statement = statement.where(Product.price <= budget)
        statement = statement.order_by(Product.rating.desc(), Product.price.asc()).limit(limit)
        return list(self.db.scalars(statement).all())


class ConversationRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(
        self,
        session_id: str,
        role: str,
        content: str,
        *,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ChatMessage:
        row = ChatMessage(
            session_id=session_id,
            role=role,
            content=content,
            agent_name=agent_name,
            metadata_json=metadata or {},
        )
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def history(self, session_id: str, limit: int = 12) -> list[dict[str, str]]:
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        rows = list(reversed(self.db.scalars(statement).all()))
        return [{"role": row.role, "content": row.content} for row in rows]


class ResearchRepository:
    def __init__(self, db: Session):
        self.db = db

    def create(self, session_id: str, topic: str) -> ResearchTask:
        row = ResearchTask(session_id=session_id, topic=topic, status="running")
        self.db.add(row)
        self.db.commit()
        self.db.refresh(row)
        return row

    def complete(
        self,
        task_id: str,
        *,
        report: str,
        sources: list[dict[str, Any]],
        quality_score: float,
    ) -> ResearchTask:
        row = self.db.get(ResearchTask, task_id)
        if row is None:
            raise LookupError(f"研究任务不存在: {task_id}")
        row.status = "completed"
        row.report = report
        row.sources = sources
        row.quality_score = quality_score
        self.db.commit()
        self.db.refresh(row)
        return row

    def fail(self, task_id: str, message: str) -> None:
        row = self.db.get(ResearchTask, task_id)
        if row:
            row.status = "failed"
            row.report = message
            self.db.commit()

    def get(self, task_id: str) -> ResearchTask | None:
        return self.db.get(ResearchTask, task_id)


class TraceRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(
        self,
        request_id: str,
        node_name: str,
        input_summary: str,
        output_summary: str,
        latency_ms: int,
    ) -> None:
        self.db.add(
            AgentTrace(
                request_id=request_id,
                node_name=node_name,
                input_summary=input_summary[:2000],
                output_summary=output_summary[:2000],
                latency_ms=latency_ms,
            )
        )
        self.db.commit()

