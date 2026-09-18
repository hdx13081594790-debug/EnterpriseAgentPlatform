"""数据访问层，把业务语义转换为受控 SQLAlchemy 查询。

数据流：Service/Agent → Repository 方法 → ORM/SQL → MySQL；查询结果再返回上层。
上层不直接拼接 SQL，因而更容易测试、替换数据库并约束 Agent 权限。
"""

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
    """知识文档和分块的持久化入口。"""

    def __init__(self, db: Session):
        """绑定当前工作单元的短生命周期 Session。"""

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
        """按内容哈希幂等写入文档与所有分块，并返回是否为新文档。"""

        # 数据流：原文 → SHA-256 → 查重；未命中 → document + chunks → 单次提交。
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
        # flush 只把 INSERT 发送到当前事务，以便取得主键；真正提交在所有分块加入后。
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
        """一次取出候选分块及所属文档，供演示版应用层余弦检索。"""

        # 数据流：knowledge_chunks JOIN document → RAGService → 相似度排序。
        statement: Select = (
            select(KnowledgeChunk)
            .options(joinedload(KnowledgeChunk.document))
            .order_by(KnowledgeChunk.document_id, KnowledgeChunk.chunk_index)
            .limit(limit)
        )
        return list(self.db.scalars(statement).unique().all())

    def count_documents(self) -> int:
        """统计知识文档数，供健康检查与演示面板使用。"""

        return int(self.db.scalar(select(func.count(KnowledgeDocument.id))) or 0)

    def count_chunks(self) -> int:
        """统计知识分块数。"""

        return int(self.db.scalar(select(func.count(KnowledgeChunk.id))) or 0)


class CommerceRepository:
    """订单与商品的只读查询入口。"""

    def __init__(self, db: Session):
        """绑定当前查询使用的 Session。"""

        self.db = db

    def get_order(self, order_id: str) -> Order | None:
        """使用主键精确查询订单，避免模型猜测业务状态。"""

        return self.db.get(Order, order_id.upper())

    def search_products(
        self,
        keyword: str | None = None,
        budget: Decimal | None = None,
        limit: int = 5,
    ) -> list[Product]:
        """组合可选关键词和预算过滤，并按评分与价格排序。"""

        # 数据流：用户条件 → SQL WHERE/ORDER/LIMIT → Product Agent 格式化。
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
    """会话的长期存储入口。"""

    def __init__(self, db: Session):
        """绑定当前会话读写使用的 Session。"""

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
        """持久化单条消息；助手消息可附 Agent 名称和质量元数据。"""

        # 数据流：LangGraph 最终状态 → chat_messages → MySQL。
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
        """取最近消息并恢复为正序，供 Redis 回填和 Agent 上下文使用。"""

        # SQL 先倒序 LIMIT 提高效率，返回前再翻转为自然对话顺序。
        statement = (
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(limit)
        )
        rows = list(reversed(self.db.scalars(statement).all()))
        return [{"role": row.role, "content": row.content} for row in rows]


class ResearchRepository:
    """研究任务状态机的持久化入口。"""

    def __init__(self, db: Session):
        """绑定当前研究任务事务使用的 Session。"""

        self.db = db

    def create(self, session_id: str, topic: str) -> ResearchTask:
        """在运行研究图之前创建 ``running`` 任务记录。"""

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
        """研究图成功结束后原子写入报告、来源与质量分。"""

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
        """异常路径把任务标记为失败，避免记录永久停留在 running。"""

        row = self.db.get(ResearchTask, task_id)
        if row:
            row.status = "failed"
            row.report = message
            self.db.commit()

    def get(self, task_id: str) -> ResearchTask | None:
        """按任务 ID 读取研究结果。"""

        return self.db.get(ResearchTask, task_id)


class TraceRepository:
    """Agent 节点追踪写入入口。"""

    def __init__(self, db: Session):
        """绑定当前追踪写入使用的 Session。"""

        self.db = db

    def add(
        self,
        request_id: str,
        node_name: str,
        input_summary: str,
        output_summary: str,
        latency_ms: int,
    ) -> None:
        """保存裁剪后的输入/输出摘要和耗时，防止追踪表无限放大单行。"""

        # 数据流：节点 wrapper → AgentTrace → MySQL；失败时上层选择不中断主业务。
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
