"""RAG 核心服务：知识导入、向量召回、答案生成和缓存。

导入数据流：原文 → 分块 → Embedding → Repository → MySQL → 清理 Redis。
问答数据流：问题 → Redis → 查询向量 → MySQL 分块 → Top-K → LLM → 来源与答案。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings
from app.db.database import Database
from app.db.repositories import KnowledgeRepository
from app.infrastructure.cache import Cache
from app.rag.embeddings import build_embeddings
from app.services.llm import LLMClient


@dataclass(slots=True)
class RetrievedChunk:
    """检索层向上返回的稳定数据结构，隔离 ORM 实体。"""

    chunk_id: str
    document_id: str
    title: str
    source: str
    content: str
    score: float
    chunk_index: int


@dataclass(slots=True)
class RAGAnswer:
    """RAG 对上层 Agent 的统一输出。"""

    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    cache_hit: bool = False


class RAGService:
    """MySQL 持久化、应用层余弦检索的 RAG 服务。"""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        cache: Cache,
        llm: LLMClient,
    ):
        """组装分块器、Embedding、数据库、缓存和 LLM 依赖。"""

        self.settings = settings
        self.database = database
        self.cache = cache
        self.llm = llm
        self.embeddings = build_embeddings(settings)
        # 分隔符从段落逐级降到字符，尽量在语义边界切分。
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.rag_chunk_size,
            chunk_overlap=settings.rag_chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""],
        )

    def ingest(
        self,
        *,
        title: str,
        source: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """幂等导入一篇文档并返回文档 ID、分块数和是否新建。"""

        # 数据流 1：content → RecursiveCharacterTextSplitter → chunks。
        chunks = self.splitter.split_text(content)
        # 数据流 2：chunks → embedding provider → vectors。
        vectors = self.embeddings.embed_documents(chunks)
        chunk_rows = [
            (text, vector, len(text)) for text, vector in zip(chunks, vectors, strict=True)
        ]
        # 数据流 3：文本块 + 向量 → KnowledgeRepository → MySQL 事务。
        with self.database.session_factory() as db:
            document, created = KnowledgeRepository(db).add_document(
                title=title,
                source=source,
                content=content,
                metadata=metadata or {},
                chunks=chunk_rows,
            )
            chunk_count = len(document.chunks) if not created else len(chunks)
        if created:
            # 写后失效：知识发生变化后删除检索与答案缓存，防止返回旧结果。
            self.cache.delete_pattern(Cache.namespaced("rag", "*"))
        return {
            "document_id": document.id,
            "title": document.title,
            "chunks": chunk_count,
            "created": created,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """返回与查询最相关的 Top-K 分块，优先读取 Redis。"""

        top_k = top_k or self.settings.rag_top_k
        # 查询文本和 top_k 一起参与哈希，避免不同召回数量复用错误缓存。
        key = Cache.namespaced("rag", f"search:{self._hash({'q': query, 'k': top_k})}")
        cached = self.cache.get_json(key)
        if isinstance(cached, list):
            # 数据流（命中）：Redis JSON → RetrievedChunk → 调用方。
            return [RetrievedChunk(**item) for item in cached]

        # 数据流（未命中）：query → query vector；MySQL chunks → cosine → sort → Top-K。
        query_vector = self.embeddings.embed_query(query)
        with self.database.session_factory() as db:
            chunks = KnowledgeRepository(db).all_chunks()
            scored = [
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    title=chunk.document.title,
                    source=chunk.document.source,
                    content=chunk.content,
                    score=self._cosine(query_vector, chunk.embedding),
                    chunk_index=chunk.chunk_index,
                )
                for chunk in chunks
            ]
        results = sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]
        # 召回结果可重建，因此只写缓存，不作为事实数据持久化。
        self.cache.set_json(key, [asdict(item) for item in results])
        return results

    def answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> RAGAnswer:
        """把召回证据组装成上下文，生成带来源与置信度的回答。"""

        history = history or []
        # 最近历史参与答案缓存键，避免多轮上下文不同却错误共用答案。
        cache_key = Cache.namespaced(
            "rag",
            f"answer:{self._hash({'q': query, 'h': history[-4:]})}",
        )
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            # 数据流（命中）：Redis answer → 标记 cache_hit → 上层 Agent。
            cached["cache_hit"] = True
            return RAGAnswer(**cached)

        # 数据流（未命中）：query → retrieve → Top-K 证据。
        chunks = self.retrieve(query)
        if not chunks:
            result = RAGAnswer(
                answer="知识库中还没有可检索的资料，请先通过知识库接口导入文档。",
                sources=[],
                confidence=0.0,
            )
            return result

        # 为每个证据块编号，模型回答中的 [n] 能映射回 sources[n-1]。
        context = "\n\n".join(
            f"[{index}] {chunk.title}\n{chunk.content}"
            for index, chunk in enumerate(chunks, start=1)
        )
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-4:]
        )
        fallback = self._fallback_answer(query, chunks)
        # 数据流：上下文 + 对话历史 + 问题 → LLM/fallback → answer。
        answer = self.llm.text(
            system=(
                "你是严格遵循证据的知识库助手。只能根据给定资料回答；资料不足时明确说明。"
                "引用事实时使用 [1]、[2] 标注来源，不要编造。"
            ),
            user=f"对话历史：\n{history_text or '无'}\n\n资料：\n{context}\n\n问题：{query}",
            fallback=fallback,
        )
        # ORM/内部向量不直接暴露；API 只返回可追溯的来源摘要与相似度。
        sources = [
            {
                "index": index,
                "document_id": chunk.document_id,
                "title": chunk.title,
                "source": chunk.source,
                "score": round(chunk.score, 4),
                "preview": chunk.content[:160],
            }
            for index, chunk in enumerate(chunks, start=1)
        ]
        # 演示置信度来自召回相似度，不声称它是经过校准的真实概率。
        positive_scores = [max(0.0, chunk.score) for chunk in chunks]
        confidence = round(min(0.98, max(0.15, sum(positive_scores) / len(chunks))), 3)
        result = RAGAnswer(answer=answer, sources=sources, confidence=confidence)
        self.cache.set_json(cache_key, asdict(result))
        return result

    def stats(self) -> dict[str, int]:
        """读取知识库规模，供健康检查和演示使用。"""

        with self.database.session_factory() as db:
            repository = KnowledgeRepository(db)
            return {
                "documents": repository.count_documents(),
                "chunks": repository.count_chunks(),
            }

    @staticmethod
    def _fallback_answer(query: str, chunks: list[RetrievedChunk]) -> str:
        """mock/模型故障时，直接用检索片段生成可追溯回答。"""

        excerpts = []
        for index, chunk in enumerate(chunks[:3], start=1):
            compact = " ".join(chunk.content.split())
            excerpts.append(f"- {compact[:220]} [{index}]")
        return (
            f"针对“{query}”，知识库中最相关的信息如下：\n\n"
            + "\n".join(excerpts)
            + "\n\n当前为离线演示模式，以上内容直接来自检索结果。"
        )

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        """计算两个等长向量的余弦相似度。"""

        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _hash(value: Any) -> str:
        """稳定序列化缓存参数并生成不泄漏原文的固定长度 Key 后缀。"""

        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
