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
    chunk_id: str
    document_id: str
    title: str
    source: str
    content: str
    score: float
    chunk_index: int


@dataclass(slots=True)
class RAGAnswer:
    answer: str
    sources: list[dict[str, Any]]
    confidence: float
    cache_hit: bool = False


class RAGService:
    """MySQL-backed document ingestion and in-process cosine retrieval."""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        cache: Cache,
        llm: LLMClient,
    ):
        self.settings = settings
        self.database = database
        self.cache = cache
        self.llm = llm
        self.embeddings = build_embeddings(settings)
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
        chunks = self.splitter.split_text(content)
        vectors = self.embeddings.embed_documents(chunks)
        chunk_rows = [
            (text, vector, len(text)) for text, vector in zip(chunks, vectors, strict=True)
        ]
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
            self.cache.delete_pattern(Cache.namespaced("rag", "*"))
        return {
            "document_id": document.id,
            "title": document.title,
            "chunks": chunk_count,
            "created": created,
        }

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        top_k = top_k or self.settings.rag_top_k
        key = Cache.namespaced("rag", f"search:{self._hash({'q': query, 'k': top_k})}")
        cached = self.cache.get_json(key)
        if isinstance(cached, list):
            return [RetrievedChunk(**item) for item in cached]

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
        self.cache.set_json(key, [asdict(item) for item in results])
        return results

    def answer(
        self,
        query: str,
        history: list[dict[str, str]] | None = None,
    ) -> RAGAnswer:
        history = history or []
        cache_key = Cache.namespaced(
            "rag",
            f"answer:{self._hash({'q': query, 'h': history[-4:]})}",
        )
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, dict):
            cached["cache_hit"] = True
            return RAGAnswer(**cached)

        chunks = self.retrieve(query)
        if not chunks:
            result = RAGAnswer(
                answer="知识库中还没有可检索的资料，请先通过知识库接口导入文档。",
                sources=[],
                confidence=0.0,
            )
            return result

        context = "\n\n".join(
            f"[{index}] {chunk.title}\n{chunk.content}"
            for index, chunk in enumerate(chunks, start=1)
        )
        history_text = "\n".join(
            f"{item.get('role', 'user')}: {item.get('content', '')}" for item in history[-4:]
        )
        fallback = self._fallback_answer(query, chunks)
        answer = self.llm.text(
            system=(
                "你是严格遵循证据的知识库助手。只能根据给定资料回答；资料不足时明确说明。"
                "引用事实时使用 [1]、[2] 标注来源，不要编造。"
            ),
            user=f"对话历史：\n{history_text or '无'}\n\n资料：\n{context}\n\n问题：{query}",
            fallback=fallback,
        )
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
        positive_scores = [max(0.0, chunk.score) for chunk in chunks]
        confidence = round(min(0.98, max(0.15, sum(positive_scores) / len(chunks))), 3)
        result = RAGAnswer(answer=answer, sources=sources, confidence=confidence)
        self.cache.set_json(cache_key, asdict(result))
        return result

    def stats(self) -> dict[str, int]:
        with self.database.session_factory() as db:
            repository = KnowledgeRepository(db)
            return {
                "documents": repository.count_documents(),
                "chunks": repository.count_chunks(),
            }

    @staticmethod
    def _fallback_answer(query: str, chunks: list[RetrievedChunk]) -> str:
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
        if not left or not right or len(left) != len(right):
            return 0.0
        dot = sum(a * b for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
        right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
        return dot / (left_norm * right_norm)

    @staticmethod
    def _hash(value: Any) -> str:
        raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

