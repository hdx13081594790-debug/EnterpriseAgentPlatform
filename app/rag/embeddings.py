"""Embedding 适配器。

数据流：文档/查询文本 → 分词与特征哈希（或 OpenAI）→ 定长浮点向量 →
MySQL 持久化或余弦相似度计算。
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings


class LocalHashEmbeddings(Embeddings):
    """用于离线演示的确定性特征哈希向量。

    支持英文词、中文单字与相邻 token 二元组。它便于测试但不等同于生产语义模型。
    """

    def __init__(self, dimension: int = 384):
        """保存向量维度；相同文本和维度始终得到相同结果。"""

        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量把文档块转换为向量。"""

        # 数据流：list[str] → 逐块 _embed → list[list[float]]。
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        """把单条查询转换为与文档相同空间中的向量。"""

        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        """执行 token 哈希、带符号桶累加和 L2 归一化。"""

        tokens = self._tokens(text)
        vector = [0.0] * self.dimension
        for token in tokens:
            # 稳定哈希决定“落入哪个维度”和“正负方向”，避免 Python hash 随进程变化。
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 1) & 1 else -1.0
            vector[index] += sign
        # 单位化后可以用点积直接表达余弦相似度。
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        """提取英文词/数字/中文单字，并加入相邻二元组保留局部顺序。"""

        lowered = text.lower()
        base = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
        pairs = [f"{base[index]}::{base[index + 1]}" for index in range(len(base) - 1)]
        return base + pairs


def build_embeddings(settings: Settings) -> Any:
    """根据配置返回 OpenAI Embedding 或本地确定性实现。"""

    # 数据流：Settings.embedding_provider → provider adapter → RAGService。
    if settings.embedding_provider.lower() == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须设置 OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return LocalHashEmbeddings(settings.embedding_dimension)
