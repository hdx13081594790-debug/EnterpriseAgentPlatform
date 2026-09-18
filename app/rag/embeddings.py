from __future__ import annotations

import hashlib
import math
import re
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings


class LocalHashEmbeddings(Embeddings):
    """Deterministic feature-hashing embeddings for offline demonstrations.

    This is intentionally transparent and dependency-free. It supports ASCII words,
    Chinese characters, and adjacent token pairs. It is suitable for tests and demos,
    not a replacement for a production semantic embedding model.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            index = value % self.dimension
            sign = 1.0 if (value >> 1) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

    @staticmethod
    def _tokens(text: str) -> list[str]:
        lowered = text.lower()
        base = re.findall(r"[a-z0-9_]+|[\u4e00-\u9fff]", lowered)
        pairs = [f"{base[index]}::{base[index + 1]}" for index in range(len(base) - 1)]
        return base + pairs


def build_embeddings(settings: Settings) -> Any:
    if settings.embedding_provider.lower() == "openai":
        if not settings.openai_api_key:
            raise ValueError("EMBEDDING_PROVIDER=openai 时必须设置 OPENAI_API_KEY")
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
        )
    return LocalHashEmbeddings(settings.embedding_dimension)

