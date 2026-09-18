from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed configuration loaded from environment variables or ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_name: str = "Interview Agent Platform"
    app_env: str = "development"
    log_level: str = "INFO"

    llm_provider: str = "mock"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = None
    openai_api_key: str | None = None

    embedding_provider: str = "local_hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=384, ge=64, le=3072)

    database_url: str = "sqlite:///./data/interview_agent.db"
    redis_url: str | None = None
    redis_ttl_seconds: int = Field(default=900, ge=10)
    session_ttl_seconds: int = Field(default=86_400, ge=60)

    rag_chunk_size: int = Field(default=500, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=80, ge=0, le=1000)
    rag_top_k: int = Field(default=4, ge=1, le=20)

    research_quality_threshold: float = Field(default=0.78, ge=0, le=1)
    research_max_revisions: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> Settings:
        provider = self.llm_provider.lower()
        if provider == "groq" and not self.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq 时必须设置 GROQ_API_KEY")
        if provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai 时必须设置 OPENAI_API_KEY")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP 必须小于 RAG_CHUNK_SIZE")
        return self

    def ensure_local_directories(self) -> None:
        if self.database_url.startswith("sqlite") and "///" in self.database_url:
            db_path = self.database_url.split("///", 1)[1]
            if db_path != ":memory:":
                Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

