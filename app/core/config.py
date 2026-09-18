"""应用配置中心。

数据流：操作系统环境变量 / ``.env`` → Pydantic 校验 → ``Settings`` →
数据库、缓存、LLM、RAG 和 LangGraph 工作流。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """从环境变量或 ``.env`` 读取并校验的强类型配置。

    配置对象只负责描述依赖，不在这里建立网络连接；真正的连接由
    ``ApplicationContainer`` 按依赖顺序创建。
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # 数据流：APP_* → FastAPI 元信息与日志配置。
    app_name: str = "Interview Agent Platform"
    app_env: str = "development"
    log_level: str = "INFO"

    # 数据流：LLM_* / API Key → LLMClient → Agent 文本或结构化输出。
    llm_provider: str = "mock"
    llm_model: str = "openai/gpt-oss-120b"
    groq_api_key: str | None = None
    openai_api_key: str | None = None

    # 数据流：Embedding 配置 → build_embeddings → 文档向量 / 查询向量。
    embedding_provider: str = "local_hash"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = Field(default=384, ge=64, le=3072)

    # 数据流：连接串 → SQLAlchemy Engine → Repository → MySQL/SQLite。
    database_url: str = "sqlite:///./data/interview_agent.db"
    # 数据流：Redis URL → Cache；为空时使用进程内 TTL 缓存。
    redis_url: str | None = None
    redis_ttl_seconds: int = Field(default=900, ge=10)
    session_ttl_seconds: int = Field(default=86_400, ge=60)

    # 数据流：RAG 参数 → 分块 / Top-K 检索；研究 Agent 复用同一检索服务。
    rag_chunk_size: int = Field(default=500, ge=100, le=4000)
    rag_chunk_overlap: int = Field(default=80, ge=0, le=1000)
    rag_top_k: int = Field(default=4, ge=1, le=20)

    # 数据流：评审分数 + 循环次数 → LangGraph 条件边（重写或结束）。
    research_quality_threshold: float = Field(default=0.78, ge=0, le=1)
    research_max_revisions: int = Field(default=1, ge=0, le=3)

    @model_validator(mode="after")
    def validate_provider_credentials(self) -> Settings:
        """在创建任何外部客户端前，先阻止无密钥或非法分块配置。"""

        provider = self.llm_provider.lower()
        if provider == "groq" and not self.groq_api_key:
            raise ValueError("LLM_PROVIDER=groq 时必须设置 GROQ_API_KEY")
        if provider == "openai" and not self.openai_api_key:
            raise ValueError("LLM_PROVIDER=openai 时必须设置 OPENAI_API_KEY")
        if self.rag_chunk_overlap >= self.rag_chunk_size:
            raise ValueError("RAG_CHUNK_OVERLAP 必须小于 RAG_CHUNK_SIZE")
        return self

    def ensure_local_directories(self) -> None:
        """SQLite 模式下创建数据目录；MySQL URL 不需要本地目录。"""

        if self.database_url.startswith("sqlite") and "///" in self.database_url:
            db_path = self.database_url.split("///", 1)[1]
            if db_path != ":memory:":
                Path(db_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """返回进程级配置单例，避免每个请求重复解析 ``.env``。"""

    # 数据流：首次调用构造 Settings → lru_cache → 后续调用复用同一实例。
    return Settings()
