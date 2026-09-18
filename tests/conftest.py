"""测试依赖：为每个用例创建隔离数据库与完整 FastAPI 生命周期。

数据流：tmp_path → SQLite Settings → create_app → TestClient → API 测试。
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture()
def settings(tmp_path) -> Settings:
    """使用临时 SQLite、内存缓存和 mock LLM，保证测试快速且可重复。"""

    return Settings(
        app_env="test",
        llm_provider="mock",
        embedding_provider="local_hash",
        database_url=f"sqlite:///{tmp_path / 'test.db'}",
        redis_url=None,
        rag_chunk_size=220,
        rag_chunk_overlap=30,
        research_quality_threshold=0.7,
        research_max_revisions=1,
    )


@pytest.fixture()
def client(settings: Settings):
    """进入 TestClient 时触发建表/播种，退出时触发容器关闭。"""

    # 数据流：测试 Settings → FastAPI lifespan → seeded API → 测试请求。
    app = create_app(settings, seed=True)
    with TestClient(app) as test_client:
        yield test_client
