from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


@pytest.fixture()
def settings(tmp_path) -> Settings:
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
    app = create_app(settings, seed=True)
    with TestClient(app) as test_client:
        yield test_client

