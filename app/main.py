from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import __version__
from app.api.routes import router
from app.core.config import Settings, get_settings
from app.db.seed import seed_all
from app.services.container import ApplicationContainer


def create_app(settings: Settings | None = None, *, seed: bool = True) -> FastAPI:
    active_settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, active_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        container = ApplicationContainer(active_settings)
        container.database.create_schema()
        if seed:
            seed_all(container.database, container.rag)
        app.state.container = container
        try:
            yield
        finally:
            container.close()

    application = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        description=(
            "融合 RAG、多智能体客服、研究工作流、MySQL 持久化与 Redis 缓存的面试项目。"
        ),
        lifespan=lifespan,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


app = create_app()

