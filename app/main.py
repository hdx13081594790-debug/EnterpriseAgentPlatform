"""FastAPI 应用入口与生命周期管理。

启动数据流：配置 → Container → 建表 → 幂等种子数据 → 接收请求。
关闭数据流：停止接收请求 → Cache/Database 释放连接。
"""

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
    """创建可配置的 FastAPI 实例；测试可传入隔离配置并控制是否播种。"""

    # 数据流：显式测试配置优先；否则读取进程级环境配置。
    active_settings = settings or get_settings()
    logging.basicConfig(
        level=getattr(logging, active_settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """只在应用启动/关闭各执行一次的资源生命周期。"""

        # 启动方向：Settings → Container → schema → seed → app.state。
        container = ApplicationContainer(active_settings)
        container.database.create_schema()
        if seed:
            seed_all(container.database, container.rag)
        # 路由通过 request.app.state 取得同一个容器，不使用隐藏的模块全局变量。
        app.state.container = container
        try:
            yield
        finally:
            # 关闭方向：FastAPI lifespan → Container → Redis/DB 连接池。
            container.close()

    application = FastAPI(
        title=active_settings.app_name,
        version=__version__,
        description=(
            "融合 RAG、多智能体客服、研究工作流、MySQL 持久化与 Redis 缓存的面试项目。"
        ),
        lifespan=lifespan,
    )
    # 浏览器前端跨域白名单；生产环境应由配置提供真实域名。
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.include_router(router)
    return application


# Uvicorn 默认导入的 ASGI 对象：``uvicorn app.main:app``。
app = create_app()
