"""SQLAlchemy 连接与会话生命周期管理。

数据流：``Settings.database_url`` → Engine/连接池 → Session → Repository → 数据库。
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import Base


class Database:
    """持有全局 Engine 和 Session Factory，但不共享具体 Session。"""

    def __init__(self, settings: Settings):
        """根据数据库类型创建连接池；每个业务操作再创建短会话。"""

        settings.ensure_local_directories()
        # SQLite 连接默认限制创建线程；FastAPI 同步端点在线程池运行，所以关闭该限制。
        connect_args = (
            {"check_same_thread": False}
            if settings.database_url.startswith("sqlite")
            else {}
        )
        engine_kwargs: dict = {
            "pool_pre_ping": True,
            "future": True,
            "connect_args": connect_args,
        }
        if not settings.database_url.startswith("sqlite"):
            # MySQL 长连接可能被服务端回收，pre_ping + recycle 降低失效连接概率。
            engine_kwargs.update({"pool_recycle": 1800, "pool_size": 10, "max_overflow": 20})

        self.engine: Engine = create_engine(settings.database_url, **engine_kwargs)
        self.session_factory = sessionmaker(
            bind=self.engine,
            class_=Session,
            expire_on_commit=False,
            autoflush=False,
            autocommit=False,
        )

    def create_schema(self) -> None:
        """根据 ORM 元数据创建缺失表；生产环境应替换为 Alembic 迁移。"""

        # 数据流：ORM models → SQL DDL → MySQL/SQLite 表。
        Base.metadata.create_all(self.engine)

    def session(self) -> Generator[Session, None, None]:
        """可用于 FastAPI 依赖注入的 Session 生成器。"""

        # 数据流：SessionFactory → 当前请求/操作 → finally 关闭连接归还连接池。
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def health(self) -> bool:
        """执行最小只读 SQL，供健康检查判断数据库是否可达。"""

        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        """应用退出时释放连接池资源。"""

        self.engine.dispose()
