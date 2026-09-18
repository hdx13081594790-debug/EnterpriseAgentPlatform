from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.db.models import Base


class Database:
    """Owns the SQLAlchemy engine and the application-wide session factory."""

    def __init__(self, settings: Settings):
        settings.ensure_local_directories()
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
        Base.metadata.create_all(self.engine)

    def session(self) -> Generator[Session, None, None]:
        db = self.session_factory()
        try:
            yield db
        finally:
            db.close()

    def health(self) -> bool:
        try:
            with self.engine.connect() as connection:
                connection.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    def close(self) -> None:
        self.engine.dispose()

