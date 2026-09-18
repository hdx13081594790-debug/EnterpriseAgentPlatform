from __future__ import annotations

from app.agents.customer_service import CustomerServiceWorkflow
from app.agents.research import ResearchWorkflow
from app.core.config import Settings
from app.db.database import Database
from app.infrastructure.cache import Cache
from app.rag.service import RAGService
from app.services.llm import LLMClient


class ApplicationContainer:
    """Composition root: creates adapters first and workflows last."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.database = Database(settings)
        self.cache = Cache(settings)
        self.llm = LLMClient(settings)
        self.rag = RAGService(settings, self.database, self.cache, self.llm)
        self.research = ResearchWorkflow(settings, self.database, self.rag, self.llm)
        self.customer_service = CustomerServiceWorkflow(
            self.database,
            self.cache,
            self.rag,
            self.research,
            self.llm,
        )

    def close(self) -> None:
        self.cache.close()
        self.database.close()

