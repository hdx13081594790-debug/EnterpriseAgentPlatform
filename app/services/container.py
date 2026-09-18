"""应用依赖组装入口（Composition Root）。

创建数据流：Settings → Database/Cache/LLM → RAG → Research → CustomerService。
销毁顺序则释放缓存与数据库连接；业务模块不自行创建全局依赖。
"""

from __future__ import annotations

from app.agents.customer_service import CustomerServiceWorkflow
from app.agents.research import ResearchWorkflow
from app.core.config import Settings
from app.db.database import Database
from app.infrastructure.cache import Cache
from app.rag.service import RAGService
from app.services.llm import LLMClient


class ApplicationContainer:
    """按依赖方向创建基础设施、服务和工作流。"""

    def __init__(self, settings: Settings):
        """构造进程级共享对象，避免每个请求重复初始化模型和连接池。"""

        self.settings = settings
        # 第一层：没有业务依赖的基础设施适配器。
        self.database = Database(settings)
        self.cache = Cache(settings)
        self.llm = LLMClient(settings)
        # 第二层：RAG 依赖数据库、缓存、Embedding 和 LLM。
        self.rag = RAGService(settings, self.database, self.cache, self.llm)
        # 第三层：研究图复用 RAG；客服总图再把研究图作为一个 Worker。
        self.research = ResearchWorkflow(settings, self.database, self.rag, self.llm)
        self.customer_service = CustomerServiceWorkflow(
            self.database,
            self.cache,
            self.rag,
            self.research,
            self.llm,
        )

    def close(self) -> None:
        """应用生命周期结束时释放外部资源。"""

        self.cache.close()
        self.database.close()
