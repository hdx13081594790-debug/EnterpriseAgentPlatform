"""大模型供应商适配层。

数据流：Agent 的 system/user 输入 → LangChain ChatModel → 文本/Pydantic 对象；
mock 模式或调用异常 → 调用方提供的确定性 fallback。
"""

from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.config import Settings

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient:
    """隔离 LangChain 与具体供应商差异的防腐层。

    工作流只依赖本接口，不直接依赖 Groq/OpenAI SDK。``mock`` 模式可重复、无密钥，
    适合自动测试和面试现场演示。
    """

    def __init__(self, settings: Settings):
        """按 provider 延迟创建模型；mock 模式完全不建立外部连接。"""

        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.model: Any = None
        if self.provider != "mock":
            # 数据流：Settings → init_chat_model → 统一 LangChain Runnable 接口。
            from langchain.chat_models import init_chat_model

            api_key = (
                settings.groq_api_key if self.provider == "groq" else settings.openai_api_key
            )
            self.model = init_chat_model(
                settings.llm_model,
                model_provider=self.provider,
                api_key=api_key,
                temperature=0.1,
            )

    @property
    def is_mock(self) -> bool:
        """指出当前是否跳过所有外部模型调用。"""

        return self.provider == "mock"

    def text(self, *, system: str, user: str, fallback: str) -> str:
        """执行普通文本生成；异常时返回业务定义的安全文案。"""

        # 数据流：mock → fallback；真实模式 → messages → model.invoke → string。
        if self.is_mock:
            return fallback
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = self.model.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            return self._content_to_text(response.content).strip() or fallback
        except Exception as exc:
            logger.exception("LLM call failed; using fallback: %s", exc)
            return fallback

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: type[SchemaT],
        fallback: SchemaT,
    ) -> SchemaT:
        """执行结构化生成，并用 Pydantic Schema 校验模型输出。"""

        # 数据流：Prompt → with_structured_output(schema) → 校验对象；失败 → fallback。
        if self.is_mock:
            return fallback
        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            structured_model = self.model.with_structured_output(schema)
            result = structured_model.invoke(
                [SystemMessage(content=system), HumanMessage(content=user)]
            )
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as exc:
            logger.exception("Structured LLM call failed; using fallback: %s", exc)
            return fallback

    @staticmethod
    def _content_to_text(content: Any) -> str:
        """兼容供应商返回字符串或多内容块两种消息格式。"""

        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and "text" in block:
                    parts.append(str(block["text"]))
            return "\n".join(parts)
        return str(content)
