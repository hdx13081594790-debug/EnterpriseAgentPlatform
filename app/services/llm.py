from __future__ import annotations

import logging
from typing import Any, TypeVar

from pydantic import BaseModel

from app.core.config import Settings

logger = logging.getLogger(__name__)
SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMClient:
    """Small anti-corruption layer around LangChain chat models.

    Workflows depend on this interface instead of a vendor SDK. ``mock`` mode is
    deterministic and makes the full project demonstrable without an API key.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.provider = settings.llm_provider.lower()
        self.model: Any = None
        if self.provider != "mock":
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
        return self.provider == "mock"

    def text(self, *, system: str, user: str, fallback: str) -> str:
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

