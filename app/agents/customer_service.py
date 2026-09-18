from __future__ import annotations

import re
import time
from collections.abc import Callable
from decimal import Decimal
from functools import wraps
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.agents.research import ResearchWorkflow
from app.db.database import Database
from app.db.repositories import CommerceRepository, ConversationRepository, TraceRepository
from app.infrastructure.cache import Cache
from app.rag.service import RAGService
from app.services.llm import LLMClient

Intent = Literal[
    "knowledge",
    "tech_support",
    "order_service",
    "product_consult",
    "research",
    "human",
]


class IntentDecision(BaseModel):
    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str


class QualityDecision(BaseModel):
    score: float = Field(ge=0, le=1)
    needs_human: bool
    reason: str


class CustomerServiceState(TypedDict, total=False):
    request_id: str
    session_id: str
    user_message: str
    history: list[dict[str, str]]
    intent: Intent
    route_confidence: float
    route_reason: str
    agent_name: str
    response: str
    sources: list[dict[str, Any]]
    quality_score: float
    needs_human: bool
    cache_hit: bool


class CustomerServiceWorkflow:
    """Supervisor graph that routes requests to specialized worker agents."""

    def __init__(
        self,
        database: Database,
        cache: Cache,
        rag: RAGService,
        research: ResearchWorkflow,
        llm: LLMClient,
    ):
        self.database = database
        self.cache = cache
        self.rag = rag
        self.research = research
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(CustomerServiceState)
        graph.add_node("load_context", self._traced("load_context", self._load_context))
        graph.add_node("supervisor", self._traced("supervisor", self._classify))
        graph.add_node("rag_agent", self._traced("rag_agent", self._knowledge_agent))
        graph.add_node("tech_agent", self._traced("tech_agent", self._tech_agent))
        graph.add_node("order_agent", self._traced("order_agent", self._order_agent))
        graph.add_node("product_agent", self._traced("product_agent", self._product_agent))
        graph.add_node("research_agent", self._traced("research_agent", self._research_agent))
        graph.add_node("human_agent", self._traced("human_agent", self._human_agent))
        graph.add_node("quality_agent", self._traced("quality_agent", self._quality_agent))
        graph.add_node(
            "human_review_notice",
            self._traced("human_review_notice", self._human_review_notice),
        )
        graph.add_node("persist", self._traced("persist", self._persist))

        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "supervisor")
        graph.add_conditional_edges(
            "supervisor",
            lambda state: state["intent"],
            {
                "knowledge": "rag_agent",
                "tech_support": "tech_agent",
                "order_service": "order_agent",
                "product_consult": "product_agent",
                "research": "research_agent",
                "human": "human_agent",
            },
        )
        for node in ["rag_agent", "tech_agent", "order_agent", "product_agent", "research_agent"]:
            graph.add_edge(node, "quality_agent")
        graph.add_edge("human_agent", "persist")
        graph.add_conditional_edges(
            "quality_agent",
            lambda state: "human" if state.get("needs_human") else "ok",
            {"human": "human_review_notice", "ok": "persist"},
        )
        graph.add_edge("human_review_notice", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def handle(self, *, session_id: str, message: str, request_id: str) -> dict[str, Any]:
        initial: CustomerServiceState = {
            "request_id": request_id,
            "session_id": session_id,
            "user_message": message,
            "history": [],
            "intent": "knowledge",
            "route_confidence": 0.0,
            "route_reason": "",
            "agent_name": "",
            "response": "",
            "sources": [],
            "quality_score": 0.0,
            "needs_human": False,
            "cache_hit": False,
        }
        result = self.graph.invoke(initial)
        return {
            "request_id": request_id,
            "session_id": session_id,
            "intent": result["intent"],
            "agent": result["agent_name"],
            "answer": result["response"],
            "sources": result.get("sources", []),
            "route_confidence": result["route_confidence"],
            "quality_score": result["quality_score"],
            "needs_human": result["needs_human"],
            "cache_hit": result.get("cache_hit", False),
        }

    def _load_context(self, state: CustomerServiceState) -> dict[str, Any]:
        history = self.cache.get_history(state["session_id"])
        if history is None:
            with self.database.session_factory() as db:
                history = ConversationRepository(db).history(state["session_id"])
            self.cache.set_history(state["session_id"], history)
        return {"history": history}

    def _classify(self, state: CustomerServiceState) -> dict[str, Any]:
        fallback = self._rule_route(state["user_message"])
        decision = self.llm.structured(
            system=(
                "你是客服总控 Agent。将请求严格分类为 knowledge、tech_support、order_service、"
                "product_consult、research 或 human。投诉和明确要求人工归为 human。"
            ),
            user=state["user_message"],
            schema=IntentDecision,
            fallback=fallback,
        )
        if decision.confidence < 0.55:
            decision = IntentDecision(
                intent="human", confidence=decision.confidence, reason="路由置信度过低"
            )
        return {
            "intent": decision.intent,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
        }

    def _knowledge_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        result = self.rag.answer(state["user_message"], state.get("history", []))
        return {
            "agent_name": "rag_agent",
            "response": result.answer,
            "sources": result.sources,
            "cache_hit": result.cache_hit,
        }

    def _tech_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        result = self.rag.answer(state["user_message"], state.get("history", []))
        response = (
            "我是技术支持 Agent。建议先按下面的知识库步骤逐项排查，并在每一步后验证现象：\n\n"
            f"{result.answer}\n\n"
            "如果完成以上步骤仍未解决，请提供设备型号、软件版本和报错截图，我会转人工继续处理。"
        )
        return {
            "agent_name": "tech_agent",
            "response": response,
            "sources": result.sources,
            "cache_hit": result.cache_hit,
        }

    def _order_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        match = re.search(r"\bORD[\-_ ]?\d+\b", state["user_message"], re.IGNORECASE)
        if not match:
            return {
                "agent_name": "order_agent",
                "response": "请提供订单号（例如 ORD001），我会从 MySQL 查询订单和物流状态。",
            }
        order_id = match.group(0).replace("-", "").replace("_", "").replace(" ", "").upper()
        with self.database.session_factory() as db:
            order = CommerceRepository(db).get_order(order_id)
            if order is None:
                response = f"没有查到订单 {order_id}。请核对订单号，或申请转人工客服。"
            else:
                logistics = (
                    f"{order.carrier} / {order.tracking_number}"
                    if order.tracking_number
                    else "尚未生成物流单号"
                )
                response = (
                    f"订单 {order.id} 查询结果：\n"
                    f"- 商品：{order.product_name}\n"
                    f"- 金额：¥{order.amount}\n"
                    f"- 状态：{order.status}\n"
                    f"- 物流：{logistics}\n"
                    f"- 预计送达：{order.estimated_delivery or '待更新'}"
                )
        return {"agent_name": "order_agent", "response": response}

    def _product_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        message = state["user_message"]
        budget_match = re.search(r"(?:预算|不超过|以内|低于)?\s*(\d{2,6})\s*(?:元|块)?", message)
        budget = Decimal(budget_match.group(1)) if budget_match else None
        keywords = ["智能手表", "无线耳机", "充电宝", "智能音箱", "手表", "耳机", "音箱"]
        keyword = next((item for item in keywords if item in message), None)
        normalized = {"手表": "智能手表", "耳机": "无线耳机", "音箱": "智能音箱"}.get(
            keyword, keyword
        )
        with self.database.session_factory() as db:
            products = CommerceRepository(db).search_products(
                keyword=normalized, budget=budget, limit=5
            )
        if not products:
            response = "目前没有找到符合条件的商品。可以放宽预算或更换品类，也可以转人工顾问。"
        else:
            rows = [
                f"- {item.name}：¥{item.price}，库存 {item.stock}，评分 {item.rating:.1f}；"
                f"特点：{'、'.join(item.features[:4])}"
                for item in products
            ]
            response = "根据 MySQL 中的实时商品数据，为你推荐：\n" + "\n".join(rows)
        return {"agent_name": "product_agent", "response": response}

    def _research_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        result = self.research.run(topic=state["user_message"], session_id=state["session_id"])
        return {
            "agent_name": "research_agent",
            "response": result["report"],
            "sources": result["sources"],
        }

    @staticmethod
    def _human_agent(state: CustomerServiceState) -> dict[str, Any]:
        return {
            "agent_name": "human_handoff_agent",
            "response": (
                "该请求已标记为人工处理。请留下方便联系的时间；正式生产环境中这里会创建工单，"
                "并把本次会话摘要交给人工客服。"
            ),
            "needs_human": True,
            "quality_score": 1.0,
        }

    def _quality_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        response = state.get("response", "")
        heuristic_score = min(0.95, 0.55 + min(len(response), 400) / 1000)
        needs_human = not response or "没有查到" in response or "无法" in response
        fallback = QualityDecision(
            score=round(heuristic_score, 3),
            needs_human=needs_human,
            reason="规则质检：检查空回复、失败信号、完整度和可执行性",
        )
        review = self.llm.structured(
            system="你是客服质检 Agent。评估相关性、准确性、完整性和风险；不确定时建议人工。",
            user=f"用户：{state['user_message']}\nAgent：{response}",
            schema=QualityDecision,
            fallback=fallback,
        )
        return {"quality_score": review.score, "needs_human": review.needs_human}

    @staticmethod
    def _human_review_notice(state: CustomerServiceState) -> dict[str, Any]:
        return {
            "response": (
                f"{state.get('response', '')}\n\n---\n"
                "质检 Agent 判断该回复需要人工复核，系统已保留上下文供客服接手。"
            ),
            "agent_name": f"{state.get('agent_name', 'agent')}+human_review",
        }

    def _persist(self, state: CustomerServiceState) -> dict[str, Any]:
        metadata = {
            "request_id": state["request_id"],
            "intent": state["intent"],
            "route_confidence": state["route_confidence"],
            "quality_score": state.get("quality_score", 0),
            "needs_human": state.get("needs_human", False),
        }
        with self.database.session_factory() as db:
            repository = ConversationRepository(db)
            repository.add(state["session_id"], "user", state["user_message"])
            repository.add(
                state["session_id"],
                "assistant",
                state["response"],
                agent_name=state["agent_name"],
                metadata=metadata,
            )
        history = [
            *state.get("history", []),
            {"role": "user", "content": state["user_message"]},
            {"role": "assistant", "content": state["response"]},
        ]
        self.cache.set_history(state["session_id"], history)
        return {"history": history[-20:]}

    @staticmethod
    def _rule_route(message: str) -> IntentDecision:
        lowered = message.lower()
        rules: list[tuple[Intent, list[str], str]] = [
            ("human", ["人工", "投诉", "经理", "差评", "维权"], "人工服务或投诉关键词"),
            ("research", ["研究报告", "调研", "深入研究", "研究一下"], "研究型长任务"),
            ("order_service", ["订单", "物流", "快递", "发货", "退货", "ord"], "订单或物流请求"),
            (
                "product_consult",
                ["推荐", "预算", "价格", "库存", "产品", "商品", "耳机", "手表"],
                "商品咨询",
            ),
            (
                "tech_support",
                ["故障", "连不上", "连接", "充电", "报错", "坏了", "更新"],
                "技术排障",
            ),
        ]
        for intent, keywords, reason in rules:
            if any(keyword in lowered for keyword in keywords):
                return IntentDecision(intent=intent, confidence=0.92, reason=reason)
        return IntentDecision(intent="knowledge", confidence=0.78, reason="默认进入知识库问答")

    def _traced(
        self,
        node_name: str,
        function: Callable[[CustomerServiceState], dict[str, Any]],
    ) -> Callable[[CustomerServiceState], dict[str, Any]]:
        @wraps(function)
        def wrapper(state: CustomerServiceState) -> dict[str, Any]:
            started = time.perf_counter()
            output = function(state)
            latency_ms = int((time.perf_counter() - started) * 1000)
            try:
                with self.database.session_factory() as db:
                    TraceRepository(db).add(
                        request_id=state["request_id"],
                        node_name=node_name,
                        input_summary=state.get("user_message", "")[:500],
                        output_summary=str(output)[:1000],
                        latency_ms=latency_ms,
                    )
            except Exception:
                # Observability must not make the business request fail.
                pass
            return output

        return wrapper
