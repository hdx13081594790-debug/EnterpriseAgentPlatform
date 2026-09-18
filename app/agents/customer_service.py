"""多智能体客服 Supervisor 工作流。

主数据流：用户消息 → 加载会话 → Supervisor 分类 → 专业 Worker → 质量检查 →
（可选人工复核提示）→ MySQL 会话/Trace + Redis 热会话 → API 响应。

LangGraph 节点只返回“状态增量”，框架把增量合并进 ``CustomerServiceState`` 后传给
下一节点。这样每个 Agent 不需要知道完整流程，只处理自己负责的字段。
"""

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
    """Supervisor 的结构化输出，限制路由只能落到已注册 Worker。"""

    intent: Intent
    confidence: float = Field(ge=0, le=1)
    reason: str


class QualityDecision(BaseModel):
    """质量 Agent 的结构化输出，用于决定正常结束还是人工复核。"""

    score: float = Field(ge=0, le=1)
    needs_human: bool
    reason: str


class CustomerServiceState(TypedDict, total=False):
    """客服图中所有节点共享的状态载体。

    数据沿图的边向前流动；每个节点读取需要的字段并返回少量更新字段。
    """

    # 请求标识：request_id 聚合 Trace，session_id 聚合多轮消息。
    request_id: str
    session_id: str
    user_message: str
    # 上下文流：Redis/MySQL → history → 路由与回答 Agent。
    history: list[dict[str, str]]
    # 路由流：Supervisor → intent/confidence/reason → 条件边。
    intent: Intent
    route_confidence: float
    route_reason: str
    agent_name: str
    # 回答流：专业 Worker → response/sources → Quality Agent → API。
    response: str
    sources: list[dict[str, Any]]
    quality_score: float
    needs_human: bool
    cache_hit: bool


class CustomerServiceWorkflow:
    """把请求路由到专业 Worker，并统一质检、追踪和持久化。"""

    def __init__(
        self,
        database: Database,
        cache: Cache,
        rag: RAGService,
        research: ResearchWorkflow,
        llm: LLMClient,
    ):
        """注入共享服务并只编译一次 LangGraph。"""

        self.database = database
        self.cache = cache
        self.rag = rag
        self.research = research
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """声明节点、固定边和条件边，返回可执行图。"""

        graph = StateGraph(CustomerServiceState)
        # 每个业务节点外包一层 _traced，节点执行完成后会写 AgentTrace。
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

        # 固定入口：所有请求都必须先恢复上下文，再让 Supervisor 分类。
        graph.add_edge(START, "load_context")
        graph.add_edge("load_context", "supervisor")
        # 数据流：state.intent → 条件边 → 且只进入一个专业 Worker。
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
        # 除直接人工转接外，所有机器回复都进入同一个质量门。
        for node in ["rag_agent", "tech_agent", "order_agent", "product_agent", "research_agent"]:
            graph.add_edge(node, "quality_agent")
        graph.add_edge("human_agent", "persist")
        # 数据流：needs_human=False → persist；True → 追加复核提示 → persist。
        graph.add_conditional_edges(
            "quality_agent",
            lambda state: "human" if state.get("needs_human") else "ok",
            {"human": "human_review_notice", "ok": "persist"},
        )
        graph.add_edge("human_review_notice", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def handle(self, *, session_id: str, message: str, request_id: str) -> dict[str, Any]:
        """构造初始状态、同步执行整张图，并裁剪为 API 所需字段。"""

        # 数据流：HTTP 参数 → 初始 CustomerServiceState → graph.invoke。
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
        # 图内部状态不全部暴露，API 只接收稳定、可解释的输出字段。
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
        """按 cache-aside 恢复最近会话。"""

        # 数据流：session_id → Redis；miss → MySQL → 回填 Redis → state.history。
        history = self.cache.get_history(state["session_id"])
        if history is None:
            with self.database.session_factory() as db:
                history = ConversationRepository(db).history(state["session_id"])
            self.cache.set_history(state["session_id"], history)
        return {"history": history}

    def _classify(self, state: CustomerServiceState) -> dict[str, Any]:
        """让 Supervisor 产生受 Schema 约束的意图和置信度。"""

        # 规则结果既是 mock 模式的正式结果，也是模型调用失败时的 fallback。
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
            # 低置信度不盲目调用工具，直接沿 human 条件边升级。
            decision = IntentDecision(
                intent="human", confidence=decision.confidence, reason="路由置信度过低"
            )
        return {
            "intent": decision.intent,
            "route_confidence": decision.confidence,
            "route_reason": decision.reason,
        }

    def _knowledge_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        """把通用知识问题交给 RAG，并透传来源与缓存状态。"""

        # 数据流：message + history → RAGService → answer/sources → state。
        result = self.rag.answer(state["user_message"], state.get("history", []))
        return {
            "agent_name": "rag_agent",
            "response": result.answer,
            "sources": result.sources,
            "cache_hit": result.cache_hit,
        }

    def _tech_agent(self, state: CustomerServiceState) -> dict[str, Any]:
        """复用 RAG 证据，并在外层增加可执行的技术排障规范。"""

        # 数据流：技术问题 → FAQ/RAG 证据 → 排障包装 → Quality Agent。
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
        """提取订单号并从 MySQL 精确查询，模型不参与订单事实生成。"""

        # 数据流：自然语言 → 正则提取 ORDxxx → CommerceRepository → Order。
        match = re.search(r"\bORD[\-_ ]?\d+\b", state["user_message"], re.IGNORECASE)
        if not match:
            # 缺少必需参数时不查库，返回明确的补充信息提示。
            return {
                "agent_name": "order_agent",
                "response": "请提供订单号（例如 ORD001），我会从 MySQL 查询订单和物流状态。",
            }
        order_id = match.group(0).replace("-", "").replace("_", "").replace(" ", "").upper()
        # 短 Session 只包围数据库读取，格式化回复不占用数据库连接。
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
        """从用户文本提取预算/品类，再查询结构化商品表。"""

        message = state["user_message"]
        # 数据流：自然语言 → 预算 Decimal + 受控关键词 → SQL filters。
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
        """把长研究请求委派给独立的研究子图。"""

        # 数据流：message/topic → ResearchWorkflow → Markdown report/sources → 客服 state。
        result = self.research.run(topic=state["user_message"], session_id=state["session_id"])
        return {
            "agent_name": "research_agent",
            "response": result["report"],
            "sources": result["sources"],
        }

    @staticmethod
    def _human_agent(state: CustomerServiceState) -> dict[str, Any]:
        """生成安全的人工转接结果；真实系统可在这里创建工单。"""

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
        """独立评估候选回复，避免生成 Agent 自己给自己放行。"""

        response = state.get("response", "")
        # 规则分数提供确定性基线；失败关键词会触发人工复核。
        heuristic_score = min(0.95, 0.55 + min(len(response), 400) / 1000)
        needs_human = not response or "没有查到" in response or "无法" in response
        fallback = QualityDecision(
            score=round(heuristic_score, 3),
            needs_human=needs_human,
            reason="规则质检：检查空回复、失败信号、完整度和可执行性",
        )
        # 数据流：问题 + 候选回复 → 质量 Schema → score/needs_human。
        review = self.llm.structured(
            system="你是客服质检 Agent。评估相关性、准确性、完整性和风险；不确定时建议人工。",
            user=f"用户：{state['user_message']}\nAgent：{response}",
            schema=QualityDecision,
            fallback=fallback,
        )
        return {"quality_score": review.score, "needs_human": review.needs_human}

    @staticmethod
    def _human_review_notice(state: CustomerServiceState) -> dict[str, Any]:
        """保留原回复并追加复核提示，而不是静默丢弃已有结果。"""

        return {
            "response": (
                f"{state.get('response', '')}\n\n---\n"
                "质检 Agent 判断该回复需要人工复核，系统已保留上下文供客服接手。"
            ),
            "agent_name": f"{state.get('agent_name', 'agent')}+human_review",
        }

    def _persist(self, state: CustomerServiceState) -> dict[str, Any]:
        """先持久化完整会话，再刷新 Redis 热历史。"""

        metadata = {
            "request_id": state["request_id"],
            "intent": state["intent"],
            "route_confidence": state["route_confidence"],
            "quality_score": state.get("quality_score", 0),
            "needs_human": state.get("needs_human", False),
        }
        # 写数据流：最终 state → 两条 ChatMessage → MySQL commit。
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
        # 缓存数据流：旧历史 + 本轮问答 → 最近 20 条 → Redis/session TTL。
        history = [
            *state.get("history", []),
            {"role": "user", "content": state["user_message"]},
            {"role": "assistant", "content": state["response"]},
        ]
        self.cache.set_history(state["session_id"], history)
        return {"history": history[-20:]}

    @staticmethod
    def _rule_route(message: str) -> IntentDecision:
        """按从高风险到通用问题的优先级执行确定性路由。"""

        lowered = message.lower()
        # 投诉/人工优先，避免同时出现商品词时被误路由到销售 Agent。
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
        """给节点增加耗时与摘要追踪，而不污染各节点业务代码。"""

        @wraps(function)
        def wrapper(state: CustomerServiceState) -> dict[str, Any]:
            """执行原节点，并在旁路中记录可观测信息。"""

            # 数据流：node input state → business function → output delta → AgentTrace。
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
                # 可观测性是旁路能力：追踪失败不能让已经成功的业务请求失败。
                pass
            return output

        return wrapper
