"""研究助手 LangGraph 子图。

数据流：研究主题 → Planner → 多查询 Retriever → Analyst → Writer → Reviewer →
分数条件边（重写 Writer 或结束）→ ``research_tasks`` 持久化。

所有中间产物保存在 ``ResearchState``，节点只返回状态增量；来源只来自 RAG，
写作与评审 Agent 不直接访问数据库，从结构上减少无依据内容。
"""

from __future__ import annotations

import json
from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from app.core.config import Settings
from app.db.database import Database
from app.db.repositories import ResearchRepository
from app.rag.service import RAGService, RetrievedChunk
from app.services.llm import LLMClient


class ResearchPlan(BaseModel):
    """Planner Agent 的结构化研究计划。"""

    title: str
    abstract: str
    sections: list[str]
    key_questions: list[str]
    methodology: str


class ReviewResult(BaseModel):
    """Reviewer Agent 的结构化评分与修改意见。"""

    score: float = Field(ge=0, le=1)
    strengths: list[str]
    improvements: list[str]


class ResearchState(TypedDict, total=False):
    """研究子图的共享状态；字段按节点顺序逐步补全。"""

    # 任务标识从 API/Repository 进入图，并在最终持久化时使用。
    task_id: str
    session_id: str
    topic: str
    # 内容流：plan → evidence → analysis → report → review。
    plan: dict[str, Any]
    retrieved_chunks: list[dict[str, Any]]
    sources: list[dict[str, Any]]
    analysis: str
    report: str
    quality_score: float
    review_feedback: list[str]
    # 控制流：quality_score + revision_count → 条件边。
    revision_count: int
    current_phase: str


class ResearchWorkflow:
    """规划、检索、分析、写作、评审组成的可循环研究图。"""

    def __init__(
        self,
        settings: Settings,
        database: Database,
        rag: RAGService,
        llm: LLMClient,
    ):
        """注入配置、事实数据库、RAG 和 LLM，并编译工作流。"""

        self.settings = settings
        self.database = database
        self.rag = rag
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """声明线性主链和 Reviewer 到 Writer 的有限回环。"""

        graph = StateGraph(ResearchState)
        graph.add_node("planner_agent", self._plan)
        graph.add_node("retriever_agent", self._retrieve)
        graph.add_node("analyst_agent", self._analyze)
        graph.add_node("writer_agent", self._write)
        graph.add_node("reviewer_agent", self._review)

        # 主数据流：topic → plan → evidence → analysis → report → review。
        graph.add_edge(START, "planner_agent")
        graph.add_edge("planner_agent", "retriever_agent")
        graph.add_edge("retriever_agent", "analyst_agent")
        graph.add_edge("analyst_agent", "writer_agent")
        graph.add_edge("writer_agent", "reviewer_agent")
        # 控制流：review state → _after_review → revise / complete。
        graph.add_conditional_edges(
            "reviewer_agent",
            self._after_review,
            {"revise": "writer_agent", "complete": END},
        )
        return graph.compile()

    def run(self, *, topic: str, session_id: str) -> dict[str, Any]:
        """创建任务、运行研究图，并在成功或失败路径更新 MySQL。"""

        # 数据流 1：HTTP topic/session → ResearchTask(status=running) → MySQL。
        with self.database.session_factory() as db:
            task = ResearchRepository(db).create(session_id, topic)

        # 数据流 2：task 元数据 → 图的初始共享状态。
        initial: ResearchState = {
            "task_id": task.id,
            "session_id": session_id,
            "topic": topic,
            "plan": {},
            "retrieved_chunks": [],
            "sources": [],
            "analysis": "",
            "report": "",
            "quality_score": 0.0,
            "review_feedback": [],
            "revision_count": 0,
            "current_phase": "planning",
        }
        try:
            # 节点按图执行并不断合并状态增量，result 是最终完整状态。
            result = self.graph.invoke(initial)
            # 数据流 3（成功）：final report/sources/score → ResearchTask(completed)。
            with self.database.session_factory() as db:
                ResearchRepository(db).complete(
                    task.id,
                    report=result["report"],
                    sources=result.get("sources", []),
                    quality_score=result["quality_score"],
                )
            return {
                "task_id": task.id,
                "status": "completed",
                "topic": topic,
                "report": result["report"],
                "sources": result.get("sources", []),
                "quality_score": result["quality_score"],
                "revision_count": result.get("revision_count", 0),
            }
        except Exception as exc:
            # 数据流 3（失败）：异常摘要 → ResearchTask(failed)，随后继续抛给 API。
            with self.database.session_factory() as db:
                ResearchRepository(db).fail(task.id, str(exc))
            raise

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        """读取已持久化任务并转换成不暴露 ORM 的字典。"""

        with self.database.session_factory() as db:
            row = ResearchRepository(db).get(task_id)
            if row is None:
                return None
            return {
                "task_id": row.id,
                "session_id": row.session_id,
                "topic": row.topic,
                "status": row.status,
                "report": row.report,
                "sources": row.sources,
                "quality_score": row.quality_score,
            }

    def _plan(self, state: ResearchState) -> dict[str, Any]:
        """根据主题产出大纲、问题和方法论。"""

        topic = state["topic"]
        # fallback 既服务离线 mock，也保证真实模型失败时图仍有合法计划。
        fallback = ResearchPlan(
            title=f"{topic}：基于企业知识库的研究报告",
            abstract=f"本报告围绕“{topic}”进行资料检索、证据分析与结论归纳。",
            sections=["问题背景", "核心机制", "实践方案", "风险与改进", "结论"],
            key_questions=[
                f"{topic}的核心概念是什么？",
                f"{topic}如何在工程中落地？",
                f"{topic}有哪些风险与优化方向？",
            ],
            methodology="采用本地知识库检索、来源交叉比对和基于证据的归纳分析。",
        )
        # 数据流：state.topic → LLM/Pydantic ResearchPlan → state.plan。
        plan = self.llm.structured(
            system="你是研究规划 Agent。输出可执行的研究大纲，避免空泛章节。",
            user=f"研究主题：{topic}",
            schema=ResearchPlan,
            fallback=fallback,
        )
        return {"plan": plan.model_dump(), "current_phase": "retrieval"}

    def _retrieve(self, state: ResearchState) -> dict[str, Any]:
        """对主题和各研究问题分别召回，再按 chunk_id 去重并重排。"""

        plan = state["plan"]
        # 多查询能扩大召回面；单一主题词可能遗漏大纲中的子问题证据。
        queries = [state["topic"], *plan.get("key_questions", [])]
        unique: dict[str, RetrievedChunk] = {}
        # 数据流：queries → RAG.retrieve(each) → chunk_id 去重 → score 降序。
        for query in queries:
            for chunk in self.rag.retrieve(query, top_k=3):
                existing = unique.get(chunk.chunk_id)
                if existing is None or chunk.score > existing.score:
                    unique[chunk.chunk_id] = chunk
        selected = sorted(unique.values(), key=lambda item: item.score, reverse=True)[:8]
        # ORM/内部对象转为纯字典，作为后续 Agent 的受控证据上下文。
        chunk_data = [
            {
                "chunk_id": chunk.chunk_id,
                "title": chunk.title,
                "source": chunk.source,
                "content": chunk.content,
                "score": round(chunk.score, 4),
            }
            for chunk in selected
        ]
        sources = [
            {
                "index": index,
                "title": item["title"],
                "source": item["source"],
                "score": item["score"],
            }
            for index, item in enumerate(chunk_data, start=1)
        ]
        return {
            "retrieved_chunks": chunk_data,
            "sources": sources,
            "current_phase": "analysis",
        }

    def _analyze(self, state: ResearchState) -> dict[str, Any]:
        """把原始证据转成围绕研究问题的有引用分析。"""

        # 数据流：retrieved_chunks → 编号 evidence → LLM/fallback → state.analysis。
        evidence = self._format_evidence(state.get("retrieved_chunks", []))
        fallback = self._fallback_analysis(state)
        analysis = self.llm.text(
            system=(
                "你是证据分析 Agent。区分事实、推断和资料缺口；每个关键判断必须带 [n] 引用。"
            ),
            user=(
                f"主题：{state['topic']}\n研究问题："
                f"{json.dumps(state['plan'].get('key_questions', []), ensure_ascii=False)}"
                f"\n\n证据：\n{evidence}"
            ),
            fallback=fallback,
        )
        return {"analysis": analysis, "current_phase": "writing"}

    def _write(self, state: ResearchState) -> dict[str, Any]:
        """把计划、分析和上一轮意见整合为 Markdown 报告。"""

        plan = state["plan"]
        feedback = state.get("review_feedback", [])
        # 回环时 feedback 已更新，Writer 会基于同一证据定向修改，而不是重新检索。
        fallback = self._fallback_report(state)
        report = self.llm.text(
            system=(
                "你是研究报告写作 Agent。输出 Markdown；保留 [n] 引用；不得引入证据中不存在的数字。"
            ),
            user=(
                f"大纲：{json.dumps(plan, ensure_ascii=False)}\n\n"
                f"分析：\n{state.get('analysis', '')}\n\n"
                f"上轮评审意见：{json.dumps(feedback, ensure_ascii=False)}"
            ),
            fallback=fallback,
        )
        return {"report": report, "current_phase": "review"}

    def _review(self, state: ResearchState) -> dict[str, Any]:
        """评分并输出可执行修改意见，同时递增评审轮次。"""

        source_count = len(state.get("sources", []))
        report_length = len(state.get("report", ""))
        # mock 模式的启发式分数综合来源数与报告长度，保证测试可重复。
        heuristic_score = min(
            0.9,
            0.55 + min(source_count, 5) * 0.04 + min(report_length, 1200) / 12000,
        )
        fallback = ReviewResult(
            score=round(heuristic_score, 3),
            strengths=["报告结构完整", "关键结论保留了来源编号"],
            improvements=([] if source_count >= 2 else ["补充更多独立来源并进行交叉验证"]),
        )
        # 数据流：report + source_count → ReviewResult → quality/feedback/revision_count。
        review = self.llm.structured(
            system="你是质量评审 Agent。按证据覆盖、逻辑、结构、引用四个维度评分。",
            user=f"来源数：{source_count}\n报告：\n{state.get('report', '')[:5000]}",
            schema=ReviewResult,
            fallback=fallback,
        )
        return {
            "quality_score": review.score,
            "review_feedback": review.improvements,
            "revision_count": state.get("revision_count", 0) + 1,
            "current_phase": "reviewed",
        }

    def _after_review(self, state: ResearchState) -> Literal["revise", "complete"]:
        """质量未达标且仍有预算时回到 Writer，否则结束。"""

        # 双重保护：质量阈值控制效果，最大重写次数防止 Agent 无限循环。
        if (
            state.get("quality_score", 0) < self.settings.research_quality_threshold
            and state.get("revision_count", 0) <= self.settings.research_max_revisions
        ):
            return "revise"
        return "complete"

    def _fallback_analysis(self, state: ResearchState) -> str:
        """模型不可用时直接整理检索片段，并保留 [n] 引用。"""

        chunks = state.get("retrieved_chunks", [])
        if not chunks:
            return "知识库中没有与该主题相关的证据，无法形成可靠结论。"
        lines = []
        for index, chunk in enumerate(chunks[:5], start=1):
            excerpt = " ".join(chunk["content"].split())[:220]
            lines.append(f"- 证据 {index}：{excerpt} [{index}]")
        return "资料归纳结果：\n" + "\n".join(lines)

    def _fallback_report(self, state: ResearchState) -> str:
        """用现有计划、分析和来源生成结构完整的离线 Markdown 报告。"""

        plan = state["plan"]
        sources = state.get("sources", [])
        references = "\n".join(
            f"[{item['index']}] {item['title']}，{item['source']}"
            for item in sources
        ) or "无（知识库资料不足）"
        return f"""# {plan.get('title', state['topic'])}

## 摘要

{plan.get('abstract', '')}

## 研究方法

{plan.get('methodology', '')}

## 关键发现

{state.get('analysis', '暂无足够证据。')}

## 工程实践建议

1. 先建立可追溯的数据链路，再扩展 Agent 数量。
2. 将结构化业务数据与非结构化知识检索分开治理。
3. 为外部依赖设置超时、重试、降级和可观测指标。
4. 对高风险动作增加人工确认与权限控制。

## 局限性

本报告只使用当前知识库中的资料。结论应结合线上指标、真实业务数据及更多独立来源复核。

## 参考资料

{references}
"""

    @staticmethod
    def _format_evidence(chunks: list[dict[str, Any]]) -> str:
        """把检索块编号为模型可引用的只读证据文本。"""

        return "\n\n".join(
            f"[{index}] {item['title']} ({item['source']})\n{item['content']}"
            for index, item in enumerate(chunks, start=1)
        ) or "无"
