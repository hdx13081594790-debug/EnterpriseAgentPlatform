from __future__ import annotations


def test_health_exposes_dependencies_and_seeded_knowledge(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "up"
    assert body["cache"]["backend"] == "memory"
    assert body["knowledge"]["documents"] >= 4


def test_supervisor_routes_order_request_to_mysql_agent(client):
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "order-session", "message": "请查询订单 ORD001 的物流"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "order_service"
    assert body["agent"] == "order_agent"
    assert "已发货" in body["answer"]
    assert "SF1234567890" in body["answer"]


def test_rag_answer_has_sources_and_second_session_hits_cache(client):
    first = client.post(
        "/api/v1/chat",
        json={"session_id": "rag-session-a", "message": "RAG 的典型流程是什么？"},
    )
    second = client.post(
        "/api/v1/chat",
        json={"session_id": "rag-session-b", "message": "RAG 的典型流程是什么？"},
    )
    assert first.status_code == 200
    assert first.json()["agent"] == "rag_agent"
    assert first.json()["sources"]
    assert second.json()["cache_hit"] is True


def test_product_agent_reads_structured_product_data(client):
    response = client.post(
        "/api/v1/chat",
        json={"session_id": "product-session", "message": "预算 1000 元，推荐一款耳机"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["intent"] == "product_consult"
    assert "无线耳机 Max" in body["answer"]


def test_ingestion_is_idempotent_and_invalidates_rag_cache(client):
    document = {
        "title": "幂等测试文档",
        "source": "test://idempotency",
        "content": "这是用于验证知识文档内容哈希去重的测试文本。" * 4,
        "metadata": {"owner": "pytest"},
    }
    first = client.post("/api/v1/knowledge/documents", json=document)
    second = client.post("/api/v1/knowledge/documents", json=document)
    assert first.status_code == 201
    assert first.json()["created"] is True
    assert second.json()["created"] is False
    assert first.json()["document_id"] == second.json()["document_id"]


def test_research_graph_returns_persisted_markdown_report(client):
    response = client.post(
        "/api/v1/research",
        json={"session_id": "research-session", "topic": "RAG 与多智能体系统的工程实践"},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "completed"
    assert body["report"].startswith("# ")
    assert body["sources"]

    stored = client.get(f"/api/v1/research/{body['task_id']}")
    assert stored.status_code == 200
    assert stored.json()["report"] == body["report"]

