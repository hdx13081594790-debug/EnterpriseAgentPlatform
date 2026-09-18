# Interview Agent Platform

一个适合面试演示的后端项目：把原来的 RAG 问答、智能客服多 Agent、研究助手三个单文件示例，重构为一个可运行、可测试、可扩展的工程。

核心技术栈：FastAPI、LangChain、LangGraph、SQLAlchemy 2、MySQL 8、Redis、Pydantic 2、Pytest。

> 默认使用 `mock + SQLite + 内存缓存`，没有模型密钥、MySQL、Redis 也能演示完整流程。切换环境变量后，同一套代码可使用 Groq/OpenAI、MySQL 和 Redis。

## 项目亮点

- RAG：文档导入、语义分块、向量化、余弦召回、来源引用、置信度、答案缓存。
- 多智能体：Supervisor 根据意图路由到 6 类 Worker，统一经过质检并支持人工升级。
- 研究助手：规划、检索、分析、写作、评审五个 Agent；低质量报告会有限次数重写。
- MySQL：保存知识文档与分块、订单、商品、会话、研究任务和 Agent 执行轨迹。
- Redis：缓存 RAG 召回、最终回答和最近会话；Redis 故障时自动降级，不影响主流程。
- 工程化：配置校验、依赖注入、分层结构、健康检查、Docker Compose、幂等导入、测试与覆盖率。
- 可观测性：每个客服图节点的耗时、输入摘要和输出摘要写入 `agent_traces`。
- 离线可演示：本地哈希向量和确定性规则保证面试现场不依赖外网。

更详细的设计取舍、讲解话术和面试题见 [面试讲解手册](docs/INTERVIEW_GUIDE.md)。

## 系统架构

```mermaid
flowchart LR
    Client[客户端 / Swagger] --> API[FastAPI]
    API --> Supervisor[LangGraph Supervisor]
    Supervisor --> RAG[RAG Agent]
    Supervisor --> Tech[技术支持 Agent]
    Supervisor --> Order[订单 Agent]
    Supervisor --> Product[商品 Agent]
    Supervisor --> Research[研究工作流]
    Supervisor --> Human[人工转接]
    RAG --> MySQL[(MySQL)]
    Tech --> RAG
    Order --> MySQL
    Product --> MySQL
    Research --> RAG
    Supervisor --> Redis[(Redis)]
    RAG --> Redis
    Supervisor --> Quality[质量检查 Agent]
    Quality --> MySQL
```

职责边界：

- MySQL 是事实数据源，Redis 只是可丢失、可重建的加速层。
- Supervisor 只做路由，不直接处理领域业务。
- Worker 只拿自己需要的数据和工具，降低提示注入与越权风险。
- RAG 对非结构化知识做检索；订单和商品使用 SQL 精确查询，避免让模型“猜数据”。

## 目录结构

```text
.
├── app/
│   ├── agents/
│   │   ├── customer_service.py   # Supervisor + 专业客服 Agent + 质检
│   │   └── research.py           # 研究助手 LangGraph
│   ├── api/
│   │   ├── routes.py             # REST API
│   │   └── schemas.py            # Pydantic 请求/响应模型
│   ├── core/config.py            # 环境变量与参数校验
│   ├── db/
│   │   ├── database.py           # Engine 与 Session 工厂
│   │   ├── models.py             # 7 张业务表
│   │   ├── repositories.py       # 数据访问层
│   │   └── seed.py               # 面试演示数据
│   ├── infrastructure/cache.py   # Redis + 内存降级
│   ├── rag/
│   │   ├── embeddings.py         # 本地/OpenAI Embedding 适配
│   │   └── service.py            # 导入、召回、生成、缓存
│   ├── services/
│   │   ├── container.py          # 依赖组装入口
│   │   └── llm.py                # 模型供应商适配与安全降级
│   └── main.py                   # FastAPI 生命周期
├── docs/INTERVIEW_GUIDE.md
├── scripts/seed.py
├── tests/
├── docker-compose.yml
├── Dockerfile
└── pyproject.toml
```

## 快速启动：零外部依赖模式

要求 Python 3.11～3.13。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload
```

默认配置会使用：

- `LLM_PROVIDER=mock`
- `EMBEDDING_PROVIDER=local_hash`
- SQLite：`data/interview_agent.db`
- 进程内 TTL 缓存

启动后访问：

- Swagger：<http://127.0.0.1:8000/docs>
- 健康检查：<http://127.0.0.1:8000/health>

## 使用 MySQL 与 Redis

先复制配置文件：

```powershell
Copy-Item .env.example .env
docker compose up --build
```

Compose 会启动：

- API：`127.0.0.1:8000`
- MySQL 8.4：`127.0.0.1:3306`
- Redis 7.4：`127.0.0.1:6379`

容器中的 API 会覆盖 `.env` 里的本地 SQLite 配置，自动连接 `mysql` 与 `redis` 服务。应用启动时自动建表，并以幂等方式写入面试演示数据。

如果本机已单独安装 MySQL/Redis，可在 `.env` 中设置：

```dotenv
DATABASE_URL=mysql+pymysql://agent:agent_pass@127.0.0.1:3306/agent_platform?charset=utf8mb4
REDIS_URL=redis://127.0.0.1:6379/0
```

## 接入真实大模型

Groq 示例：

```dotenv
LLM_PROVIDER=groq
LLM_MODEL=openai/gpt-oss-120b
GROQ_API_KEY=你的密钥
```

OpenAI 示例（需安装可选依赖 `pip install -e ".[openai]"`）：

```dotenv
LLM_PROVIDER=openai
LLM_MODEL=gpt-5-mini
OPENAI_API_KEY=你的密钥
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-3-small
```

真实模型调用失败时，LLM 适配层会记录异常并使用安全 fallback；不会让整个 API 因一次模型错误而退出。

## API 演示

### 1. 知识库问答

```powershell
$body = @{
  session_id = "demo-rag"
  message = "RAG 的典型流程和优势是什么？"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/chat `
  -Method Post -ContentType "application/json" -Body $body
```

响应包括命中的 Agent、来源、路由置信度、质检分数和缓存命中状态。

### 2. 订单 Agent 查询 MySQL

```powershell
$body = @{session_id="demo-order"; message="查一下 ORD001 的物流"} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/chat `
  -Method Post -ContentType "application/json" -Body $body
```

### 3. 商品 Agent

```powershell
$body = @{session_id="demo-product"; message="预算 1000 元，推荐一款耳机"} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/chat `
  -Method Post -ContentType "application/json" -Body $body
```

### 4. 导入知识文档

```powershell
$body = @{
  title = "退款业务规则"
  source = "internal://refund-policy"
  content = "退款申请需要订单号。未发货订单可直接申请退款；已发货订单需先拒收或退回。"
  metadata = @{department="售后"}
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/knowledge/documents `
  -Method Post -ContentType "application/json" -Body $body
```

相同内容再次导入会根据 SHA-256 内容哈希去重。新增成功后会失效 `rag:*` 缓存。

### 5. 生成研究报告

```powershell
$body = @{
  session_id = "demo-research"
  topic = "RAG 与多智能体系统的工程实践"
} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/api/v1/research `
  -Method Post -ContentType "application/json" -Body $body
```

也可以直接在聊天接口输入“请深入研究 RAG 与多智能体架构”，Supervisor 会路由到研究 Agent。

## 路由规则

| 用户请求 | Worker | 数据来源 |
|---|---|---|
| 通用知识问题 | `rag_agent` | MySQL 知识分块 + Redis 缓存 |
| 设备故障与排查 | `tech_agent` | RAG + 技术支持提示 |
| 订单与物流 | `order_agent` | MySQL `orders` |
| 商品、预算、库存 | `product_agent` | MySQL `products` |
| 调研与研究报告 | `research_agent` | 多查询 RAG + 研究图 |
| 投诉、明确要求人工、低置信度 | `human_handoff_agent` | 会话上下文 |

生产模式中使用模型的结构化输出路由；离线模式使用确定性规则。低于置信度阈值的路由不会盲猜，而是转人工。

## 数据表

| 表 | 用途 |
|---|---|
| `knowledge_documents` | 原始文档、来源、元数据、内容哈希 |
| `knowledge_chunks` | 文本块与向量 |
| `products` | 商品、价格、库存、特征与评分 |
| `orders` | 订单和物流状态 |
| `chat_messages` | 长期会话记录 |
| `research_tasks` | 研究报告、来源、状态、质量分 |
| `agent_traces` | 每个 LangGraph 节点的耗时与摘要 |

## Redis Key 设计

```text
agent-platform:rag:search:<sha256>   # 检索结果，默认 15 分钟
agent-platform:rag:answer:<sha256>   # 最终回答，默认 15 分钟
agent-platform:session:<session_id>  # 最近 20 条消息，默认 24 小时
```

缓存内容不承载唯一数据。Redis 丢失后可从 MySQL 重新构建，因此系统采用“可用性优先”的降级策略。

## 测试与代码检查

```powershell
python -m pytest --cov=app --cov-report=term-missing
python -m ruff check app tests scripts
```

当前测试覆盖：健康检查、订单路由、RAG 来源与缓存命中、商品查询、文档幂等导入、研究图和本地向量确定性。

## 已知边界与生产化方向

本项目故意保持为一个可讲清楚的面试项目，而不是假装已经解决所有生产问题：

- 当前向量保存在 MySQL JSON 字段并由应用层计算余弦相似度，适合中小型演示；大规模场景应接入专用向量数据库或搜索引擎。
- 研究接口目前同步执行；长任务应改为消息队列 + Worker，并增加进度查询或 SSE。
- 自动建表便于演示；正式环境应使用 Alembic 管理迁移。
- 生产环境还要增加认证授权、租户隔离、限流、敏感信息脱敏、提示注入检测和完整监控告警。
- 对退款、发货、删除等有副作用工具，应加入幂等键、权限校验和 human-in-the-loop 确认。

这些边界不是缺陷隐瞒，而是面试中可以主动说明的架构演进路线。

