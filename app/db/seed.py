from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select

from app.db.database import Database
from app.db.models import Order, Product
from app.rag.service import RAGService

PRODUCTS = [
    {
        "name": "智能手表 Pro",
        "category": "智能穿戴",
        "price": Decimal("1299.00"),
        "features": ["心率监测", "GPS 定位", "50 米防水", "7 天续航"],
        "stock": 50,
        "rating": 4.8,
    },
    {
        "name": "无线耳机 Max",
        "category": "音频设备",
        "price": Decimal("899.00"),
        "features": ["主动降噪", "40 小时续航", "蓝牙 5.3", "通话降噪"],
        "stock": 120,
        "rating": 4.6,
    },
    {
        "name": "便携充电宝",
        "category": "数码配件",
        "price": Decimal("199.00"),
        "features": ["20000mAh", "快充", "双 USB 输出", "LED 电量显示"],
        "stock": 200,
        "rating": 4.5,
    },
    {
        "name": "智能音箱",
        "category": "智能家居",
        "price": Decimal("499.00"),
        "features": ["语音控制", "多房间音频", "智能家居联动", "Hi-Fi 音质"],
        "stock": 80,
        "rating": 4.7,
    },
]

ORDERS = [
    {
        "id": "ORD001",
        "product_name": "智能手表 Pro",
        "amount": Decimal("1299.00"),
        "status": "已发货",
        "carrier": "顺丰快递",
        "tracking_number": "SF1234567890",
        "estimated_delivery": "预计 2 天内",
    },
    {
        "id": "ORD002",
        "product_name": "无线耳机 Max",
        "amount": Decimal("899.00"),
        "status": "处理中",
        "carrier": None,
        "tracking_number": None,
        "estimated_delivery": "预计 3-5 天",
    },
    {
        "id": "ORD003",
        "product_name": "便携充电宝",
        "amount": Decimal("199.00"),
        "status": "已完成",
        "carrier": "圆通快递",
        "tracking_number": "YT9876543210",
        "estimated_delivery": "已签收",
    },
]

KNOWLEDGE_DOCUMENTS = [
    {
        "title": "RAG 检索增强生成原理与实践",
        "source": "seed://rag-principles",
        "metadata": {"topic": "rag", "type": "engineering_note"},
        "content": """RAG（检索增强生成）把外部知识检索与大语言模型生成结合起来。典型流程包括：文档清洗、分块、向量化、向量存储、召回、重排序、上下文组装、答案生成和引用校验。RAG 的主要价值是减少模型幻觉、让知识可以独立更新、保留来源可追溯性。

分块不能只看固定字数，还要考虑语义边界。块过大会带入噪声并浪费上下文，块过小会破坏完整语义。检索阶段可结合向量检索、关键词检索和元数据过滤；生成阶段应要求模型只基于证据作答，并显式输出引用。生产系统还应监控召回率、答案忠实度、延迟、Token 成本和缓存命中率。

本项目将原始文档和分块持久化到 MySQL，使用向量余弦相似度进行演示检索；Redis 缓存检索结果与最终答案。新增文档时主动失效 RAG 缓存，避免读取旧结果。规模达到百万级分块后，应迁移到 Milvus、Elasticsearch、OpenSearch 或 pgvector 等专用检索引擎。""",
    },
    {
        "title": "LangGraph 多智能体工作流",
        "source": "seed://langgraph-agents",
        "metadata": {"topic": "multi-agent", "type": "engineering_note"},
        "content": """LangGraph 用状态、节点和边表达有状态工作流。状态保存一次请求在节点间共享的数据；节点负责单一业务能力；普通边定义固定顺序；条件边根据当前状态决定下一步。图结构特别适合多步骤推理、人工介入、失败重试和多智能体协作。

多智能体系统不等于创建越多 Agent 越好。本项目采用 Supervisor 路由模式：总控 Agent 只判断意图，知识库、技术支持、订单、商品、研究和人工转接由不同 Worker 处理，最后统一经过质量检查。这样能够限制上下文、隔离工具权限、让每个节点独立测试，并通过 AgentTrace 表记录节点耗时和输入输出摘要。

生产实践中，需要给循环设置最大次数，为工具设置超时和幂等键，对写操作加入权限校验与人工确认，同时防范提示注入。结构化输出应使用 Pydantic Schema 校验，解析失败时使用安全降级结果。""",
    },
    {
        "title": "MySQL 与 Redis 在智能体平台中的职责",
        "source": "seed://mysql-redis",
        "metadata": {"topic": "infrastructure", "type": "architecture_note"},
        "content": """MySQL 是事实数据源，适合保存需要事务、一致性和长期审计的数据。本项目用 MySQL 保存知识文档与分块、商品、订单、会话消息、研究任务和 Agent 执行轨迹。SQLAlchemy 的 Session 按请求创建，事务完成后提交，异常时由上下文关闭连接。

Redis 用于短生命周期和高频访问数据，包括 RAG 检索结果、问答结果和最近会话历史。项目使用 cache-aside：先读缓存，未命中时查询 MySQL或执行计算，再写入 Redis并设置 TTL。知识文档变化时删除 rag 命名空间缓存。Redis 故障时自动降级到进程内 TTL 缓存，因此核心请求仍可完成，但多实例间缓存和会话不再共享。

缓存不能替代数据库，因为缓存可能过期、被淘汰或因故障丢失。需要重点处理缓存穿透、击穿、雪崩和数据一致性；常用手段包括空值短缓存、互斥锁、TTL 随机抖动、热点预热以及先更新数据库再删除缓存。""",
    },
    {
        "title": "智能设备客服 FAQ",
        "source": "seed://support-faq",
        "metadata": {"topic": "customer_support", "type": "faq"},
        "content": """蓝牙连接问题：先确认手机和设备蓝牙已开启，确保设备电量充足；然后删除旧配对记录，重启双方设备并重新配对；仍失败时检查 App 与固件版本，并记录设备型号和错误提示。

充电问题：优先使用原装或符合规格的充电器与线材，清理充电接口，尝试更换插座和线材。若设备明显发热、鼓包或有异味，应立即停止充电并联系人工售后，不要继续自行测试。

软件更新：在设备 App 的“设置—关于—检查更新”中执行。更新前保持电量高于 50%，不要中断网络或强制关机。更新失败时保存错误码并联系技术支持。

退货政策：演示商城支持签收后 7 天无理由退货，商品需保持完好并保留包装与购买凭证；质量问题在 30 天内可申请换货。真实业务应以订单页面当时展示的政策为准。""",
    },
]


def seed_all(database: Database, rag: RAGService) -> dict[str, int]:
    with database.session_factory() as db:
        if int(db.scalar(select(func.count(Product.id))) or 0) == 0:
            db.add_all(Product(**item) for item in PRODUCTS)
        if int(db.scalar(select(func.count(Order.id))) or 0) == 0:
            db.add_all(Order(**item) for item in ORDERS)
        db.commit()

    created_documents = 0
    for item in KNOWLEDGE_DOCUMENTS:
        result = rag.ingest(**item)
        created_documents += int(result["created"])
    return {
        "products": len(PRODUCTS),
        "orders": len(ORDERS),
        "new_documents": created_documents,
    }

