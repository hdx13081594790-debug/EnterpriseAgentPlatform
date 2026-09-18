# 构建数据流：项目源码 → Python 镜像 → 安装依赖 → Uvicorn ASGI 服务。
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先复制依赖描述，源码未变化时可复用 Docker 依赖层缓存。
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install .

EXPOSE 8000

# 运行数据流：容器 8000 端口 → FastAPI app → Agent/RAG → MySQL/Redis。
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
