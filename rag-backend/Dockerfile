FROM python:3.11-slim

# 设置工作目录与环境变量
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/hf_cache

# 安装编译所需的系统底层工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖列表并安装（利用 Docker 缓存层）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制业务代码
COPY app/ ./app

# 暴露 FastAPI 端口
EXPOSE 8000

# 启动 Uvicorn 服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]