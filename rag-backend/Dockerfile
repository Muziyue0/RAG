FROM python:3.11-slim

# 设置工作目录与环境变量
WORKDIR /workspace
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    HF_HOME=/workspace/hf_cache \
    HF_HUB_DISABLE_TELEMETRY=1

# 安装编译所需的系统底层工具
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    gosu \
    && rm -rf /var/lib/apt/lists/*

# 先复制依赖列表并安装（利用 Docker 缓存层）
COPY requirements.runtime.txt .
RUN pip install --no-cache-dir --no-compile --default-timeout=180 --retries 5 \
        --index-url https://download.pytorch.org/whl/cpu \
        torch==2.6.0 \
    && pip install --no-cache-dir --no-compile --default-timeout=180 --retries 5 \
        -r requirements.runtime.txt \
    && pip check \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

# 复制业务代码
COPY app/ ./app
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home --shell /usr/sbin/nologin appuser \
    && chmod 755 /usr/local/bin/docker-entrypoint.sh \
    && chown -R appuser:appuser /workspace

# 暴露 FastAPI 端口
EXPOSE 8000

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]

# 启动 Uvicorn 服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]