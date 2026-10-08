import logging
import secrets
import time
from collections import defaultdict, deque
from functools import lru_cache
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis
from fastapi.responses import HTMLResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from app.schemas import ChatRequest, ChatResponse
from app.rag_service import rag_service
from app.config import settings


logger = logging.getLogger(__name__)

app = FastAPI(
    title="Domain Manual RAG Service",
    description="基于重排与滑动窗口记忆的高精度检索后端服务",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
)


class RequestProtectionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app):
        super().__init__(app)
        self.requests_by_client = defaultdict(deque)
        self.redis = Redis.from_url(settings.REDIS_URL) if settings.REDIS_URL else None

    async def _allow_request(self, client_id: str) -> bool:
        if self.redis is None:
            return await self._allow_request_without_redis(client_id)

        key = f"rag:rate-limit:{client_id}"
        try:
            count = await self.redis.incr(key)
            if count == 1:
                await self.redis.expire(key, settings.RATE_LIMIT_WINDOW_SECONDS)
            return count <= settings.RATE_LIMIT_REQUESTS
        except Exception:
            logger.exception("Redis rate limiter unavailable; using local fallback")
            return await self._allow_request_without_redis(client_id)

    async def _allow_request_without_redis(self, client_id: str) -> bool:
        now = time.monotonic()
        requests = self.requests_by_client[client_id]
        while requests and now - requests[0] >= settings.RATE_LIMIT_WINDOW_SECONDS:
            requests.popleft()
        if len(requests) >= settings.RATE_LIMIT_REQUESTS:
            return False
        requests.append(now)
        return True

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        request.state.request_id = request_id
        started_at = time.perf_counter()

        if request.url.path == "/api/v1/chat":
            client_id = request.client.host if request.client else "unknown"
            if not await self._allow_request(client_id):
                return JSONResponse(
                    status_code=429,
                    content={"detail": "请求过于频繁，请稍后重试。"},
                    headers={"Retry-After": str(settings.RATE_LIMIT_WINDOW_SECONDS)},
                )

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("Unhandled request error request_id=%s", request_id)
            raise

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        logger.info(
            "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response


app.add_middleware(RequestProtectionMiddleware)

CHAT_TEMPLATE_PATH = Path(__file__).parent / "templates" / "chat.html"


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/ready")
def readiness_check():
    if rag_service is None:
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    if rag_service.redis_client is not None:
        try:
            rag_service.redis_client.ping()
        except Exception:
            logger.exception("Redis readiness check failed")
            raise HTTPException(status_code=503, detail="依赖服务尚未就绪")
    return {"status": "ready"}


@app.get("/", response_class=HTMLResponse)
def chat_page():
    return HTMLResponse(_load_chat_template())


@lru_cache(maxsize=1)
def _load_chat_template() -> str:
    return CHAT_TEMPLATE_PATH.read_text(encoding="utf-8")


@app.post("/api/v1/chat", response_model=ChatResponse)
def chat_endpoint(request: Request, chat_request: ChatRequest):
    if settings.API_AUTH_TOKEN:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(
            token, settings.API_AUTH_TOKEN
        ):
            raise HTTPException(status_code=401, detail="需要有效的 API Token。")
    request_data = chat_request
    if len(request_data.question.strip()) > settings.MAX_QUESTION_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"问题长度不能超过 {settings.MAX_QUESTION_LENGTH} 个字符。",
        )
    if len(request_data.session_id) > settings.MAX_SESSION_ID_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"会话标识长度不能超过 {settings.MAX_SESSION_ID_LENGTH} 个字符。",
        )
    try:
        answer = rag_service.execute_query(
            question=request_data.question, session_id=request_data.session_id
        )
        return ChatResponse(session_id=request_data.session_id, answer=answer)
    except Exception:
        logger.exception("Chat request failed")
        raise HTTPException(
            status_code=500, detail="服务暂时不可用，请稍后重试。"
        )
