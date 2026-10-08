import os
from dotenv import load_dotenv

load_dotenv()


def _get_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.getenv(name, str(default))
    try:
        parsed = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer") from error
    if not minimum <= parsed <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return parsed


def _get_list(name: str, default: str) -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in value.split(",") if item.strip()]


def _get_secret(name: str) -> str:
    value = os.getenv(name, "").strip()
    if value and len(value) < 32:
        raise ValueError(f"{name} must be at least 32 characters when configured")
    return value


class Settings:
    API_AUTH_TOKEN: str = _get_secret("API_AUTH_TOKEN")
    REDIS_URL: str = os.getenv("REDIS_URL", "").strip()
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "")
    RETRIEVAL_K: int = _get_int("RETRIEVAL_K", 30, 1, 100)
    RERANK_TOP_N: int = _get_int("RERANK_TOP_N", 12, 1, 50)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    MAX_QUESTION_LENGTH: int = _get_int("MAX_QUESTION_LENGTH", 4000, 100, 20000)
    MAX_SESSION_ID_LENGTH: int = _get_int("MAX_SESSION_ID_LENGTH", 128, 16, 512)
    LLM_TIMEOUT_SECONDS: int = _get_int("LLM_TIMEOUT_SECONDS", 90, 10, 600)
    LLM_MAX_RETRIES: int = _get_int("LLM_MAX_RETRIES", 2, 0, 5)
    RATE_LIMIT_REQUESTS: int = _get_int("RATE_LIMIT_REQUESTS", 20, 1, 1000)
    RATE_LIMIT_WINDOW_SECONDS: int = _get_int("RATE_LIMIT_WINDOW_SECONDS", 60, 1, 3600)
    MAX_SESSIONS: int = _get_int("MAX_SESSIONS", 1000, 10, 100000)
    MAX_HISTORY_MESSAGES: int = _get_int("MAX_HISTORY_MESSAGES", 12, 2, 100)
    SESSION_TTL_SECONDS: int = _get_int("SESSION_TTL_SECONDS", 86400, 300, 2592000)
    CORS_ORIGINS: list[str] = _get_list("CORS_ORIGINS", "http://127.0.0.1:8000")
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "RAG_Service")

settings = Settings()