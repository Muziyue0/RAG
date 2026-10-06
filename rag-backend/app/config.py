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


class Settings:
    DEEPSEEK_API_KEY: str = os.getenv("DEEPSEEK_API_KEY", "")
    PINECONE_API_KEY: str = os.getenv("PINECONE_API_KEY", "")
    PINECONE_INDEX_NAME: str = os.getenv("PINECONE_INDEX_NAME", "")
    RETRIEVAL_K: int = _get_int("RETRIEVAL_K", 30, 1, 100)
    RERANK_TOP_N: int = _get_int("RERANK_TOP_N", 12, 1, 50)
    RERANK_MODEL: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
    MAX_QUESTION_LENGTH: int = _get_int("MAX_QUESTION_LENGTH", 4000, 100, 20000)
    MAX_SESSION_ID_LENGTH: int = _get_int("MAX_SESSION_ID_LENGTH", 128, 16, 512)
    LANGSMITH_TRACING: str = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_ENDPOINT: str = os.getenv("LANGSMITH_ENDPOINT", "")
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "RAG_Service")

settings = Settings()