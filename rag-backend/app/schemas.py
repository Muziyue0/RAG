from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000, description="用户提问")
    session_id: str = Field(
        default="default_session", min_length=1, max_length=128, description="会话标识符"
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
