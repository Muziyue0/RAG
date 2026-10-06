from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="用户提问")
    session_id: str = Field(default="default_session", description="会话标识符")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
