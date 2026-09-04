"""Pydantic request/response models for the HTTP layer."""
from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    tools_used: list[str]
    guardrails: list[str]
    latency_ms: int
