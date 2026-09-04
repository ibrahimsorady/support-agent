"""Chat endpoints: one buffered reply, one server-sent-events stream."""
import json
import time

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import answer, answer_stream

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    t0 = time.perf_counter()
    reply, meta = answer(req.message)
    return ChatResponse(
        reply=reply,
        sources=meta.get("sources", []),
        tools_used=meta.get("tools_used", []),
        guardrails=meta.get("guardrails", []),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def sse():
        for event in answer_stream(req.message):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
