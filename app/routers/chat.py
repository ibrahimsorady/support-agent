"""Chat endpoints: one buffered reply, one server-sent-events stream."""
import json
import time

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.agent import answer, answer_stream
from app.services.auth import AuthedUser, get_current_user

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, user: AuthedUser = Depends(get_current_user)):
    t0 = time.perf_counter()
    reply, meta = answer(
        req.message, token=user.token, user_id=user.user_id, conversation_id=req.conversation_id,
    )
    return ChatResponse(
        reply=reply,
        conversation_id=meta["conversation_id"],
        sources=meta.get("sources", []),
        tools_used=meta.get("tools_used", []),
        guardrails=meta.get("guardrails", []),
        latency_ms=int((time.perf_counter() - t0) * 1000),
    )


@router.post("/chat/stream")
def chat_stream(req: ChatRequest, user: AuthedUser = Depends(get_current_user)):
    def sse():
        for event in answer_stream(
            req.message, token=user.token, user_id=user.user_id, conversation_id=req.conversation_id,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")
