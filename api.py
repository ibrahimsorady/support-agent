"""FastAPI HTTP layer for the agent service."""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel

from src.agent import answer
from src.config import VECTOR_BACKEND


class ChatRequest(BaseModel):
    message: str
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    tools_used: list[str]
    guardrails: list[str]
    latency_ms: int


app = FastAPI(title="Agent Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],   # Angular dev server
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/info")
def info():
    return {"backend": VECTOR_BACKEND}


@app.post("/chat", response_model=ChatResponse)
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


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
