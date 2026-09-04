"""Liveness, service info, and Prometheus metrics endpoints."""
from fastapi import APIRouter, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from app.config import VECTOR_BACKEND

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/info")
def info():
    return {"backend": VECTOR_BACKEND}


@router.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
