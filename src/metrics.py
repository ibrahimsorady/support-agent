"""Observability: Prometheus metrics for the agent.

Three things worth watching in production, each for a different stakeholder:
  - latency         how long a reply takes            (user experience)
  - token cost      estimated $ per conversation      (unit economics / finance)
  - request outcome deflected / escalated / blocked   (business value)

Prometheus metric types used here:
  Counter   only ever increases  -> totals (requests, tokens, cost)
  Histogram a distribution        -> latency buckets (Grafana computes p95)

Nothing here starts a server on import. Call start_metrics_server() from a
long-running entry point (app.py or the traffic simulator) so Prometheus has a
/metrics endpoint to scrape.
"""
import time
from contextlib import contextmanager

from prometheus_client import Counter, Histogram, start_http_server

from src.config import (METRICS_PORT, PRICE_EMBED_PER_1M, PRICE_INPUT_PER_1M,
                        PRICE_OUTPUT_PER_1M)

REQUESTS = Counter("agent_requests_total", "Agent requests by outcome", ["outcome"])
LATENCY = Histogram("agent_request_latency_seconds", "End-to-end request latency (s)")
TOKENS = Counter("agent_tokens_total", "Tokens used", ["model", "kind"])
COST = Counter("agent_cost_usd_total", "Estimated cumulative cost (USD)")

_started = False


def start_metrics_server(port=None):
    """Expose GET /metrics on the given port (idempotent)."""
    global _started
    if not _started:
        start_http_server(port or METRICS_PORT)
        _started = True


def record_chat_usage(model, usage):
    """Record token counts + estimated cost from a Responses API call."""
    if usage is None:
        return
    inp = getattr(usage, "input_tokens", 0) or 0
    out = getattr(usage, "output_tokens", 0) or 0
    TOKENS.labels(model=model, kind="input").inc(inp)
    TOKENS.labels(model=model, kind="output").inc(out)
    COST.inc(inp / 1_000_000 * PRICE_INPUT_PER_1M + out / 1_000_000 * PRICE_OUTPUT_PER_1M)


def record_embed_usage(model, usage):
    """Record token counts + estimated cost from an embeddings call."""
    if usage is None:
        return
    tok = getattr(usage, "total_tokens", 0) or getattr(usage, "prompt_tokens", 0) or 0
    TOKENS.labels(model=model, kind="embedding").inc(tok)
    COST.inc(tok / 1_000_000 * PRICE_EMBED_PER_1M)


def record_request(outcome):
    """outcome is one of: deflected, escalated, blocked."""
    REQUESTS.labels(outcome=outcome).inc()


@contextmanager
def track_latency():
    start = time.perf_counter()
    try:
        yield
    finally:
        LATENCY.observe(time.perf_counter() - start)
