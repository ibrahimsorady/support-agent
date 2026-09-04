"""RAG + tool-calling agent, wrapped in guardrails and instrumented for metrics.

Flow per request:
  1. INPUT guardrail  -> block unsafe requests before any model call
  2. RAG retrieval    -> pull relevant KB snippets
  3. Agent loop       -> model answers, or requests tools we run and feed back
  4. OUTPUT guardrail -> redact leaks / catch ungrounded answers before replying

Every request is timed, its token usage recorded, and its outcome classified
(deflected / escalated / blocked) so Prometheus + Grafana can show latency,
cost, and deflection rate. See app/observability/metrics.py.
"""
import time

from openai import OpenAI

from app.config import CHAT_MODEL
from app.observability import metrics
from app.services.guardrails import check_input, check_output
from app.services.retriever import retrieve
from app.tools.tools import TOOL_SCHEMAS, run_tool

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

SYSTEM = (
    "You are a customer-support agent for a telecom company.\n"
    "You have two sources of help:\n"
    "1. Context snippets from the knowledge base - use these to answer questions "
    "about policies, plans, roaming, billing rules, SIM/eSIM, and troubleshooting.\n"
    "2. Tools - use these for account-specific actions: look up an order, check an "
    "account's status, or create a support ticket for human follow-up.\n"
    "Rules: answer policy questions ONLY from the provided context; if it isn't "
    "there, say so and offer to escalate. For account-specific requests, call the "
    "appropriate tool rather than guessing. Never invent order details, balances, "
    "or account data - always use a tool to get them. Keep replies short and friendly."
)


def answer(query, max_turns=5):
    """Return (reply_text, meta) where meta = {sources, tools_used, guardrails}."""
    with metrics.track_latency():
        guardrails = []

        # 1. INPUT guardrail -- runs before any retrieval or model call.
        allowed, reason, safe_msg = check_input(query)
        if not allowed:
            guardrails.append(f"input:{reason}")
            metrics.record_request("blocked")
            return safe_msg, {"sources": [], "tools_used": [], "guardrails": guardrails}

        # 2. RAG retrieval.
        hits = retrieve(query)
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

        input_list = [
            {"role": "user",
             "content": f"Context snippets:\n{context}\n\nCustomer question: {query}"}
        ]
        tools_used, tool_outputs = [], []

        # 3. Agent loop.
        for _ in range(max_turns):
            resp = _get_client().responses.create(
                model=CHAT_MODEL,
                instructions=SYSTEM,
                tools=TOOL_SCHEMAS,
                input=input_list,
            )
            metrics.record_chat_usage(CHAT_MODEL, getattr(resp, "usage", None))
            calls = [item for item in resp.output if item.type == "function_call"]

            if not calls:
                # 4. OUTPUT guardrail -- before the reply reaches the user.
                reply, fired = check_output(resp.output_text, context, tool_outputs)
                guardrails += fired
                # Deflected unless the output guardrail forced an escalation.
                outcome = "escalated" if "output:ungrounded" in fired else "deflected"
                metrics.record_request(outcome)
                return reply, {
                    "sources": [h["source"] for h in hits],
                    "tools_used": tools_used,
                    "guardrails": guardrails,
                }

            input_list += resp.output
            for call in calls:
                tools_used.append(call.name)
                result = run_tool(call.name, call.arguments)
                tool_outputs.append(result)
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                })

        # Safety valve: too many tool round-trips without a final answer.
        metrics.record_request("escalated")
        return (
            "I'm having trouble completing that right now - let me escalate you to a human agent.",
            {"sources": [h["source"] for h in hits], "tools_used": tools_used, "guardrails": guardrails},
        )


def answer_stream(query, max_turns=5):
    """Streaming twin of answer(): same flow and outcome, but yields events as
    they happen instead of returning once at the end.

    Yields dicts of one of three shapes:
      {"type": "status", "text": ...}          - a tool-calling turn is running
      {"type": "token",  "text": ...}          - a chunk of the final answer text
      {"type": "done",   "meta": {...}}        - terminal event, same meta as answer()

    Tool-calling turns are NOT streamed as text (only a generic status event is
    emitted); only the final answering turn streams token-by-token.
    """
    t0 = time.perf_counter()
    with metrics.track_latency():
        guardrails = []

        # 1. INPUT guardrail -- identical to answer().
        allowed, reason, safe_msg = check_input(query)
        if not allowed:
            guardrails.append(f"input:{reason}")
            metrics.record_request("blocked")
            yield {"type": "token", "text": safe_msg}
            yield {"type": "done", "meta": {
                "sources": [], "tools_used": [], "guardrails": guardrails,
                "reply": safe_msg,
                "latency_ms": int((time.perf_counter() - t0) * 1000),
            }}
            return

        # 2. RAG retrieval.
        hits = retrieve(query)
        context = "\n\n".join(f"[{h['source']}] {h['text']}" for h in hits)

        input_list = [
            {"role": "user",
             "content": f"Context snippets:\n{context}\n\nCustomer question: {query}"}
        ]
        tools_used, tool_outputs = [], []

        # 3. Agent loop -- each turn streams from the model; text deltas are
        # only forwarded once we know the turn isn't a tool call (a
        # tool-calling turn emits function_call items, not output text).
        for _ in range(max_turns):
            resp = None
            saw_tool_call = False
            stream = _get_client().responses.create(
                model=CHAT_MODEL,
                instructions=SYSTEM,
                tools=TOOL_SCHEMAS,
                input=input_list,
                stream=True,
            )
            for event in stream:
                if event.type == "response.output_item.added" and getattr(event.item, "type", None) == "function_call":
                    if not saw_tool_call:
                        saw_tool_call = True
                        yield {"type": "status", "text": "Working on it…"}
                elif event.type == "response.output_text.delta" and not saw_tool_call:
                    yield {"type": "token", "text": event.delta}
                elif event.type == "response.completed":
                    resp = event.response

            metrics.record_chat_usage(CHAT_MODEL, getattr(resp, "usage", None))
            calls = [item for item in resp.output if item.type == "function_call"]

            if not calls:
                # 4. OUTPUT guardrail -- runs on the full text, same as answer().
                # Note: token events above already streamed the raw model text,
                # so a guardrail rewrite (redaction/escalation) here changes the
                # authoritative `reply` in the done event's meta but can't
                # retract tokens already sent to the client.
                reply, fired = check_output(resp.output_text, context, tool_outputs)
                guardrails += fired
                outcome = "escalated" if "output:ungrounded" in fired else "deflected"
                metrics.record_request(outcome)
                yield {"type": "done", "meta": {
                    "sources": [h["source"] for h in hits],
                    "tools_used": tools_used,
                    "guardrails": guardrails,
                    "reply": reply,
                    "latency_ms": int((time.perf_counter() - t0) * 1000),
                }}
                return

            input_list += resp.output
            for call in calls:
                tools_used.append(call.name)
                result = run_tool(call.name, call.arguments)
                tool_outputs.append(result)
                input_list.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": result,
                })

        # Safety valve: too many tool round-trips without a final answer.
        fallback = "I'm having trouble completing that right now - let me escalate you to a human agent."
        metrics.record_request("escalated")
        yield {"type": "token", "text": fallback}
        yield {"type": "done", "meta": {
            "sources": [h["source"] for h in hits],
            "tools_used": tools_used,
            "guardrails": guardrails,
            "reply": fallback,
            "latency_ms": int((time.perf_counter() - t0) * 1000),
        }}


if __name__ == "__main__":
    reply, meta = answer("Where is my order ORD-1001?")
    print(reply)
    print("\nTools:", meta["tools_used"], "| Sources:", meta["sources"],
          "| Guardrails:", meta["guardrails"])
