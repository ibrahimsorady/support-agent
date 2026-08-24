"""RAG + tool-calling agent, wrapped in guardrails and instrumented for metrics.

Flow per request:
  1. INPUT guardrail  -> block unsafe requests before any model call
  2. RAG retrieval    -> pull relevant KB snippets
  3. Agent loop       -> model answers, or requests tools we run and feed back
  4. OUTPUT guardrail -> redact leaks / catch ungrounded answers before replying

Every request is timed, its token usage recorded, and its outcome classified
(deflected / escalated / blocked) so Prometheus + Grafana can show latency,
cost, and deflection rate. See src/metrics.py.
"""
from openai import OpenAI

from src import metrics
from src.config import CHAT_MODEL
from src.guardrails import check_input, check_output
from src.retriever import retrieve
from src.tools import TOOL_SCHEMAS, run_tool

client = OpenAI()

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
            resp = client.responses.create(
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


if __name__ == "__main__":
    reply, meta = answer("Where is my order ORD-1001?")
    print(reply)
    print("\nTools:", meta["tools_used"], "| Sources:", meta["sources"],
          "| Guardrails:", meta["guardrails"])
