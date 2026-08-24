"""Minimal command-line chat for the support agent.

Starts a Prometheus /metrics endpoint on startup, then runs the chat loop.
Run from the repo root:  python app.py
"""
from src.agent import answer
from src.config import METRICS_PORT
from src.metrics import start_metrics_server


def main():
    start_metrics_server()
    print(f"Support agent. Metrics on :{METRICS_PORT}/metrics. "
          f"Type 'quit' to exit.\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if query.lower() in {"quit", "exit"}:
            break
        if not query:
            continue
        reply, meta = answer(query)
        print(f"\nAgent: {reply}")
        extras = []
        if meta["tools_used"]:
            extras.append("tools: " + ", ".join(meta["tools_used"]))
        if meta["sources"]:
            extras.append("sources: " + ", ".join(meta["sources"]))
        if meta.get("guardrails"):
            extras.append("guardrails: " + ", ".join(meta["guardrails"]))
        if extras:
            print(f"  ({' | '.join(extras)})\n")


if __name__ == "__main__":
    main()
