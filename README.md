# Support Agent

[![CI](https://github.com/ibrahimsorady/support-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/ibrahimsorady/support-agent/actions/workflows/ci.yml)

An AI customer-support agent for a (fictional) telecom, built to demonstrate
production patterns for deploying LLMs into real enterprise support workflows:
retrieval-augmented generation (RAG), grounded answers with source attribution,
and refusal/escalation when the answer isn't known.

## What it does today

- Embeds a small telecom knowledge base and retrieves the most relevant snippets
  for each customer question (RAG).
- Answers **only** from retrieved context, cites which docs it used, and
  **escalates to a human** when the KB doesn't cover the question — instead of
  hallucinating a policy or price.

## Quickstart

```bash
# 1. Set up a virtual environment and install dependencies
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Add your OpenAI key
cp .env.example .env        # then edit .env and paste your key

# 3. Build the vector index from the knowledge base (one-time; re-run if KB changes)
python -m src.ingest

# 4. Chat with the agent
python app.py

# 5. Grade the agent against the golden dataset
python -m evals.run_evals            # failures only
python -m evals.run_evals --verbose  # every check
python -m evals.run_evals --trace    # full per-case trace (incl. judge internals)
python -m evals.run_evals --only order_not_found --trace   # trace one case

# 6. Run the offline unit tests (guardrails + eval-check logic, no API key needed)
python -m pytest tests/
```

Try these to see grounding, refusal, and tool-calling in action:

- `How much does roaming cost in Europe?` → grounded answer from the KB (no tool)
- `What's the CEO's personal phone number?` → should refuse and offer to escalate
- `Where is my order ORD-1001?` → calls `lookup_order` and answers from the result
- `Is account +971500000003 active?` → calls `check_account_status`
- `My roaming isn't working, please raise a ticket for +971500000002` → `create_ticket`

## Web UI (demo)

A minimal Streamlit chat UI sits on top of the same `answer()` engine used by
`app.py` — same guardrails, retrieval, tools, and metrics.

```bash
streamlit run streamlit_app.py
```

Each reply's expander shows the retrieved sources, tools used, guardrails
fired, and measured latency.

A second **Backend** tab gives stakeholders a window into the mock backend:
read-only order and account tables, plus a live tickets table (with a
"Resolve" action) backed by Postgres. Tickets require `VECTOR_BACKEND=pgvector`
with the DB running (see below) — the tab shows a friendly message instead of
tickets otherwise, and chat keeps working on the default numpy backend either way.

## Project structure

```
telco-support-agent/
├── app.py                  # CLI chat entry point
├── requirements.txt
├── .env.example
├── docker-compose.yml      # Postgres + pgvector, Prometheus, and Grafana
├── data/kb/                # knowledge base (markdown support docs)
├── mock_backend/
│   └── crm.py              # fake orders/accounts (stands in for Salesforce etc.)
├── evals/
│   ├── cases.yaml          # golden dataset — the graded test cases
│   └── run_evals.py        # runner: feeds cases to the agent, scores results
├── observability/          # Prometheus scrape config + Grafana dashboards
├── scripts/
│   └── simulate_traffic.py # generate sample traffic to populate dashboards
├── tests/                  # offline unit tests (guardrails, eval-check logic)
└── src/
    ├── config.py      # models, retrieval, vector-backend, metrics settings
    ├── db.py          # Postgres/pgvector connection + schema (pgvector only)
    ├── guardrails.py  # input/output safety layer (block / redact / escalate)
    ├── metrics.py     # Prometheus metrics (latency, tokens, cost, outcomes)
    ├── ingest.py      # chunk + embed KB -> numpy file OR Postgres
    ├── retriever.py   # retrieval dispatch: numpy cosine OR pgvector <=>
    ├── tools.py       # tool schemas (the "menu") + dispatcher
    └── agent.py       # RAG + tools + guardrails + metrics (Responses API)
```

## Design notes

- **Responses API** is OpenAI's recommended surface for agentic workflows, so the
  agent is built on it from the start.
- Retrieval defaults to plain numpy cosine similarity to keep the mechanics
  transparent. Set `VECTOR_BACKEND=pgvector` to swap in **pgvector on Postgres**
  — a production-grade vector store — without touching the agent, tools, or evals.

## Optional: pgvector backend

The default numpy backend needs zero setup. To use Postgres + pgvector instead:

```bash
# 1. Start Postgres with pgvector (Docker)
docker compose up -d

# 2. Point the app at it (in .env)
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/telco

# 3. Smoke-test the DB (creates the extension, table, and index)
python -m src.db          # -> "Connected OK. pgvector version: ..., kb_chunks rows: 0"

# 4. Re-ingest (now writes rows into Postgres instead of a file)
python -m src.ingest

# 5. Everything else runs identically
python app.py
python -m evals.run_evals
```

Flip `VECTOR_BACKEND` back to `numpy` anytime — both backends coexist.

## Optional: observability (Prometheus + Grafana)

The agent records latency, token cost, and request outcomes as Prometheus metrics.

```bash
# 1. Start Prometheus + Grafana
docker compose up -d prometheus grafana

# 2. Run the app (or the traffic simulator) so metrics are exposed on :8000
python -m scripts.simulate_traffic --rounds 40 --delay 1
#   ...or just chat: python app.py

# 3. Open Grafana -> the "Support Agent" dashboard
#    http://localhost:3000  (admin / admin; anonymous viewing is enabled)
```

Dashboard panels: deflection rate, cumulative estimated cost, latency p50/p95,
requests by outcome, and tokens by kind. Set your real token prices in `.env`
(`PRICE_INPUT_PER_1M`, `PRICE_OUTPUT_PER_1M`) for an accurate cost figure.
Default model is `gpt-5-mini` (fast, cheap, good for well-scoped support tasks),
configurable in `src/config.py`.

## Roadmap

- [x] **Tools / function-calling**: `lookup_order`, `check_account_status`,
      `create_ticket`, backed by a mock CRM, driven by an agent loop.
- [x] **Eval harness**: golden dataset of 17 cases scoring tool choice, grounded
      correctness, refusals, and guardrails — via structured checks, keywords, and LLM-as-judge.
- [x] **pgvector** backend (Postgres), switchable via `VECTOR_BACKEND`.
- [x] **Guardrails**: input prompt-injection block, output card-leak redaction,
      and an optional model-based grounding check (`ENABLE_GROUNDING_GUARD`).
- [x] **Observability**: Prometheus + Grafana dashboards for latency, token cost,
      request outcomes, and deflection rate.
- [ ] Connection pooling and a small web UI.
