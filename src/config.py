"""Central configuration: models, retrieval settings, and paths.

Everything you might want to tweak lives here so the rest of the code stays clean.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads OPENAI_API_KEY (and any overrides) from a local .env file

# --- Models ---------------------------------------------------------------
# gpt-5-mini is a documented, widely-available API model id and a good
# cost/quality default for a support agent. If your account has access to a
# newer mini and you want to bump it, change CHAT_MODEL here (or in .env).
#
# To see exactly which models your key can call, run:
#   python -c "from openai import OpenAI; print([m.id for m in OpenAI().models.list()])"
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-5-mini")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
# The model that grades eval answers (LLM-as-judge). Defaults to the chat model;
# in production you might point this at a stronger model than the one on trial.
JUDGE_MODEL = os.getenv("JUDGE_MODEL", CHAT_MODEL)

# --- Retrieval ------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "3"))          # how many KB chunks to feed the model
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "800"))  # cap per chunk

# --- Vector store ---------------------------------------------------------
# Which backend stores/searches the embeddings:
#   "numpy"    -> file-based in-memory index (data/index.npz). Zero setup.
#   "pgvector" -> Postgres + the pgvector extension (needs a running DB).
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "numpy")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/telco")
# Embedding dimension. text-embedding-3-small = 1536. Must match the model and
# the vector(...) column width in Postgres.
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))

# --- Guardrails -----------------------------------------------------------
# The model-based output grounding guard costs one extra API call per reply, so
# it's off by default. Set ENABLE_GROUNDING_GUARD=true to turn it on.
ENABLE_GROUNDING_GUARD = os.getenv("ENABLE_GROUNDING_GUARD", "false").lower() == "true"

# --- Observability --------------------------------------------------------
# Port where the app exposes Prometheus metrics (GET /metrics).
METRICS_PORT = int(os.getenv("METRICS_PORT", "8000"))
# Prices in USD per 1M tokens, used only to ESTIMATE the cost metric.
# These are placeholders -- set them to your model's current published prices.
PRICE_INPUT_PER_1M = float(os.getenv("PRICE_INPUT_PER_1M", "0.25"))
PRICE_OUTPUT_PER_1M = float(os.getenv("PRICE_OUTPUT_PER_1M", "2.00"))
PRICE_EMBED_PER_1M = float(os.getenv("PRICE_EMBED_PER_1M", "0.02"))

# --- Paths ----------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
KB_DIR = ROOT / "data" / "kb"
INDEX_PATH = ROOT / "data" / "index.npz"
