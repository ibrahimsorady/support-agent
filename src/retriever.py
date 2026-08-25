"""Semantic retrieval over the configured vector backend.

Public API is just retrieve(query, k) -> [{text, source, score}], so the agent,
tools, and eval harness never need to know which backend is in use.

    VECTOR_BACKEND=numpy     -> load data/index.npz, cosine similarity in numpy
    VECTOR_BACKEND=pgvector  -> one SQL query using pgvector's <=> operator

The pgvector cosine-distance operator <=> computes exactly what the numpy path
does by hand -- this is the production version of the same math.
"""
import numpy as np
from openai import OpenAI

from src.config import EMBED_MODEL, INDEX_PATH, TOP_K, VECTOR_BACKEND

_client: OpenAI | None = None
_data = None  # lazy-loaded numpy index cache


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _embed_query(text):
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=[text])
    from src import metrics  # lazy import to avoid a hard dependency cycle
    metrics.record_embed_usage(EMBED_MODEL, getattr(resp, "usage", None))
    return np.array(resp.data[0].embedding, dtype=np.float32)


# --- numpy backend --------------------------------------------------------
def _load_numpy():
    global _data
    if _data is None:
        if not INDEX_PATH.exists():
            raise SystemExit(f"No index at {INDEX_PATH}. Run:  python -m src.ingest")
        npz = np.load(INDEX_PATH, allow_pickle=True)
        _data = {
            "vectors": npz["vectors"].astype(np.float32),
            "chunks": npz["chunks"],
            "sources": npz["sources"],
        }
    return _data


def _retrieve_numpy(query, k):
    data = _load_numpy()
    vectors = data["vectors"]
    q = _embed_query(query)
    sims = vectors @ q / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(q) + 1e-8)
    top = np.argsort(sims)[::-1][:k]
    return [
        {"text": str(data["chunks"][i]), "source": str(data["sources"][i]),
         "score": float(sims[i])}
        for i in top
    ]


# --- pgvector backend -----------------------------------------------------
def _retrieve_pgvector(query, k):
    from src import db  # lazy import: only needed for this backend

    q = _embed_query(query)
    with db.connection() as conn:  # borrowed from the pool, returned on exit
        with conn.cursor() as cur:
            # <=> is cosine DISTANCE (0 = identical). We ORDER BY it ascending to get
            # nearest neighbours, and report 1 - distance as a similarity score so the
            # numbers line up with the numpy path.
            cur.execute(
                """
                SELECT source, content, 1 - (embedding <=> %s) AS score
                FROM kb_chunks
                ORDER BY embedding <=> %s
                LIMIT %s
                """,
                (q, q, k),
            )
            rows = cur.fetchall()
    return [{"source": r[0], "text": r[1], "score": float(r[2])} for r in rows]


# --- public dispatch ------------------------------------------------------
def retrieve(query, k=TOP_K):
    if VECTOR_BACKEND == "pgvector":
        return _retrieve_pgvector(query, k)
    return _retrieve_numpy(query, k)
