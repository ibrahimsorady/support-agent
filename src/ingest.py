"""Build the vector index from the knowledge base.

Reads every markdown file in data/kb/, splits it into chunks, embeds them once,
and stores them in the configured vector backend:
    VECTOR_BACKEND=numpy     -> saves data/index.npz  (default, zero setup)
    VECTOR_BACKEND=pgvector  -> inserts rows into Postgres

The chunking + embedding is identical either way -- only the storage target
changes. Run it with:  python -m src.ingest
"""
import numpy as np
from openai import OpenAI

from src.config import CHUNK_MAX_CHARS, EMBED_MODEL, INDEX_PATH, KB_DIR, VECTOR_BACKEND

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def load_chunks():
    """Split each KB doc into paragraph-ish chunks, tracking its source file."""
    chunks, sources = [], []
    for path in sorted(KB_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for para in text.split("\n\n"):
            para = para.strip()
            if not para:
                continue
            while len(para) > CHUNK_MAX_CHARS:
                chunks.append(para[:CHUNK_MAX_CHARS])
                sources.append(path.stem)
                para = para[CHUNK_MAX_CHARS:]
            chunks.append(para)
            sources.append(path.stem)
    return chunks, sources


def embed(texts):
    """Embed a list of strings in a single batched API call."""
    resp = _get_client().embeddings.create(model=EMBED_MODEL, input=texts)
    return np.array([d.embedding for d in resp.data], dtype=np.float32)


def _store_numpy(chunks, sources, vectors):
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        INDEX_PATH,
        vectors=vectors,
        chunks=np.array(chunks, dtype=object),
        sources=np.array(sources, dtype=object),
    )
    print(f"Saved index -> {INDEX_PATH}  ({vectors.shape[0]} vectors, dim {vectors.shape[1]})")


def _store_pgvector(chunks, sources, vectors):
    # Imported lazily so the numpy backend needs no database libraries installed.
    from src import db

    with db.connection() as conn:  # borrowed from the pool, returned on exit
        db.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE kb_chunks RESTART IDENTITY;")  # rebuild cleanly
            cur.executemany(
                "INSERT INTO kb_chunks (source, content, embedding) VALUES (%s, %s, %s)",
                [(s, c, v.tolist()) for s, c, v in zip(sources, chunks, vectors)],
            )
        conn.commit()
    print(f"Inserted {len(chunks)} rows into Postgres (kb_chunks), dim {vectors.shape[1]}")


def main():
    chunks, sources = load_chunks()
    if not chunks:
        raise SystemExit(f"No KB docs found in {KB_DIR}. Add some .md files first.")
    print(f"Embedding {len(chunks)} chunks from {KB_DIR} (backend: {VECTOR_BACKEND}) ...")
    vectors = embed(chunks)
    if VECTOR_BACKEND == "pgvector":
        _store_pgvector(chunks, sources, vectors)
    else:
        _store_numpy(chunks, sources, vectors)


if __name__ == "__main__":
    main()
