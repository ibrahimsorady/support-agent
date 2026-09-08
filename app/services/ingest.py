"""Build the vector index from the knowledge base.

Reads every markdown file in data/kb/, splits it into chunks, embeds them once,
and stores them in the configured vector backend:
    VECTOR_BACKEND=numpy     -> saves data/index.npz  (default, zero setup)
    VECTOR_BACKEND=pgvector  -> inserts rows into Postgres

The chunking + embedding is identical either way -- only the storage target
changes. Run it with:  python -m app.services.ingest
"""
import re
from datetime import datetime, timezone

import numpy as np
from openai import OpenAI

from app.config import CHUNK_MAX_CHARS, EMBED_MODEL, INDEX_PATH, KB_DIR, VECTOR_BACKEND
from app.services import retriever

_client: OpenAI | None = None
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


def _safe_name(raw: str) -> str:
    """Turn a client-supplied filename/doc name into a safe KB doc stem.

    Strips any path prefix and a .md/.txt suffix, then requires the rest to
    be plain filename characters starting with an alphanumeric -- so a
    crafted name (e.g. "../../etc/passwd") can never escape KB_DIR.
    """
    name = (raw or "").strip().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    if name.lower().endswith((".md", ".txt")):
        name = name.rsplit(".", 1)[0]
    if not name or not _NAME_RE.match(name):
        raise ValueError(f"Invalid document name: {raw!r}")
    return name


def _chunk_text(text: str) -> list[str]:
    """Split one document's raw text into paragraph-ish chunks.

    A markdown heading on its own ("## Pay-as-you-go roaming rates") carries
    no retrievable content by itself, so it's glued to the paragraph that
    follows rather than becoming its own chunk -- otherwise it can win a
    top-K retrieval slot and crowd out the paragraph with the actual answer.
    """
    chunks = []
    pending_heading = None
    for para in text.split("\n\n"):
        para = para.strip()
        if not para:
            continue
        if para.startswith("#"):
            pending_heading = para if pending_heading is None else f"{pending_heading}\n{para}"
            continue
        if pending_heading is not None:
            para = f"{pending_heading}\n{para}"
            pending_heading = None
        while len(para) > CHUNK_MAX_CHARS:
            chunks.append(para[:CHUNK_MAX_CHARS])
            para = para[CHUNK_MAX_CHARS:]
        chunks.append(para)
    if pending_heading is not None:  # trailing heading with no body
        chunks.append(pending_heading)
    return chunks


def load_chunks():
    """Chunk every KB doc in KB_DIR, tracking each chunk's source file."""
    chunks, sources = [], []
    for path in sorted(KB_DIR.glob("*.md")):
        doc_chunks = _chunk_text(path.read_text(encoding="utf-8"))
        chunks.extend(doc_chunks)
        sources.extend([path.stem] * len(doc_chunks))
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
    from app.repositories import vector_store

    with vector_store.connection() as conn:  # borrowed from the pool, returned on exit
        vector_store.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("TRUNCATE kb_chunks RESTART IDENTITY;")  # rebuild cleanly
            cur.executemany(
                "INSERT INTO kb_chunks (source, content, embedding) VALUES (%s, %s, %s)",
                [(s, c, v.tolist()) for s, c, v in zip(sources, chunks, vectors)],
            )
        conn.commit()
    print(f"Inserted {len(chunks)} rows into Postgres (kb_chunks), dim {vectors.shape[1]}")


# --- single-document CRUD (used by the /kb router) -------------------------
# Bulk main() above rebuilds the whole index from KB_DIR; these instead touch
# only one document's chunks, so an upload/delete doesn't re-embed the rest
# of the knowledge base.

def _load_numpy_index():
    if not INDEX_PATH.exists():
        return [], [], None
    npz = np.load(INDEX_PATH, allow_pickle=True)
    return npz["chunks"].tolist(), npz["sources"].tolist(), npz["vectors"].astype(np.float32)


def _upsert_numpy(name, doc_chunks, vectors):
    chunks, sources, existing_vectors = _load_numpy_index()
    keep = [i for i, s in enumerate(sources) if s != name]
    chunks = [chunks[i] for i in keep] + doc_chunks
    sources = [sources[i] for i in keep] + [name] * len(doc_chunks)
    all_vectors = np.vstack([existing_vectors[keep], vectors]) if keep else vectors
    _store_numpy(chunks, sources, all_vectors)


def _delete_numpy(name):
    chunks, sources, existing_vectors = _load_numpy_index()
    keep = [i for i, s in enumerate(sources) if s != name]
    removed = len(sources) - len(keep)
    if removed == 0:
        return 0
    if keep:
        _store_numpy([chunks[i] for i in keep], [sources[i] for i in keep], existing_vectors[keep])
    else:
        INDEX_PATH.unlink()
    return removed


def _list_numpy():
    _, sources, _ = _load_numpy_index()
    counts: dict[str, int] = {}
    for s in sources:
        counts[s] = counts.get(s, 0) + 1
    docs = []
    for path in sorted(KB_DIR.glob("*.md")):
        stat = path.stat()
        docs.append({
            "name": path.stem,
            "chunks": counts.get(path.stem, 0),
            "updated_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        })
    return docs


def _upsert_pgvector_doc(name, doc_chunks, vectors):
    # Imported lazily so the numpy backend needs no database libraries installed.
    from app.repositories import vector_store

    with vector_store.connection() as conn:
        vector_store.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_chunks WHERE source = %s;", (name,))
            cur.executemany(
                "INSERT INTO kb_chunks (source, content, embedding) VALUES (%s, %s, %s)",
                [(name, c, v.tolist()) for c, v in zip(doc_chunks, vectors)],
            )
        conn.commit()


def _delete_pgvector_doc(name):
    from app.repositories import vector_store

    with vector_store.connection() as conn:
        vector_store.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_chunks WHERE source = %s;", (name,))
            removed = cur.rowcount
        conn.commit()
    return removed


def _list_pgvector_docs():
    from app.repositories import vector_store

    with vector_store.connection() as conn:
        vector_store.ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT source, count(*), max(created_at) FROM kb_chunks "
                "GROUP BY source ORDER BY source;"
            )
            rows = cur.fetchall()
    return [{"name": r[0], "chunks": r[1], "updated_at": r[2].isoformat()} for r in rows]


def upsert_doc(raw_name: str, content: str) -> dict:
    """Chunk+embed+store one document, replacing any prior chunks under the
    same name. The raw text is persisted to KB_DIR so it survives restarts
    and future `python -m app.services.ingest` rebuilds. Returns {name, chunks}.
    """
    if not content.strip():
        raise ValueError("Document content is empty")
    name = _safe_name(raw_name)
    doc_chunks = _chunk_text(content)
    if not doc_chunks:
        raise ValueError("Document has no indexable content")
    vectors = embed(doc_chunks)

    KB_DIR.mkdir(parents=True, exist_ok=True)
    (KB_DIR / f"{name}.md").write_text(content, encoding="utf-8")

    if VECTOR_BACKEND == "pgvector":
        _upsert_pgvector_doc(name, doc_chunks, vectors)
    else:
        _upsert_numpy(name, doc_chunks, vectors)
    retriever.invalidate_cache()
    return {"name": name, "chunks": len(doc_chunks)}


def delete_doc(raw_name: str) -> dict:
    """Remove a document's stored file and every indexed chunk for it.

    Raises LookupError if no such document (file or chunks) exists.
    """
    name = _safe_name(raw_name)
    path = KB_DIR / f"{name}.md"
    existed = path.exists()
    if existed:
        path.unlink()

    removed = _delete_pgvector_doc(name) if VECTOR_BACKEND == "pgvector" else _delete_numpy(name)
    if not existed and removed == 0:
        raise LookupError(name)

    retriever.invalidate_cache()
    return {"name": name, "deleted": True}


def list_docs() -> list[dict]:
    """[{name, chunks, updated_at}] for every currently stored document."""
    if VECTOR_BACKEND == "pgvector":
        return _list_pgvector_docs()
    return _list_numpy()


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
