"""Postgres + pgvector: connection and schema.

Only used when VECTOR_BACKEND=pgvector. Keeps all the database plumbing in one
place so ingest.py and retriever.py stay readable.

Smoke-test your database with:
    python -m src.db
"""
import psycopg
from pgvector.psycopg import register_vector

from src.config import DATABASE_URL, EMBED_DIM


def connect():
    """Open a connection with the pgvector type registered.

    We CREATE EXTENSION *before* register_vector, because registering the Python
    <-> 'vector' type mapping requires the extension (and thus the type) to
    already exist in the database.
    """
    conn = psycopg.connect(DATABASE_URL)
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)
    return conn


def ensure_schema(conn):
    """Create the table and an approximate-nearest-neighbour index if missing."""
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id        BIGSERIAL PRIMARY KEY,
                source    TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding vector({EMBED_DIM}) NOT NULL
            );
        """)
        # HNSW index for cosine distance. Overkill for a tiny KB, but this is the
        # line that keeps search fast at millions of rows -- the reason to use a
        # real vector store at all.
        cur.execute("""
            CREATE INDEX IF NOT EXISTS kb_chunks_embedding_idx
            ON kb_chunks USING hnsw (embedding vector_cosine_ops);
        """)
    conn.commit()


if __name__ == "__main__":
    conn = connect()
    ensure_schema(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
        row = cur.fetchone()
        cur.execute("SELECT count(*) FROM kb_chunks;")
        n = cur.fetchone()[0]
    conn.close()
    print(f"Connected OK. pgvector version: {row[0] if row else 'NOT INSTALLED'}, "
          f"kb_chunks rows: {n}")
