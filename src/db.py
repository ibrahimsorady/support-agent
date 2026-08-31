"""Postgres + pgvector: a pooled connection layer and schema.

Only used when VECTOR_BACKEND=pgvector. A ConnectionPool keeps a small set of
connections open and hands them out on demand, so we pay the expensive
connect + auth + pgvector-registration cost ONCE per connection instead of on
every retrieval. Callers borrow with:

    with db.connection() as conn:
        ...   # connection is automatically returned to the pool on exit

Smoke-test your database with:  python -m src.db
"""
import atexit
import threading

from psycopg_pool import ConnectionPool
from pgvector.psycopg import register_vector

from src.config import DATABASE_URL, DB_POOL_MAX, DB_POOL_MIN, EMBED_DIM

_pool = None  # created lazily so importing this module never opens a connection
_pool_lock = threading.Lock()


def _configure(conn):
    """Runs once per physical connection as the pool creates it.

    CREATE EXTENSION must happen before register_vector, because registering the
    Python <-> 'vector' type mapping needs the type to already exist.
    """
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    conn.commit()
    register_vector(conn)


def get_pool():
    """Return the process-wide pool, opening it on first use."""
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                pool = ConnectionPool(
                    conninfo=DATABASE_URL,
                    min_size=DB_POOL_MIN,
                    max_size=DB_POOL_MAX,
                    configure=_configure,
                    open=False,
                )
                pool.open()
                atexit.register(close_pool)
                _pool = pool
    return _pool


def connection():
    """Borrow a connection from the pool (use as a context manager)."""
    return get_pool().connection()


def close_pool():
    """Close all pooled connections (e.g. on shutdown)."""
    global _pool
    if _pool is not None:
        _pool.close()
        _pool = None


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
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tickets (
                id           BIGSERIAL PRIMARY KEY,
                phone_number TEXT NOT NULL,
                summary      TEXT NOT NULL,
                status       TEXT NOT NULL DEFAULT 'open',
                created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
            );
        """)
    conn.commit()


def insert_ticket(phone_number, summary):
    """Insert a support ticket and return its generated id."""
    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tickets (phone_number, summary) VALUES (%s, %s) RETURNING id;",
                (phone_number, summary),
            )
            ticket_id = cur.fetchone()[0]
        conn.commit()
    return ticket_id


def list_tickets():
    """Return all tickets, most recently created first."""
    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, phone_number, summary, status, created_at "
                "FROM tickets ORDER BY created_at DESC;"
            )
            rows = cur.fetchall()
    return [
        {"id": r[0], "phone_number": r[1], "summary": r[2], "status": r[3], "created_at": r[4]}
        for r in rows
    ]


def set_ticket_status(ticket_id, status):
    """Update a ticket's status (e.g. 'open' -> 'resolved')."""
    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("UPDATE tickets SET status = %s WHERE id = %s;", (status, ticket_id))
        conn.commit()


if __name__ == "__main__":
    with connection() as conn:
        ensure_schema(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector';")
            row = cur.fetchone()
            cur.execute("SELECT count(*) FROM kb_chunks;")
            n = cur.fetchone()[0]
    close_pool()
    print(f"Connected OK (pooled). pgvector version: {row[0] if row else 'NOT INSTALLED'}, "
          f"kb_chunks rows: {n}")
