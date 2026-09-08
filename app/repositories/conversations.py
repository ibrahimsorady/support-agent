"""Conversation memory: each chat turn is persisted to Postgres (via the same
pooled connection as app.repositories.vector_store) so the agent has
continuity across requests.

A "conversation" isn't backed by its own table -- it's just a UUID, minted
and owned by the backend, that groups rows in `messages`. Callers never trust
a client-supplied conversation_id at face value: conversation_belongs_to()
must confirm it already carries a message from that user before it's reused.

Every public function degrades to a safe no-op (False / [] / silently
skipping the write) if Postgres isn't reachable, so callers that don't care
about memory -- evals, scripts, tests -- never need a live database.
"""
import uuid

from app.repositories import vector_store


def create_conversation(user_id: str) -> str:
    """Mint a new backend-owned conversation id.

    No row is written yet -- the id only becomes a real conversation once the
    first message is added.
    """
    return str(uuid.uuid4())


def conversation_belongs_to(conversation_id: str, user_id: str) -> bool:
    """True only if this conversation already has a message from user_id.

    A conversation_id the backend never issued to this user (unknown, or
    belonging to someone else) comes back False, so the caller falls back to
    a fresh conversation instead of resuming -- or leaking -- someone else's
    history.
    """
    try:
        with vector_store.connection() as conn:
            vector_store.ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT 1 FROM messages WHERE conversation_id = %s AND user_id = %s LIMIT 1;",
                    (conversation_id, user_id),
                )
                return cur.fetchone() is not None
    except Exception:
        return False


def add_message(conversation_id: str, user_id: str, role: str, content: str) -> None:
    try:
        with vector_store.connection() as conn:
            vector_store.ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO messages (conversation_id, user_id, role, content) "
                    "VALUES (%s, %s, %s, %s);",
                    (conversation_id, user_id, role, content),
                )
            conn.commit()
    except Exception:
        pass


def recent_messages(conversation_id: str, limit: int) -> list[dict]:
    """Last `limit` messages for this conversation, oldest first."""
    try:
        with vector_store.connection() as conn:
            vector_store.ensure_schema(conn)
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT role, content FROM messages WHERE conversation_id = %s "
                    "ORDER BY created_at DESC LIMIT %s;",
                    (conversation_id, limit),
                )
                rows = cur.fetchall()
        return [{"role": r[0], "content": r[1]} for r in reversed(rows)]
    except Exception:
        return []
