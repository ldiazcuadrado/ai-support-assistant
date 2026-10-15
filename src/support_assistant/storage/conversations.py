from __future__ import annotations

from support_assistant.domain import Message, Outcome, Role
from support_assistant.storage.database import Database, utcnow

FAILURE_OUTCOMES = (Outcome.UNANSWERED, Outcome.ERROR)


class ConversationStore:
    """Conversation log, per-session memory and the interaction record used for analytics."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def add_exchange(
        self,
        session_id: str,
        channel: str,
        question: str,
        answer: str,
        outcome: Outcome,
        *,
        reason: str | None = None,
        top_relevance: float | None = None,
    ) -> None:
        now = utcnow()
        with self._db.connection() as conn:
            conn.executemany(
                "INSERT INTO messages(session_id, channel, role, content, created_at) "
                "VALUES (?,?,?,?,?)",
                [
                    (session_id, channel, Role.USER.value, question, now),
                    (session_id, channel, Role.ASSISTANT.value, answer, now),
                ],
            )
            conn.execute(
                "INSERT INTO interactions"
                "(session_id, channel, question, outcome, reason, top_relevance, created_at) "
                "VALUES (?,?,?,?,?,?,?)",
                (session_id, channel, question, outcome.value, reason, top_relevance, now),
            )

    def recent(self, session_id: str, turns: int) -> list[Message]:
        """Return up to ``turns`` user/assistant exchanges, oldest first."""
        if turns <= 0:
            return []
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, turns * 2),
            ).fetchall()
        messages = [Message(Role(row["role"]), row["content"]) for row in reversed(rows)]
        while messages and messages[0].role is not Role.USER:
            messages.pop(0)
        return messages

    def consecutive_failures(self, session_id: str, limit: int) -> int:
        """Count trailing failed interactions in a session, up to ``limit``."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT outcome FROM interactions WHERE session_id = ? ORDER BY id DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        count = 0
        for row in rows:
            if Outcome(row["outcome"]) not in FAILURE_OUTCOMES:
                break
            count += 1
        return count

    def transcript(self, session_id: str) -> list[tuple[str, str, str]]:
        """Return ``(role, content, created_at)`` for every message in a session."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [(row["role"], row["content"], row["created_at"]) for row in rows]
