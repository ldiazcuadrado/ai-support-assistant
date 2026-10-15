from __future__ import annotations

import sqlite3

from support_assistant.domain import HandoffReason, Ticket, TicketMessage, TicketStatus
from support_assistant.storage.database import Database, utcnow


def _message(row: sqlite3.Row) -> TicketMessage:
    return TicketMessage(
        id=row["id"],
        ticket_id=row["ticket_id"],
        author=row["author"],
        content=row["content"],
        created_at=row["created_at"],
    )


def _ticket(row: sqlite3.Row, messages: tuple[TicketMessage, ...] = ()) -> Ticket:
    return Ticket(
        id=row["id"],
        session_id=row["session_id"],
        channel=row["channel"],
        status=TicketStatus(row["status"]),
        reason=HandoffReason(row["reason"]),
        question=row["question"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        messages=messages,
    )


class TicketStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def create(self, session_id: str, channel: str, reason: HandoffReason, question: str) -> Ticket:
        now = utcnow()
        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO tickets(session_id, channel, status, reason, question, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
                (session_id, channel, TicketStatus.OPEN.value, reason.value, question, now, now),
            )
            assert cur.lastrowid is not None
            ticket_id = cur.lastrowid
            conn.execute(
                "INSERT INTO ticket_messages(ticket_id, author, content, created_at) "
                "VALUES (?,?,?,?)",
                (ticket_id, "customer", question, now),
            )
        ticket = self.get(ticket_id)
        assert ticket is not None
        return ticket

    def get(self, ticket_id: int) -> Ticket | None:
        with self._db.connection() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE id = ?", (ticket_id,)).fetchone()
            if row is None:
                return None
            msgs = conn.execute(
                "SELECT * FROM ticket_messages WHERE ticket_id = ? ORDER BY id", (ticket_id,)
            ).fetchall()
        return _ticket(row, tuple(_message(m) for m in msgs))

    def list_tickets(self, status: TicketStatus | None = None, limit: int = 100) -> list[Ticket]:
        sql = "SELECT * FROM tickets"
        params: tuple[object, ...] = ()
        if status is not None:
            sql += " WHERE status = ?"
            params = (status.value,)
        sql += " ORDER BY id DESC LIMIT ?"
        with self._db.connection() as conn:
            rows = conn.execute(sql, (*params, limit)).fetchall()
        return [_ticket(row) for row in rows]

    def find_active(self, session_id: str) -> Ticket | None:
        """Return the newest ticket of a session that is not closed."""
        with self._db.connection() as conn:
            row = conn.execute(
                "SELECT id FROM tickets WHERE session_id = ? AND status != ? "
                "ORDER BY id DESC LIMIT 1",
                (session_id, TicketStatus.CLOSED.value),
            ).fetchone()
        return None if row is None else self.get(row["id"])

    def add_message(self, ticket_id: int, author: str, content: str) -> TicketMessage:
        now = utcnow()
        with self._db.connection() as conn:
            cur = conn.execute(
                "INSERT INTO ticket_messages(ticket_id, author, content, created_at) "
                "VALUES (?,?,?,?)",
                (ticket_id, author, content, now),
            )
            conn.execute("UPDATE tickets SET updated_at = ? WHERE id = ?", (now, ticket_id))
            row = conn.execute(
                "SELECT * FROM ticket_messages WHERE id = ?", (cur.lastrowid,)
            ).fetchone()
        return _message(row)

    def set_status(self, ticket_id: int, status: TicketStatus) -> None:
        with self._db.connection() as conn:
            conn.execute(
                "UPDATE tickets SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, utcnow(), ticket_id),
            )

    def operator_replies(self, session_id: str, after_id: int = 0) -> list[TicketMessage]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT m.* FROM ticket_messages m JOIN tickets t ON t.id = m.ticket_id "
                "WHERE t.session_id = ? AND m.author = 'operator' AND m.id > ? ORDER BY m.id",
                (session_id, after_id),
            ).fetchall()
        return [_message(row) for row in rows]
