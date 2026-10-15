from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from support_assistant.storage.database import Database


@dataclass(frozen=True, slots=True)
class UnansweredQuestion:
    question: str
    count: int
    last_asked: str


@dataclass(frozen=True, slots=True)
class AnalyticsReport:
    days: int
    total_questions: int
    answered: int
    unanswered: int
    handoffs: int
    errors: int
    handoff_rate: float
    questions_per_day: dict[str, int]
    unanswered_questions: list[UnansweredQuestion]


class AnalyticsStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def report(self, days: int = 30, unanswered_limit: int = 50) -> AnalyticsReport:
        since = (datetime.now(UTC) - timedelta(days=days)).isoformat(timespec="seconds")
        with self._db.connection() as conn:
            counts = {
                row["outcome"]: row["n"]
                for row in conn.execute(
                    "SELECT outcome, COUNT(*) AS n FROM interactions WHERE created_at >= ? "
                    "GROUP BY outcome",
                    (since,),
                )
            }
            per_day = {
                row["day"]: row["n"]
                for row in conn.execute(
                    "SELECT substr(created_at, 1, 10) AS day, COUNT(*) AS n FROM interactions "
                    "WHERE created_at >= ? GROUP BY day ORDER BY day",
                    (since,),
                )
            }
            gaps = conn.execute(
                "SELECT MIN(question) AS question, COUNT(*) AS n, MAX(created_at) AS last "
                "FROM interactions WHERE created_at >= ? AND "
                "(outcome = 'unanswered' OR (outcome = 'handoff' AND reason = 'low_confidence')) "
                "GROUP BY lower(trim(question)) ORDER BY n DESC, last DESC LIMIT ?",
                (since, unanswered_limit),
            ).fetchall()
        total = sum(counts.values())
        handoffs = counts.get("handoff", 0)
        return AnalyticsReport(
            days=days,
            total_questions=total,
            answered=counts.get("answered", 0),
            unanswered=counts.get("unanswered", 0),
            handoffs=handoffs,
            errors=counts.get("error", 0),
            handoff_rate=round(handoffs / total, 4) if total else 0.0,
            questions_per_day=per_day,
            unanswered_questions=[
                UnansweredQuestion(row["question"], row["n"], row["last"]) for row in gaps
            ],
        )
