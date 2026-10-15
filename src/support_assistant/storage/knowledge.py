from __future__ import annotations

import sqlite3
from array import array
from collections.abc import Sequence

from support_assistant.domain import Chunk
from support_assistant.storage.database import Database


def _chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        id=row["id"],
        source=row["source"],
        title=row["title"],
        page=row["page"],
        position=row["position"],
        text=row["text"],
    )


class KnowledgeStore:
    def __init__(self, db: Database) -> None:
        self._db = db

    def replace_source(self, source: str, chunks: Sequence[Chunk]) -> list[int]:
        """Atomically replace every chunk of ``source``; returns the new chunk ids."""
        with self._db.connection() as conn:
            conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            ids: list[int] = []
            for chunk in chunks:
                cur = conn.execute(
                    "INSERT INTO chunks(source, title, page, position, text) VALUES (?,?,?,?,?)",
                    (chunk.source, chunk.title, chunk.page, chunk.position, chunk.text),
                )
                assert cur.lastrowid is not None
                ids.append(cur.lastrowid)
        return ids

    def clear(self) -> None:
        with self._db.connection() as conn:
            conn.execute("DELETE FROM chunks")

    def get_many(self, ids: Sequence[int]) -> dict[int, Chunk]:
        if not ids:
            return {}
        marks = ",".join("?" * len(ids))
        with self._db.connection() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE id IN ({marks})", tuple(ids)
            ).fetchall()
        return {row["id"]: _chunk(row) for row in rows}

    def search_fts(self, match: str, limit: int) -> list[tuple[Chunk, float]]:
        """Run an FTS5 query. Scores are positive; higher means a better match."""
        sql = (
            "SELECT c.*, -bm25(chunks_fts, 4.0, 1.0) AS score "
            "FROM chunks_fts JOIN chunks c ON c.id = chunks_fts.rowid "
            "WHERE chunks_fts MATCH ? ORDER BY score DESC LIMIT ?"
        )
        with self._db.connection() as conn:
            rows = conn.execute(sql, (match, limit)).fetchall()
        return [(_chunk(row), float(row["score"])) for row in rows]

    def store_embeddings(self, model: str, vectors: dict[int, Sequence[float]]) -> None:
        with self._db.connection() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO embeddings(chunk_id, model, vector) VALUES (?,?,?)",
                [(cid, model, array("f", vec).tobytes()) for cid, vec in vectors.items()],
            )

    def load_embeddings(self, model: str) -> list[tuple[int, array[float]]]:
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT chunk_id, vector FROM embeddings WHERE model = ?", (model,)
            ).fetchall()
        result: list[tuple[int, array[float]]] = []
        for row in rows:
            vec: array[float] = array("f")
            vec.frombytes(row["vector"])
            result.append((row["chunk_id"], vec))
        return result

    def count_chunks(self) -> int:
        with self._db.connection() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])

    def list_sources(self) -> list[tuple[str, str, int]]:
        """Return ``(source, title, chunk_count)`` for every ingested source."""
        with self._db.connection() as conn:
            rows = conn.execute(
                "SELECT source, MIN(title) AS title, COUNT(*) AS n FROM chunks "
                "GROUP BY source ORDER BY source"
            ).fetchall()
        return [(row["source"], row["title"], row["n"]) for row in rows]
