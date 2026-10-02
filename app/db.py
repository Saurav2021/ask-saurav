"""SQLite interaction log + SQL analytics. Raw IPs are never stored, only a salted hash."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    session_id          TEXT,
    visitor             TEXT,
    question            TEXT    NOT NULL,
    standalone_question TEXT,
    answer              TEXT,
    sources             TEXT    NOT NULL DEFAULT '[]',
    latency_ms          INTEGER,
    status              TEXT    NOT NULL DEFAULT 'ok'
);
CREATE INDEX IF NOT EXISTS idx_interactions_ts ON interactions(ts);
CREATE INDEX IF NOT EXISTS idx_interactions_status ON interactions(status);
"""


class ChatLog:
    def __init__(self, path: Path | str, salt: str = "") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._salt = salt
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)

    def visitor_hash(self, ip: str) -> str:
        return hashlib.sha256(f"{self._salt}|{ip}".encode()).hexdigest()[:12]

    def log(self, *, session_id, ip, question, standalone_question, answer, sources, latency_ms, status="ok") -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO interactions (session_id, visitor, question, standalone_question, answer,"
                " sources, latency_ms, status) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, self.visitor_hash(ip), question, standalone_question, answer,
                 json.dumps(sources), latency_ms, status),
            )
            self._conn.commit()

    def _q(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, params).fetchall()]

    def analytics(self, days: int = 30) -> dict:
        since = f"-{int(days)} days"
        totals = self._q(
            """SELECT COUNT(*)                                    AS questions,
                      COUNT(DISTINCT visitor)                     AS unique_visitors,
                      COUNT(DISTINCT session_id)                  AS sessions,
                      ROUND(AVG(latency_ms))                      AS avg_latency_ms,
                      SUM(CASE WHEN status != 'ok' THEN 1 ELSE 0 END) AS errors
               FROM interactions WHERE ts >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)""",
            (since,),
        )[0]
        per_day = self._q(
            """SELECT substr(ts, 1, 10) AS day, COUNT(*) AS questions
               FROM interactions WHERE ts >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
               GROUP BY day ORDER BY day""",
            (since,),
        )
        # json_each unpacks the sources array so we can see which parts of the profile get asked about most
        top_sections = self._q(
            """SELECT json_extract(s.value, '$.section') AS section, COUNT(*) AS hits
               FROM interactions i, json_each(i.sources) s
               WHERE i.ts >= strftime('%Y-%m-%dT%H:%M:%fZ', 'now', ?)
               GROUP BY section ORDER BY hits DESC LIMIT 10""",
            (since,),
        )
        p95 = self._q(
            """SELECT latency_ms FROM interactions
               WHERE status = 'ok' AND latency_ms IS NOT NULL
               ORDER BY latency_ms
               LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.95 AS INT) FROM interactions
                               WHERE status = 'ok' AND latency_ms IS NOT NULL)"""
        )
        recent = self._q(
            "SELECT ts, question, latency_ms, status FROM interactions ORDER BY id DESC LIMIT 20"
        )
        return {
            "window_days": days,
            **totals,
            "p95_latency_ms": p95[0]["latency_ms"] if p95 else None,
            "questions_per_day": per_day,
            "top_sections": top_sections,
            "recent_questions": recent,
        }
