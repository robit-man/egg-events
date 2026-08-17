"""Event recognition and temporal reasoning."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class EventOccurrence:
    event_id: str
    event_type_id: str
    roles: dict[str, str]
    valid_from: str
    valid_to: str | None = None
    observed_at: str = ""
    source_id: str = ""
    evidence_ids: list[str] | None = None
    confidence: float = 0.0
    epistemic_kind: str = "observation"

    def __post_init__(self):
        if self.evidence_ids is None:
            self.evidence_ids = []


@dataclass
class TemporalRelation:
    event_a_id: str
    relation: str  # before, after, during, overlaps, same_as
    event_b_id: str
    confidence: float = 1.0


class EventStore:
    """Event store with temporal relation support."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS world_events (
                    event_id TEXT PRIMARY KEY,
                    event_type_id TEXT NOT NULL,
                    roles_json TEXT NOT NULL DEFAULT '{}',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    observed_at TEXT NOT NULL,
                    source_id TEXT NOT NULL DEFAULT '',
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation'
                );

                CREATE TABLE IF NOT EXISTS temporal_relations (
                    event_a_id TEXT NOT NULL,
                    relation TEXT NOT NULL,
                    event_b_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    PRIMARY KEY (event_a_id, relation, event_b_id)
                );

                CREATE INDEX IF NOT EXISTS idx_we_type ON world_events(event_type_id);
                CREATE INDEX IF NOT EXISTS idx_we_valid ON world_events(valid_from, valid_to);
                CREATE INDEX IF NOT EXISTS idx_tr_a ON temporal_relations(event_a_id);
                CREATE INDEX IF NOT EXISTS idx_tr_b ON temporal_relations(event_b_id);
                """
            )

    def record(self, event: EventOccurrence) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO world_events
                (event_id, event_type_id, roles_json, valid_from, valid_to, observed_at,
                 source_id, evidence_ids_json, confidence, epistemic_kind)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id, event.event_type_id,
                    json.dumps(event.roles, default=str),
                    event.valid_from, event.valid_to,
                    event.observed_at, event.source_id,
                    json.dumps(event.evidence_ids or []),
                    event.confidence, event.epistemic_kind,
                ),
            )
            self._conn.commit()

    def add_temporal_relation(self, rel: TemporalRelation) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO temporal_relations (event_a_id, relation, event_b_id, confidence) VALUES (?, ?, ?, ?)",
                (rel.event_a_id, rel.relation, rel.event_b_id, rel.confidence),
            )
            self._conn.commit()

    def get_events(self, event_type_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            if event_type_id:
                rows = self._conn.execute(
                    "SELECT event_id, event_type_id, roles_json, valid_from, valid_to, observed_at, source_id, evidence_ids_json, confidence FROM world_events WHERE event_type_id = ? ORDER BY valid_from DESC LIMIT ?",
                    (event_type_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT event_id, event_type_id, roles_json, valid_from, valid_to, observed_at, source_id, evidence_ids_json, confidence FROM world_events ORDER BY valid_from DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                {
                    "event_id": r[0], "event_type_id": r[1],
                    "roles": json.loads(r[2]), "valid_from": r[3],
                    "valid_to": r[4], "observed_at": r[5],
                    "source_id": r[6], "evidence_ids": json.loads(r[7]),
                    "confidence": r[8],
                }
                for r in rows
            ]

    def temporal_neighbors(self, event_id: str, relation: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            if relation:
                rows = self._conn.execute(
                    "SELECT event_a_id, relation, event_b_id, confidence FROM temporal_relations WHERE (event_a_id = ? OR event_b_id = ?) AND relation = ?",
                    (event_id, event_id, relation),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT event_a_id, relation, event_b_id, confidence FROM temporal_relations WHERE event_a_id = ? OR event_b_id = ?",
                    (event_id, event_id),
                ).fetchall()
            return [
                {"event_a_id": r[0], "relation": r[1], "event_b_id": r[2], "confidence": r[3]}
                for r in rows
            ]

    def events_in_window(self, start: str, end: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT event_id, event_type_id, roles_json, valid_from, valid_to, observed_at, source_id, evidence_ids_json, confidence
                FROM world_events
                WHERE valid_from <= ? AND (valid_to IS NULL OR valid_to >= ?)
                ORDER BY valid_from""",
                (end, start),
            ).fetchall()
            return [
                {
                    "event_id": r[0], "event_type_id": r[1],
                    "roles": json.loads(r[2]), "valid_from": r[3],
                    "valid_to": r[4], "observed_at": r[5],
                    "source_id": r[6], "evidence_ids": json.loads(r[7]),
                    "confidence": r[8],
                }
                for r in rows
            ]

    def recent_events(self, seconds: float = 300.0) -> list[dict[str, Any]]:
        import datetime
        cutoff = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=seconds)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """SELECT event_id, event_type_id, roles_json, valid_from, valid_to, observed_at, source_id, evidence_ids_json, confidence
                FROM world_events WHERE valid_from >= ? ORDER BY valid_from DESC""",
                (cutoff,),
            ).fetchall()
            return [
                {
                    "event_id": r[0], "event_type_id": r[1],
                    "roles": json.loads(r[2]), "valid_from": r[3],
                    "valid_to": r[4], "observed_at": r[5],
                    "source_id": r[6], "evidence_ids": json.loads(r[7]),
                    "confidence": r[8],
                }
                for r in rows
            ]
