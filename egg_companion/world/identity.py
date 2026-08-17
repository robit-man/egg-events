"""Identity subgraph: tracks entity merges, splits, and provenance."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any


@dataclass
class IdentityEvent:
    kind: str  # merge, split, claim, correction
    entities: list[str]
    evidence_ids: list[str]
    source_id: str
    description: str
    recorded_at: str


@dataclass
class IdentityChain:
    entity_id: str
    chain: list[str]
    created_from: list[str]
    created_at: str
    last_event: str


class IdentityGraph:
    """Tracks entity identity provenance: merges, splits, claims, corrections."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS identity_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    entities_json TEXT NOT NULL,
                    evidence_json TEXT NOT NULL DEFAULT '[]',
                    source_id TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS identity_chains (
                    entity_id TEXT PRIMARY KEY,
                    chain_json TEXT NOT NULL DEFAULT '[]',
                    created_from_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    last_event TEXT NOT NULL DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_ie_kind ON identity_events(kind);
                CREATE INDEX IF NOT EXISTS idx_ie_recorded ON identity_events(recorded_at);
                """
            )

    def claim(self, entity_id: str, evidence_ids: list[str], source_id: str, description: str = "", recorded_at: str = "") -> None:
        import datetime
        at = recorded_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            existing = self._conn.execute(
                "SELECT entity_id FROM identity_chains WHERE entity_id = ?", (entity_id,)
            ).fetchone()
            if existing is None:
                self._conn.execute(
                    "INSERT INTO identity_chains (entity_id, chain_json, created_from_json, created_at, last_event) VALUES (?, ?, ?, ?, ?)",
                    (entity_id, json.dumps([entity_id]), json.dumps([]), at, description),
                )
            self._conn.execute(
                "INSERT INTO identity_events (kind, entities_json, evidence_json, source_id, description, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("claim", json.dumps([entity_id]), json.dumps(evidence_ids), source_id, description, at),
            )
            self._conn.commit()

    def merge(self, keep_id: str, merged_ids: list[str], evidence_ids: list[str], source_id: str, description: str = "", recorded_at: str = "") -> None:
        import datetime
        at = recorded_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            for mid in merged_ids:
                self._conn.execute("DELETE FROM identity_chains WHERE entity_id = ?", (mid,))
            chain = [keep_id] + merged_ids
            self._conn.execute(
                "INSERT OR REPLACE INTO identity_chains (entity_id, chain_json, created_from_json, created_at, last_event) VALUES (?, ?, ?, ?, ?)",
                (keep_id, json.dumps(chain), json.dumps(merged_ids), at, description),
            )
            self._conn.execute(
                "INSERT INTO identity_events (kind, entities_json, evidence_json, source_id, description, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("merge", json.dumps([keep_id] + merged_ids), json.dumps(evidence_ids), source_id, description, at),
            )
            self._conn.commit()

    def split(self, parent_id: str, child_ids: list[str], evidence_ids: list[str], source_id: str, description: str = "", recorded_at: str = "") -> None:
        import datetime
        at = recorded_at or datetime.datetime.now(datetime.timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("DELETE FROM identity_chains WHERE entity_id = ?", (parent_id,))
            for cid in child_ids:
                self._conn.execute(
                    "INSERT OR REPLACE INTO identity_chains (entity_id, chain_json, created_from_json, created_at, last_event) VALUES (?, ?, ?, ?, ?)",
                    (cid, json.dumps([cid]), json.dumps([parent_id]), at, description),
                )
            self._conn.execute(
                "INSERT INTO identity_events (kind, entities_json, evidence_json, source_id, description, recorded_at) VALUES (?, ?, ?, ?, ?, ?)",
                ("split", json.dumps([parent_id] + child_ids), json.dumps(evidence_ids), source_id, description, at),
            )
            self._conn.commit()

    def get_chain(self, entity_id: str) -> IdentityChain | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT chain_json, created_from_json, created_at, last_event FROM identity_chains WHERE entity_id = ?",
                (entity_id,),
            ).fetchone()
            if row is None:
                return None
            return IdentityChain(
                entity_id=entity_id,
                chain=json.loads(row[0]),
                created_from=json.loads(row[1]),
                created_at=row[2],
                last_event=row[3],
            )

    def history(self, entity_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if entity_id:
                rows = self._conn.execute(
                    "SELECT kind, entities_json, evidence_json, source_id, description, recorded_at FROM identity_events WHERE entities_json LIKE ? ORDER BY recorded_at DESC LIMIT ?",
                    (f'%"{entity_id}"%', limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT kind, entities_json, evidence_json, source_id, description, recorded_at FROM identity_events ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                {
                    "kind": r[0], "entities": json.loads(r[1]),
                    "evidence_ids": json.loads(r[2]), "source_id": r[3],
                    "description": r[4], "recorded_at": r[5],
                }
                for r in rows
            ]

    def get_claimed_entity_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT entity_id FROM identity_chains").fetchall()
            return [r[0] for r in rows]
