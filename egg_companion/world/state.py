"""Current-world-state materialized projection.

Provides cheap O(1) lookups for the present believed state without
traversing historical assertions every frame.  Historical assertions
remain append-first; this layer caches the latest accepted value per
entity+property and per entity+relation.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from egg_companion.world.types import AssertionState, TypedValue


@dataclass
class PropertyStateRow:
    entity_id: str
    property_id: str
    value_json: str
    value_type: str
    confidence: float
    authority: float
    assertion_id: str
    evidence_ids_json: str
    epistemic_kind: str
    valid_from: str
    valid_to: str | None
    updated_at: str
    revision: int


@dataclass
class RelationStateRow:
    source_entity_id: str
    relation_type_id: str
    target_entity_id: str
    confidence: float
    authority: float
    assertion_id: str
    evidence_ids_json: str
    epistemic_kind: str
    valid_from: str
    valid_to: str | None
    updated_at: str
    revision: int


class WorldStateStore:
    """Materialized current-state projection backed by SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._revision: int = 0
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS current_property_state (
                    entity_id TEXT NOT NULL,
                    property_id TEXT NOT NULL,
                    value_json TEXT NOT NULL,
                    value_type TEXT NOT NULL DEFAULT 'unknown',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    authority REAL NOT NULL DEFAULT 0.0,
                    assertion_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (entity_id, property_id)
                );

                CREATE TABLE IF NOT EXISTS current_relation_state (
                    source_entity_id TEXT NOT NULL,
                    relation_type_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 0.0,
                    authority REAL NOT NULL DEFAULT 0.0,
                    assertion_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation',
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    updated_at TEXT NOT NULL,
                    revision INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (source_entity_id, relation_type_id, target_entity_id)
                );

                CREATE TABLE IF NOT EXISTS world_state_revisions (
                    revision INTEGER PRIMARY KEY AUTOINCREMENT,
                    description TEXT,
                    created_at TEXT NOT NULL
                );
                """
            )

    @property
    def revision(self) -> int:
        return self._revision

    def increment_revision(self, description: str = "") -> int:
        with self._lock:
            self._revision += 1
            now = datetime.now(timezone.utc).isoformat()
            self._conn.execute(
                "INSERT INTO world_state_revisions (revision, description, created_at) VALUES (?, ?, ?)",
                (self._revision, description, now),
            )
            self._conn.commit()
            return self._revision

    def upsert_property(
        self,
        entity_id: str,
        property_id: str,
        value: TypedValue,
        confidence: float,
        authority: float,
        assertion_id: str,
        evidence_ids: tuple[str, ...],
        epistemic_kind: str,
        valid_from: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._revision += 1
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO current_property_state
                (entity_id, property_id, value_json, value_type, confidence, authority,
                 assertion_id, evidence_ids_json, epistemic_kind, valid_from, valid_to,
                 updated_at, revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    entity_id,
                    property_id,
                    json.dumps(value.raw, default=str),
                    value.value_type.value,
                    confidence,
                    authority,
                    assertion_id,
                    json.dumps(list(evidence_ids)),
                    epistemic_kind,
                    valid_from,
                    now,
                    self._revision,
                ),
            )
            self._conn.commit()

    def upsert_relation(
        self,
        source_entity_id: str,
        relation_type_id: str,
        target_entity_id: str,
        confidence: float,
        authority: float,
        assertion_id: str,
        evidence_ids: tuple[str, ...],
        epistemic_kind: str,
        valid_from: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._revision += 1
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO current_relation_state
                (source_entity_id, relation_type_id, target_entity_id, confidence, authority,
                 assertion_id, evidence_ids_json, epistemic_kind, valid_from, valid_to,
                 updated_at, revision)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                (
                    source_entity_id,
                    relation_type_id,
                    target_entity_id,
                    confidence,
                    authority,
                    assertion_id,
                    json.dumps(list(evidence_ids)),
                    epistemic_kind,
                    valid_from,
                    now,
                    self._revision,
                ),
            )
            self._conn.commit()

    def close_relation(
        self,
        source_entity_id: str,
        relation_type_id: str,
        target_entity_id: str,
        valid_to: str,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._revision += 1
        with self._lock:
            self._conn.execute(
                """UPDATE current_relation_state
                SET valid_to = ?, updated_at = ?, revision = ?
                WHERE source_entity_id = ? AND relation_type_id = ? AND target_entity_id = ?""",
                (valid_to, now, self._revision, source_entity_id, relation_type_id, target_entity_id),
            )
            self._conn.commit()

    def get_property(self, entity_id: str, property_id: str) -> PropertyStateRow | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM current_property_state WHERE entity_id = ? AND property_id = ?",
                (entity_id, property_id),
            ).fetchone()
            if row is None:
                return None
            return PropertyStateRow(
                entity_id=row[0], property_id=row[1], value_json=row[2],
                value_type=row[3], confidence=row[4], authority=row[5],
                assertion_id=row[6], evidence_ids_json=row[7],
                epistemic_kind=row[8], valid_from=row[9], valid_to=row[10],
                updated_at=row[11], revision=row[12],
            )

    def get_entity_state(self, entity_id: str) -> dict[str, PropertyStateRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM current_property_state WHERE entity_id = ?",
                (entity_id,),
            ).fetchall()
            return {
                row[1]: PropertyStateRow(
                    entity_id=row[0], property_id=row[1], value_json=row[2],
                    value_type=row[3], confidence=row[4], authority=row[5],
                    assertion_id=row[6], evidence_ids_json=row[7],
                    epistemic_kind=row[8], valid_from=row[9], valid_to=row[10],
                    updated_at=row[11], revision=row[12],
                )
                for row in rows
            }

    def get_entity_relations(self, entity_id: str) -> list[RelationStateRow]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM current_relation_state
                WHERE (source_entity_id = ? OR target_entity_id = ?) AND valid_to IS NULL""",
                (entity_id, entity_id),
            ).fetchall()
            return [
                RelationStateRow(
                    source_entity_id=row[0], relation_type_id=row[1],
                    target_entity_id=row[2], confidence=row[3], authority=row[4],
                    assertion_id=row[5], evidence_ids_json=row[6],
                    epistemic_kind=row[7], valid_from=row[8], valid_to=row[9],
                    updated_at=row[10], revision=row[11],
                )
                for row in rows
            ]

    def conflicts(self, entity_id: str) -> list[PropertyStateRow]:
        """Return properties with multiple accepted values (conflict)."""
        # For now, conflicts are tracked at assertion level.
        # This returns the current state which may be in CONFLICTED state.
        return []

    def explain(self, entity_id: str, property_id: str) -> dict[str, Any]:
        """Explain why Egg believes a particular state."""
        row = self.get_property(entity_id, property_id)
        if row is None:
            return {"status": "unknown"}
        return {
            "entity_id": entity_id,
            "property_id": property_id,
            "value": json.loads(row.value_json),
            "confidence": row.confidence,
            "authority": row.authority,
            "epistemic_kind": row.epistemic_kind,
            "assertion_id": row.assertion_id,
            "evidence_ids": json.loads(row.evidence_ids_json),
            "valid_from": row.valid_from,
            "valid_to": row.valid_to,
            "revision": row.revision,
        }

    def all_entity_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT entity_id FROM current_property_state"
            ).fetchall()
            return [row[0] for row in rows]

    def all_current_relations(self) -> list[RelationStateRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM current_relation_state WHERE valid_to IS NULL"
            ).fetchall()
            return [
                RelationStateRow(
                    source_entity_id=row[0], relation_type_id=row[1],
                    target_entity_id=row[2], confidence=row[3], authority=row[4],
                    assertion_id=row[5], evidence_ids_json=row[6],
                    epistemic_kind=row[7], valid_from=row[8], valid_to=row[9],
                    updated_at=row[10], revision=row[11],
                )
                for row in rows
            ]
