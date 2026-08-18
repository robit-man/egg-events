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
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Generator

from egg_companion.world.types import TypedValue


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
    """Materialized current-state projection backed by SQLite.

    Atomicity guarantee: all inner mutation methods (upsert_property,
    upsert_relation, close_relation) NEVER commit on their own.  Only
    the ``world_transaction()`` context manager commits or rolls back.
    """

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()
        self._revision = self._load_max_revision()
        self._in_transaction = False
        self._current_revision: int = 0

    def _load_max_revision(self) -> int:
        """Load the maximum persisted revision for restart safety."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM world_state_revisions"
            ).fetchone()
            return int(row[0]) if row else 0

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

    def allocate_revision(self, description: str = "") -> int:
        """Allocate a new revision within the current transaction.

        When inside a transaction this is used by the reconciler to assign
        one revision to all mutations in a single WorldDelta.  Outside a
        transaction it creates and commits a standalone revision row.
        """
        with self._lock:
            now = datetime.now(timezone.utc).isoformat()
            cursor = self._conn.execute(
                "INSERT INTO world_state_revisions (description, created_at) VALUES (?, ?)",
                (description, now),
            )
            self._revision = cursor.lastrowid
            self._current_revision = self._revision
            if not self._in_transaction:
                self._conn.commit()
            return self._revision

    def _allocate_standalone_revision(self) -> int:
        """Allocate revision outside a transaction (fallback)."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self._conn.execute(
            "INSERT INTO world_state_revisions (description, created_at) VALUES (?, ?)",
            ("standalone", now),
        )
        self._revision = cursor.lastrowid
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
        revision: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rev = revision or (
            self._current_revision if self._in_transaction
            else self._allocate_standalone_revision()
        )
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
                    rev,
                ),
            )

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
        revision: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rev = revision or (
            self._current_revision if self._in_transaction
            else self._allocate_standalone_revision()
        )
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
                    rev,
                ),
            )

    def close_relation(
        self,
        source_entity_id: str,
        relation_type_id: str,
        target_entity_id: str,
        valid_to: str,
        revision: int | None = None,
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        rev = revision or (
            self._current_revision if self._in_transaction
            else self._allocate_standalone_revision()
        )
        with self._lock:
            self._conn.execute(
                """UPDATE current_relation_state
                SET valid_to = ?, updated_at = ?, revision = ?
                WHERE source_entity_id = ? AND relation_type_id = ? AND target_entity_id = ?""",
                (valid_to, now, rev, source_entity_id, relation_type_id, target_entity_id),
            )

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

    def conflicts(self, entity_id: str = "") -> list[PropertyStateRow]:
        """Return properties that have multiple active accepted assertions.

        Queries the assertion log for (entity_id, property_id) pairs that
        have more than one accepted/conflicted assertion.  Returns the
        corresponding current-state rows.
        """
        with self._lock:
            if entity_id:
                rows = self._conn.execute(
                    """SELECT DISTINCT subject_id, property_id FROM world_assertions
                    WHERE subject_id = ? AND state IN ('conflicted', 'accepted')
                    GROUP BY subject_id, property_id HAVING COUNT(*) > 1""",
                    (entity_id,),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """SELECT DISTINCT subject_id, property_id FROM world_assertions
                    WHERE state IN ('conflicted', 'accepted')
                    GROUP BY subject_id, property_id HAVING COUNT(*) > 1"""
                ).fetchall()
            result = []
            for subject, prop in rows:
                row = self._conn.execute(
                    "SELECT * FROM current_property_state WHERE entity_id = ? AND property_id = ?",
                    (subject, prop),
                ).fetchone()
                if row:
                    result.append(PropertyStateRow(
                        entity_id=row[0], property_id=row[1], value_json=row[2],
                        value_type=row[3], confidence=row[4], authority=row[5],
                        assertion_id=row[6], evidence_ids_json=row[7],
                        epistemic_kind=row[8], valid_from=row[9], valid_to=row[10],
                        updated_at=row[11], revision=row[12],
                    ))
            return result

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

    def entity_brief_counts(self) -> dict[str, dict[str, object]]:
        """Bulk property/relation counts per entity in a single query."""
        with self._lock:
            prop_counts = {}
            for row in self._conn.execute(
                "SELECT entity_id, COUNT(*), MAX(updated_at) FROM current_property_state GROUP BY entity_id"
            ).fetchall():
                prop_counts[row[0]] = {"property_count": row[1], "last_updated": row[2]}
            rel_counts = {}
            for row in self._conn.execute(
                "SELECT source_entity_id, COUNT(*) FROM current_relation_state WHERE valid_to IS NULL GROUP BY source_entity_id"
            ).fetchall():
                rel_counts[row[0]] = row[1]
            for row in self._conn.execute(
                "SELECT target_entity_id, COUNT(*) FROM current_relation_state WHERE valid_to IS NULL GROUP BY target_entity_id"
            ).fetchall():
                rel_counts[row[0]] = rel_counts.get(row[0], 0) + row[1]
            result = {}
            for eid, info in prop_counts.items():
                result[eid] = {
                    "property_count": info["property_count"],
                    "relation_count": rel_counts.get(eid, 0),
                    "last_updated": info["last_updated"],
                }
            return result

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

    @contextmanager
    def world_transaction(self, description: str = "world_delta") -> Generator[None, None, None]:
        """Atomic world state transaction.

        Sets ``_in_transaction = True`` so that inner methods never commit
        independently.  Allocates a single revision for all mutations in
        this transaction.  Commits everything atomically at the end; rolls
        back on any exception.
        """
        with self._lock:
            self._in_transaction = True
            previous_revision = self._revision
            try:
                self._current_revision = self.allocate_revision(description)
                yield
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                self._revision = previous_revision
                raise
            finally:
                self._in_transaction = False
