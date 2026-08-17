"""Reconciler: merges, validates, and settles typed assertions."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from egg_companion.world.assertions import EventAssertion, RelationAssertion, WorldAssertion
from egg_companion.world.state import WorldStateStore
from egg_companion.world.types import (
    AssertionKind,
    AssertionState,
    EpistemicKind,
    TypedValue,
    WorldDelta,
)


@dataclass
class ConflictRecord:
    assertion_id: str
    entity_id: str
    property_id: str
    current_value: Any
    proposed_value: Any
    reason: str


class Reconciler:
    """Accepts WorldDelta, validates, deduplicates, merges conflicts."""

    def __init__(
        self,
        assertion_conn: sqlite3.Connection,
        state_store: WorldStateStore,
    ) -> None:
        self._conn = assertion_conn
        self._state = state_store
        self._lock = threading.RLock()
        self._ensure_tables()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS world_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL DEFAULT 'property',
                    subject_id TEXT NOT NULL,
                    property_id TEXT,
                    value_json TEXT NOT NULL,
                    value_type TEXT NOT NULL DEFAULT 'unknown',
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation',
                    source_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    authority REAL NOT NULL DEFAULT 0.0,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    observed_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'proposed',
                    revision_of TEXT,
                    ontology_revision INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_wa_subject ON world_assertions(subject_id);
                CREATE INDEX IF NOT EXISTS idx_wa_property ON world_assertions(subject_id, property_id);
                CREATE INDEX IF NOT EXISTS idx_wa_state ON world_assertions(state);
                CREATE INDEX IF NOT EXISTS idx_wa_valid ON world_assertions(valid_from, valid_to);

                CREATE TABLE IF NOT EXISTS relation_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    source_entity_id TEXT NOT NULL,
                    relation_type_id TEXT NOT NULL,
                    target_entity_id TEXT NOT NULL,
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation',
                    source_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    authority REAL NOT NULL DEFAULT 0.0,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    observed_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'proposed',
                    revision_of TEXT,
                    ontology_revision INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_ra_source ON relation_assertions(source_entity_id);
                CREATE INDEX IF NOT EXISTS idx_ra_target ON relation_assertions(target_entity_id);
                CREATE INDEX IF NOT EXISTS idx_ra_relation ON relation_assertions(relation_type_id);
                CREATE INDEX IF NOT EXISTS idx_ra_state ON relation_assertions(state);

                CREATE TABLE IF NOT EXISTS event_assertions (
                    assertion_id TEXT PRIMARY KEY,
                    event_type_id TEXT NOT NULL,
                    roles_json TEXT NOT NULL DEFAULT '{}',
                    epistemic_kind TEXT NOT NULL DEFAULT 'observation',
                    source_id TEXT NOT NULL,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    confidence REAL NOT NULL DEFAULT 0.0,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    observed_at TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    state TEXT NOT NULL DEFAULT 'proposed',
                    ontology_revision INTEGER NOT NULL DEFAULT 1
                );

                CREATE INDEX IF NOT EXISTS idx_ea_type ON event_assertions(event_type_id);
                CREATE INDEX IF NOT EXISTS idx_ea_valid ON event_assertions(valid_from, valid_to);
                """
            )

    def _next_id(self, prefix: str) -> str:
        import uuid
        return f"{prefix}:{uuid.uuid4().hex[:12]}"

    def _insert_assertion(self, a: dict[str, Any], kind: str) -> str:
        aid = self._next_id("assert")
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            if kind == "property":
                self._conn.execute(
                    """INSERT INTO world_assertions
                    (assertion_id, kind, subject_id, property_id, value_json, value_type,
                     epistemic_kind, source_id, evidence_ids_json, confidence, authority,
                     valid_from, valid_to, observed_at, recorded_at, state, ontology_revision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'proposed', ?)""",
                    (
                        aid, kind,
                        a["subject_id"], a.get("property_id", ""),
                        json.dumps(a["value"].raw, default=str),
                        a["value"].value_type.value,
                        a["epistemic_kind"], a["source_id"],
                        json.dumps(list(a.get("evidence_ids", ()))),
                        a.get("confidence", 0.0),
                        a.get("authority", 0.0),
                        a["valid_from"],
                        a.get("observed_at", now), now,
                        a.get("ontology_revision", 1),
                    ),
                )
            elif kind == "relation":
                self._conn.execute(
                    """INSERT INTO relation_assertions
                    (assertion_id, source_entity_id, relation_type_id, target_entity_id,
                     epistemic_kind, source_id, evidence_ids_json, confidence, authority,
                     valid_from, valid_to, observed_at, recorded_at, state, ontology_revision)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, 'proposed', ?)""",
                    (
                        aid,
                        a["source_entity_id"], a["relation_type_id"], a["target_entity_id"],
                        a.get("epistemic_kind", "observation"), a.get("source_id", ""),
                        json.dumps(list(a.get("evidence_ids", ()))),
                        a.get("confidence", 0.0), a.get("authority", 0.0),
                        a["valid_from"],
                        a.get("observed_at", now), now,
                        a.get("ontology_revision", 1),
                    ),
                )
            self._conn.commit()
        return aid

    def _check_conflict(self, entity_id: str, property_id: str, value: Any, source_id: str) -> ConflictRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT assertion_id, value_json, authority FROM world_assertions
                WHERE subject_id = ? AND property_id = ? AND state IN ('accepted', 'conflicted')
                ORDER BY authority DESC, valid_from DESC LIMIT 1""",
                (entity_id, property_id),
            ).fetchone()
            if row is None:
                return None
            current_value = json.loads(row[1])
            if current_value == value:
                return None
            return ConflictRecord(
                assertion_id=row[0],
                entity_id=entity_id,
                property_id=property_id,
                current_value=current_value,
                proposed_value=value,
                reason=f"New value '{value}' conflicts with accepted '{current_value}' (authority={row[2]:.2f})",
            )

    def _resolve_conflict(self, conflict: ConflictRecord, new_authority: float, existing_authority: float) -> AssertionState:
        if new_authority > existing_authority:
            return AssertionState.ACCEPTED
        elif new_authority == existing_authority:
            return AssertionState.CONFLICTED
        else:
            return AssertionState.PROPOSED

    def _promote_to_current(self, assertion_id: str, entity_id: str, property_id: str, value: TypedValue, confidence: float, authority: float, evidence_ids: tuple[str, ...], epistemic_kind: str, valid_from: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE world_assertions SET state = 'accepted' WHERE assertion_id = ?",
                (assertion_id,),
            )
            self._conn.commit()
        self._state.upsert_property(
            entity_id=entity_id,
            property_id=property_id,
            value=value,
            confidence=confidence,
            authority=authority,
            assertion_id=assertion_id,
            evidence_ids=evidence_ids,
            epistemic_kind=epistemic_kind,
            valid_from=valid_from,
        )

    def _promote_relation_to_current(self, assertion_id: str, source: str, relation: str, target: str, confidence: float, authority: float, evidence_ids: tuple[str, ...], epistemic_kind: str, valid_from: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE relation_assertions SET state = 'accepted' WHERE assertion_id = ?",
                (assertion_id,),
            )
            self._conn.commit()
        self._state.upsert_relation(
            source_entity_id=source,
            relation_type_id=relation,
            target_entity_id=target,
            confidence=confidence,
            authority=authority,
            assertion_id=assertion_id,
            evidence_ids=evidence_ids,
            epistemic_kind=epistemic_kind,
            valid_from=valid_from,
        )

    def ingest(self, delta: WorldDelta) -> list[ConflictRecord]:
        conflicts: list[ConflictRecord] = []
        for a in delta.assertions:
            entity_id = a["subject_id"]
            prop_id = a.get("property_id", "")
            value = a["value"]
            source_id = a.get("source_id", "unknown")
            confidence = a.get("confidence", 0.0)
            authority = a.get("authority", 0.5)
            evidence_ids = tuple(a.get("evidence_ids", ()))
            epistemic_kind = a.get("epistemic_kind", "observation")
            valid_from = a.get("valid_from", datetime.now(timezone.utc).isoformat())
            conflict = self._check_conflict(entity_id, prop_id, value.raw, source_id)
            if conflict is not None:
                conflicts.append(conflict)
            aid = self._insert_assertion(a, "property")
            state = self._resolve_conflict(conflict, authority, 0.5) if conflict else AssertionState.ACCEPTED
            if state == AssertionState.ACCEPTED:
                self._promote_to_current(aid, entity_id, prop_id, value, confidence, authority, evidence_ids, epistemic_kind, valid_from)
        for ra in delta.relation_assertions:
            aid = self._insert_assertion(ra, "relation")
            self._promote_relation_to_current(
                aid,
                ra["source_entity_id"],
                ra["relation_type_id"],
                ra["target_entity_id"],
                ra.get("confidence", 0.0),
                ra.get("authority", 0.5),
                tuple(ra.get("evidence_ids", ())),
                ra.get("epistemic_kind", "observation"),
                ra["valid_from"],
            )
        for ea in delta.events:
            aid = self._next_id("event")
            now = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._conn.execute(
                    """INSERT INTO event_assertions
                    (assertion_id, event_type_id, roles_json, epistemic_kind, source_id,
                     evidence_ids_json, confidence, valid_from, observed_at, recorded_at, state)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')""",
                    (
                        aid,
                        ea["event_type_id"],
                        json.dumps(ea.get("roles", {})),
                        ea.get("epistemic_kind", "observation"),
                        ea.get("source_id", ""),
                        json.dumps(list(ea.get("evidence_ids", ()))),
                        ea.get("confidence", 0.0),
                        ea.get("observed_at", now),
                        ea.get("observed_at", now),
                        now,
                    ),
                )
                self._conn.commit()
        return conflicts

    def get_entity_assertions(self, entity_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT assertion_id, property_id, value_json, value_type, epistemic_kind,
                source_id, evidence_ids_json, confidence, authority, valid_from, valid_to, state
                FROM world_assertions
                WHERE subject_id = ? AND valid_to IS NULL
                ORDER BY authority DESC, valid_from DESC""",
                (entity_id,),
            ).fetchall()
            return [
                {
                    "assertion_id": row[0], "property_id": row[1],
                    "value": json.loads(row[2]), "value_type": row[3],
                    "epistemic_kind": row[4], "source_id": row[5],
                    "evidence_ids": json.loads(row[6]),
                    "confidence": row[7], "authority": row[8],
                    "valid_from": row[9], "valid_to": row[10], "state": row[11],
                }
                for row in rows
            ]

    def get_assertion_history(self, entity_id: str, property_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT assertion_id, value_json, epistemic_kind, source_id, confidence,
                authority, valid_from, valid_to, state, recorded_at
                FROM world_assertions
                WHERE subject_id = ? AND property_id = ?
                ORDER BY valid_from DESC""",
                (entity_id, property_id),
            ).fetchall()
            return [
                {
                    "assertion_id": row[0], "value": json.loads(row[1]),
                    "epistemic_kind": row[2], "source_id": row[3],
                    "confidence": row[4], "authority": row[5],
                    "valid_from": row[6], "valid_to": row[7],
                    "state": row[8], "recorded_at": row[9],
                }
                for row in rows
            ]

    def get_conflicts(self) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT assertion_id, subject_id, property_id, value_json, epistemic_kind,
                source_id, confidence, authority
                FROM world_assertions WHERE state = 'conflicted'"""
            ).fetchall()
            return [
                {
                    "assertion_id": row[0], "entity_id": row[1],
                    "property_id": row[2], "value": json.loads(row[3]),
                    "epistemic_kind": row[4], "source_id": row[5],
                    "confidence": row[6], "authority": row[7],
                }
                for row in rows
            ]

    def all_entity_ids(self) -> list[str]:
        return self._state.all_entity_ids()

    def explain(self, entity_id: str, property_id: str) -> dict[str, Any]:
        return self._state.explain(entity_id, property_id)
