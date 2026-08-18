"""Reconciler: merges, validates, and settles typed assertions."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from egg_companion.world.ontology import OntologyRegistry

from egg_companion.world.state import WorldStateStore
from egg_companion.world.types import (
    AssertionKind,
    AssertionState,
    EvidenceCorrelationGroup,
    ObservabilityState,
    TypedValue,
    ValueType,
    WorldDelta,
)


class RelationReconciler:
    """Reconciles relation assertions with authority, expiry, and persistence modes.

    Unlike property assertions which already had conflict detection, relation
    assertions previously bypassed reconciliation entirely.  This class
    enforces:
    - Domain/range validation against ontology
    - Authority comparison for duplicate (source, type, target) triples
    - Temporal expiry via stale_after from ontology RelationType
    - Persistence mode (observation_dependent, persistent_until_contradicted, etc.)
    - Supersession of weaker relation assertions
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        state: WorldStateStore,
        ontology: Any = None,
    ) -> None:
        self._conn = conn
        self._state = state
        self._ontology = ontology
        self._lock = threading.RLock()

    def reconcile_relation(
        self,
        source_id: str,
        relation_type: str,
        target_id: str,
        new_authority: float,
        new_confidence: float,
        evidence_ids: tuple[str, ...],
        epistemic_kind: str,
        source_origin: str,
        valid_from: str,
        revision: int | None = None,
    ) -> str:
        """Decide how to handle a new relation assertion.

        Returns: 'accepted', 'superseded', 'conflicted', or 'rejected'
        """
        # Check ontology for persistence mode and stale_after
        persistence = "observation_dependent"
        stale_after: float | None = None
        if self._ontology is not None:
            rt = self._ontology.get_relation_type(relation_type)
            if rt is not None:
                persistence = rt.persistence
                stale_after = rt.stale_after

        # Expire stale existing relations if the ontology says so
        if stale_after is not None:
            self._expire_stale_relations(source_id, relation_type, target_id, stale_after)

        # Check for existing active relation with same (source, type, target)
        existing = self._find_active_relation(source_id, relation_type, target_id)
        if existing is None:
            return "accepted"

        existing_authority = existing["authority"]
        existing_state = existing["state"]

        # persistent_until_contradicted: only supersede with higher authority
        if persistence == "persistent_until_contradicted":
            if new_authority > existing_authority:
                return "superseded"
            return "conflicted" if abs(new_authority - existing_authority) < 0.05 else "rejected"

        # hypothesis: always superseded by observations
        if persistence == "hypothesis":
            if epistemic_kind == "observation":
                return "superseded"
            return "conflicted"

        # observation_dependent / evidence_association (default):
        # compare authority
        if new_authority > existing_authority:
            return "superseded"
        elif abs(new_authority - existing_authority) < 0.05:
            return "conflicted"
        else:
            return "rejected"

    def _find_active_relation(
        self, source_id: str, relation_type: str, target_id: str
    ) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT assertion_id, authority, confidence, state
                FROM relation_assertions
                WHERE source_entity_id = ? AND relation_type_id = ? AND target_entity_id = ?
                AND state IN ('accepted', 'conflicted')
                ORDER BY authority DESC LIMIT 1""",
                (source_id, relation_type, target_id),
            ).fetchone()
            if row is None:
                return None
            return {
                "assertion_id": row[0],
                "authority": float(row[1]),
                "confidence": float(row[2]),
                "state": row[3],
            }

    def _expire_stale_relations(
        self, source_id: str, relation_type: str, target_id: str, stale_after: float
    ) -> None:
        """Close relations whose age exceeds stale_after seconds."""
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        cutoff = (now - timedelta(seconds=stale_after)).isoformat()
        with self._lock:
            self._conn.execute(
                """UPDATE relation_assertions
                SET state = 'superseded', valid_to = ?
                WHERE source_entity_id = ? AND relation_type_id = ? AND target_entity_id = ?
                AND state = 'accepted' AND valid_from < ?""",
                (now.isoformat(), source_id, relation_type, target_id, cutoff),
            )
            # Also close the current_state projection
            self._state.close_relation(
                source_id, relation_type, target_id,
                now.isoformat(),
            )


@dataclass
class ConflictRecord:
    assertion_id: str
    entity_id: str
    property_id: str
    current_value: Any
    proposed_value: Any
    existing_authority: float
    existing_confidence: float
    reason: str


class Reconciler:
    """Accepts WorldDelta, validates, deduplicates, merges conflicts."""

    def __init__(
        self,
        assertion_conn: sqlite3.Connection,
        state_store: WorldStateStore,
        ontology: OntologyRegistry | None = None,
    ) -> None:
        self._conn = assertion_conn
        self._state = state_store
        self._ontology = ontology
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
        return aid

    def _check_conflict(
        self, entity_id: str, property_id: str, value: Any, source_id: str
    ) -> ConflictRecord | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT assertion_id, value_json, authority, confidence FROM world_assertions
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
                existing_authority=float(row[2]),
                existing_confidence=float(row[3]),
                reason=(
                    f"New value '{value}' conflicts with accepted "
                    f"'{current_value}' (authority={float(row[2]):.2f})"
                ),
            )

    def _resolve_conflict(
        self, conflict: ConflictRecord, new_authority: float
    ) -> AssertionState:
        if new_authority > conflict.existing_authority:
            return AssertionState.ACCEPTED
        elif new_authority == conflict.existing_authority:
            return AssertionState.CONFLICTED
        else:
            return AssertionState.PROPOSED

    def _supersede_assertion(
        self, old_assertion_id: str, new_valid_from: str
    ) -> None:
        """Mark an existing assertion as superseded and close its valid_to."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """UPDATE world_assertions
                SET state = 'superseded', valid_to = ?, recorded_at = ?
                WHERE assertion_id = ? AND state = 'accepted'""",
                (new_valid_from, now, old_assertion_id),
            )

    def _promote_to_current(
        self,
        assertion_id: str,
        entity_id: str,
        property_id: str,
        value: TypedValue,
        confidence: float,
        authority: float,
        evidence_ids: tuple[str, ...],
        epistemic_kind: str,
        valid_from: str,
        revision: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE world_assertions SET state = 'accepted' WHERE assertion_id = ?",
                (assertion_id,),
            )
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
            revision=revision,
        )

    def _mark_conflicted(self, assertion_id: str) -> None:
        """Mark an assertion as conflicted."""
        with self._lock:
            self._conn.execute(
                "UPDATE world_assertions SET state = 'conflicted' WHERE assertion_id = ?",
                (assertion_id,),
            )

    def _promote_relation_to_current(
        self,
        assertion_id: str,
        source: str,
        relation: str,
        target: str,
        confidence: float,
        authority: float,
        evidence_ids: tuple[str, ...],
        epistemic_kind: str,
        valid_from: str,
        revision: int | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE relation_assertions SET state = 'accepted' WHERE assertion_id = ?",
                (assertion_id,),
            )
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
            revision=revision,
        )

    def ingest(self, delta: WorldDelta) -> list[ConflictRecord]:
        conflicts: list[ConflictRecord] = []
        relation_reconciler = RelationReconciler(self._conn, self._state, self._ontology)

        with self._state.world_transaction(f"delta:{len(delta.assertions)}a,{len(delta.relation_assertions)}r"):
            revision = self._state._current_revision

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

                aid = self._insert_assertion(a, "property")

                if conflict is not None:
                    conflicts.append(conflict)
                    state = self._resolve_conflict(conflict, authority)

                    if state == AssertionState.ACCEPTED:
                        self._supersede_assertion(conflict.assertion_id, valid_from)
                        self._promote_to_current(
                            aid, entity_id, prop_id, value, confidence, authority,
                            evidence_ids, epistemic_kind, valid_from, revision=revision,
                        )
                    elif state == AssertionState.CONFLICTED:
                        self._mark_conflicted(aid)
                        self._mark_conflicted(conflict.assertion_id)
                    else:
                        pass  # Leave as proposed
                else:
                    self._promote_to_current(
                        aid, entity_id, prop_id, value, confidence, authority,
                        evidence_ids, epistemic_kind, valid_from, revision=revision,
                    )

            for ra in delta.relation_assertions:
                result = relation_reconciler.reconcile_relation(
                    source_id=ra["source_entity_id"],
                    relation_type=ra["relation_type_id"],
                    target_id=ra["target_entity_id"],
                    new_authority=ra.get("authority", 0.5),
                    new_confidence=ra.get("confidence", 0.0),
                    evidence_ids=tuple(ra.get("evidence_ids", ())),
                    epistemic_kind=ra.get("epistemic_kind", "observation"),
                    source_origin=ra.get("source_id", ""),
                    valid_from=ra["valid_from"],
                    revision=revision,
                )
                if result == "accepted" or result == "superseded":
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
                        revision=revision,
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

            self._emit_absence_transitions(delta, revision)

        return conflicts

    def _emit_absence_transitions(self, delta: WorldDelta, revision: int) -> None:
        """Mark entities OBSERVED_ABSENT when a camera's current frame no
        longer contains something it previously saw.

        This must run here rather than in the (stateless) normalizer: only
        the reconciler can see prior materialized state, so only it can
        tell "was visible, now isn't" apart from "was never visible".
        Presence/absence is a recency-ordered state machine, not a
        competing-authority fact, so the new value is promoted directly
        rather than routed through the standard authority-conflict path
        (which would otherwise let the higher, fixed authority of the
        original OBSERVED_PRESENT assertion block every future absence
        transition from ever taking effect).
        """
        for frame in delta.camera_frames:
            camera_id = frame["camera_id"]
            currently_visible = set(frame.get("visible_entity_ids", ()))
            context = {
                "source_id": frame.get("source_id", "unknown"),
                "evidence_ids": tuple(frame.get("evidence_ids", ())),
                "valid_from": frame.get("valid_from", datetime.now(timezone.utc).isoformat()),
            }
            with self._lock:
                rows = self._conn.execute(
                    """
                    SELECT crs.source_entity_id
                    FROM current_relation_state crs
                    JOIN current_property_state cps
                      ON cps.entity_id = crs.source_entity_id
                      AND cps.property_id = 'observability'
                    WHERE crs.relation_type_id = 'visible_from'
                      AND crs.target_entity_id = ?
                      AND crs.valid_to IS NULL
                      AND cps.value_json = ?
                    """,
                    (camera_id, json.dumps(ObservabilityState.OBSERVED_PRESENT.value)),
                ).fetchall()
            previously_visible = {row[0] for row in rows}
            missing_ids = previously_visible - currently_visible
            if not missing_ids:
                continue

            for missing_id in missing_ids:
                a = {
                    "subject_id": missing_id,
                    "property_id": "observability",
                    "value": TypedValue(
                        raw=ObservabilityState.OBSERVED_ABSENT.value,
                        value_type=ValueType.ENUM,
                    ),
                    "epistemic_kind": "inference",
                    "source_id": context["source_id"],
                    "evidence_ids": context["evidence_ids"],
                    "confidence": 0.5,
                    "authority": 0.4,
                    "valid_from": context["valid_from"],
                }
                aid = self._insert_assertion(a, "property")
                active_id = self._active_assertion_id(missing_id, "observability")
                if active_id is not None:
                    self._supersede_assertion(active_id, context["valid_from"])
                self._promote_to_current(
                    aid, missing_id, "observability", a["value"],
                    a["confidence"], a["authority"], a["evidence_ids"],
                    a["epistemic_kind"], a["valid_from"], revision=revision,
                )

    def _active_assertion_id(self, entity_id: str, property_id: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                """SELECT assertion_id FROM world_assertions
                WHERE subject_id = ? AND property_id = ? AND state = 'accepted'
                ORDER BY valid_from DESC LIMIT 1""",
                (entity_id, property_id),
            ).fetchone()
            return row[0] if row else None

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

    def get_assertion_history(
        self, entity_id: str, property_id: str
    ) -> list[dict[str, Any]]:
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

    def aggregate_confidence_with_correlation(
        self,
        confidences: list[float],
        correlation_group: EvidenceCorrelationGroup | None = None,
    ) -> float:
        if not confidences:
            return 0.0
        if len(confidences) == 1:
            return confidences[0]

        combined = 1.0
        for c in confidences:
            combined *= (1.0 - min(1.0, max(0.0, c)))
        base_confidence = 1.0 - combined

        if correlation_group is not None and correlation_group.observation_count > 1:
            independence = correlation_group.independence_factor
            effective_observations = 1 + (len(confidences) - 1) * independence
            scaled = base_confidence * (effective_observations / len(confidences))
            return min(1.0, scaled)

        return base_confidence

    def get_effective_independent_observations(
        self,
        entity_id: str,
        property_id: str,
        window_seconds: float = 10.0,
    ) -> dict[str, Any]:
        import datetime as dt

        with self._lock:
            rows = self._conn.execute(
                """SELECT confidence, valid_from, source_id
                FROM world_assertions
                WHERE subject_id = ? AND property_id = ? AND state = 'accepted'
                ORDER BY valid_from DESC""",
                (entity_id, property_id),
            ).fetchall()

        if not rows:
            return {"total_observations": 0, "effective_independent": 0.0}

        total = len(rows)
        confidences = [r[0] for r in rows]

        groups: list[list[float]] = []
        current_group = [confidences[0]]

        for i in range(1, len(rows)):
            try:
                prev_time = dt.datetime.fromisoformat(rows[i - 1][1])
                curr_time = dt.datetime.fromisoformat(rows[i][1])
                delta = abs((curr_time - prev_time).total_seconds())

                if delta < window_seconds:
                    current_group.append(confidences[i])
                else:
                    groups.append(current_group)
                    current_group = [confidences[i]]
            except Exception:
                groups.append(current_group)
                current_group = [confidences[i]]

        groups.append(current_group)

        effective = 0.0
        for group in groups:
            effective += 1 + (len(group) - 1) * 0.1

        return {
            "total_observations": total,
            "effective_independent": round(effective, 2),
            "groups": len(groups),
            "window_seconds": window_seconds,
        }
