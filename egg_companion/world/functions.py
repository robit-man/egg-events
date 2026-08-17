"""Function registry: typed world model update operations."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FunctionSpec:
    function_id: str
    name: str
    description: str
    input_schema: dict[str, Any] = field(default_factory=dict)
    output_schema: dict[str, Any] = field(default_factory=dict)
    side_effects: list[str] = field(default_factory=list)
    required_evidence: list[str] = field(default_factory=list)
    max_age_seconds: float = float("inf")
    version: str = "1.0.0"
    enabled: bool = True


@dataclass
class FunctionCall:
    call_id: str
    function_id: str
    args: dict[str, Any]
    result: dict[str, Any] | None = None
    success: bool = False
    error: str | None = None
    evidence_ids: list[str] = field(default_factory=list)
    recorded_at: str = ""


class FunctionRegistry:
    """Registry of typed operations that can update the world model."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self._conn = connection
        self._lock = threading.RLock()
        self._ensure_tables()
        self._register_defaults()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS function_registry (
                    function_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    input_schema_json TEXT NOT NULL DEFAULT '{}',
                    output_schema_json TEXT NOT NULL DEFAULT '{}',
                    side_effects_json TEXT NOT NULL DEFAULT '[]',
                    required_evidence_json TEXT NOT NULL DEFAULT '[]',
                    max_age_seconds REAL NOT NULL DEFAULT 999999999,
                    version TEXT NOT NULL DEFAULT '1.0.0',
                    enabled INTEGER NOT NULL DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS function_calls (
                    call_id TEXT PRIMARY KEY,
                    function_id TEXT NOT NULL,
                    args_json TEXT NOT NULL DEFAULT '{}',
                    result_json TEXT,
                    success INTEGER NOT NULL DEFAULT 0,
                    error TEXT,
                    evidence_ids_json TEXT NOT NULL DEFAULT '[]',
                    recorded_at TEXT NOT NULL,
                    FOREIGN KEY (function_id) REFERENCES function_registry(function_id)
                );
                """
            )

    def _register_defaults(self) -> None:
        defaults = [
            FunctionSpec("place_object", "Place Object", "Record that an object is at a location", {"object_id": "str", "location": "str", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["detection_with_bbox"]),
            FunctionSpec("track_person", "Track Person", "Record person presence and identity", {"person_id": "str", "camera_id": "str", "bbox": "list[float]", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["detection_with_identity"]),
            FunctionSpec("recognize_face", "Recognize Face", "Record face identity claim", {"person_id": "str", "face_embedding": "list[float]", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["face_detection"]),
            FunctionSpec("record_speech", "Record Speech", "Record a speech utterance event", {"speaker_id": "str", "transcript": "str", "confidence": "float"}, {"event_id": "str"}, ["event"], ["asr_transcription"]),
            FunctionSpec("record_behavior", "Record Behavior", "Record an observed behavior", {"entity_id": "str", "behavior": "str", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["behavior_detection"]),
            FunctionSpec("merge_entities", "Merge Entities", "Merge two entity identities", {"keep_id": "str", "merged_ids": "list[str]", "reason": "str"}, {"success": "bool"}, ["identity_merge"], ["identity_claim"]),
            FunctionSpec("propose_ontology", "Propose Ontology", "Propose a new ontology type", {"type_class": "str", "type_id": "str", "name": "str", "description": "str"}, {"proposal_id": "str"}, ["ontology_proposal"], []),
            FunctionSpec("assign_function", "Assign Function", "Assign a functional role to an entity", {"entity_id": "str", "function_id": "str", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["user_correction"]),
            FunctionSpec("record_place", "Record Place", "Record or update a named place", {"place_id": "str", "name": "str", "bounds": "dict", "confidence": "float"}, {"assertion_id": "str"}, ["assertion"], ["user_correction"]),
            FunctionSpec("resolve_conflict", "Resolve Conflict", "Manually resolve a conflict", {"assertion_id": "str", "resolution": "str", "reason": "str"}, {"success": "bool"}, ["conflict_resolution"], ["user_correction"]),
        ]
        for fn in defaults:
            self.register(fn)

    def register(self, spec: FunctionSpec) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO function_registry
                (function_id, name, description, input_schema_json, output_schema_json,
                 side_effects_json, required_evidence_json, max_age_seconds, version, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    spec.function_id, spec.name, spec.description,
                    json.dumps(spec.input_schema, default=str),
                    json.dumps(spec.output_schema, default=str),
                    json.dumps(spec.side_effects),
                    json.dumps(spec.required_evidence),
                    spec.max_age_seconds, spec.version, int(spec.enabled),
                ),
            )
            self._conn.commit()

    def get(self, function_id: str) -> FunctionSpec | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT name, description, input_schema_json, output_schema_json, side_effects_json, required_evidence_json, max_age_seconds, version, enabled FROM function_registry WHERE function_id = ?",
                (function_id,),
            ).fetchone()
            if row is None:
                return None
            return FunctionSpec(
                function_id=function_id, name=row[0], description=row[1],
                input_schema=json.loads(row[2]), output_schema=json.loads(row[3]),
                side_effects=json.loads(row[4]), required_evidence=json.loads(row[5]),
                max_age_seconds=row[6], version=row[7], enabled=bool(row[8]),
            )

    def list_all(self) -> list[FunctionSpec]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT function_id, name, description, enabled FROM function_registry ORDER BY function_id"
            ).fetchall()
            return [
                FunctionSpec(function_id=r[0], name=r[1], description=r[2], enabled=bool(r[3]))
                for r in rows
            ]

    def log_call(self, call: FunctionCall) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO function_calls
                (call_id, function_id, args_json, result_json, success, error, evidence_ids_json, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    call.call_id, call.function_id,
                    json.dumps(call.args, default=str),
                    json.dumps(call.result, default=str) if call.result else None,
                    int(call.success), call.error,
                    json.dumps(call.evidence_ids),
                    call.recorded_at,
                ),
            )
            self._conn.commit()

    def recent_calls(self, function_id: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            if function_id:
                rows = self._conn.execute(
                    "SELECT call_id, function_id, args_json, result_json, success, error, evidence_ids_json, recorded_at FROM function_calls WHERE function_id = ? ORDER BY recorded_at DESC LIMIT ?",
                    (function_id, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT call_id, function_id, args_json, result_json, success, error, evidence_ids_json, recorded_at FROM function_calls ORDER BY recorded_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [
                {
                    "call_id": r[0], "function_id": r[1],
                    "args": json.loads(r[2]),
                    "result": json.loads(r[3]) if r[3] else None,
                    "success": bool(r[4]), "error": r[5],
                    "evidence_ids": json.loads(r[6]),
                    "recorded_at": r[7],
                }
                for r in rows
            ]
