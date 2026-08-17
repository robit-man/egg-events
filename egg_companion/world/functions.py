"""Function and action registry.

Functions are pure/read/derive operations that do not mutate world state.
Actions are operations that mutate operational or external state.

The distinction matters for planning: the LLM can call functions freely
but actions require policy validation and approval workflows.
"""

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
    kind: str = "function"  # "function" for pure, "action" for mutating
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
                    kind TEXT NOT NULL DEFAULT 'function',
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
            # Pure functions (read/derive only)
            FunctionSpec("distance", "Distance", "Compute distance between entities",
                         kind="function",
                         input_schema={"entity_a": "str", "entity_b": "str"},
                         output_schema={"distance": "float"}),
            FunctionSpec("last_seen", "Last Seen", "Get when an entity was last observed",
                         kind="function",
                         input_schema={"entity": "str"},
                         output_schema={"timestamp": "str"}),
            FunctionSpec("currently_visible", "Currently Visible", "Check if an entity is currently visible",
                         kind="function",
                         input_schema={"entity": "str"},
                         output_schema={"visible": "bool"}),
            FunctionSpec("location_age", "Location Age", "Get age of last location update",
                         kind="function",
                         input_schema={"entity": "str"},
                         output_schema={"age_seconds": "float"}),
            FunctionSpec("people_in_place", "People in Place", "List people in a given place",
                         kind="function",
                         input_schema={"place": "str"},
                         output_schema={"people": "list"}),
            FunctionSpec("active_conflict_count", "Active Conflicts", "Count active conflicts for entity",
                         kind="function",
                         input_schema={"entity": "str"},
                         output_schema={"count": "int"}),
            FunctionSpec("identity_entropy", "Identity Entropy", "Compute identity uncertainty for a track",
                         kind="function",
                         input_schema={"track": "str"},
                         output_schema={"entropy": "float"}),

            # Mutating actions
            FunctionSpec("place_object", "Place Object", "Record that an object is at a location",
                         kind="action",
                         input_schema={"object_id": "str", "location": "str", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"]),
            FunctionSpec("track_person", "Track Person", "Record person presence and identity",
                         kind="action",
                         input_schema={"person_id": "str", "camera_id": "str", "bbox": "list[float]", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"]),
            FunctionSpec("recognize_face", "Recognize Face", "Record face identity claim",
                         kind="action",
                         input_schema={"person_id": "str", "face_embedding": "list[float]", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"]),
            FunctionSpec("record_speech", "Record Speech", "Record a speech utterance event",
                         kind="action",
                         input_schema={"speaker_id": "str", "transcript": "str", "confidence": "float"},
                         output_schema={"event_id": "str"},
                         side_effects=["event"]),
            FunctionSpec("record_behavior", "Record Behavior", "Record an observed behavior",
                         kind="action",
                         input_schema={"entity_id": "str", "behavior": "str", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"]),
            FunctionSpec("merge_entities", "Merge Entities", "Merge two entity identities",
                         kind="action",
                         input_schema={"keep_id": "str", "merged_ids": "list[str]", "reason": "str"},
                         output_schema={"success": "bool"},
                         side_effects=["identity_merge"],
                         required_evidence=["identity_claim"]),
            FunctionSpec("propose_ontology", "Propose Ontology", "Propose a new ontology type",
                         kind="action",
                         input_schema={"type_class": "str", "type_id": "str", "name": "str", "description": "str"},
                         output_schema={"proposal_id": "str"},
                         side_effects=["ontology_proposal"]),
            FunctionSpec("assign_function", "Assign Function", "Assign a functional role to an entity",
                         kind="action",
                         input_schema={"entity_id": "str", "function_id": "str", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"],
                         required_evidence=["user_correction"]),
            FunctionSpec("record_place", "Record Place", "Record or update a named place",
                         kind="action",
                         input_schema={"place_id": "str", "name": "str", "bounds": "dict", "confidence": "float"},
                         output_schema={"assertion_id": "str"},
                         side_effects=["assertion"],
                         required_evidence=["user_correction"]),
            FunctionSpec("resolve_conflict", "Resolve Conflict", "Manually resolve a conflict",
                         kind="action",
                         input_schema={"assertion_id": "str", "resolution": "str", "reason": "str"},
                         output_schema={"success": "bool"},
                         side_effects=["conflict_resolution"],
                         required_evidence=["user_correction"]),
        ]
        for fn in defaults:
            self.register(fn)

    def register(self, spec: FunctionSpec) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT OR REPLACE INTO function_registry
                (function_id, name, description, kind, input_schema_json, output_schema_json,
                 side_effects_json, required_evidence_json, max_age_seconds, version, enabled)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    spec.function_id, spec.name, spec.description, spec.kind,
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
                "SELECT name, description, kind, input_schema_json, output_schema_json, side_effects_json, required_evidence_json, max_age_seconds, version, enabled FROM function_registry WHERE function_id = ?",
                (function_id,),
            ).fetchone()
            if row is None:
                return None
            return FunctionSpec(
                function_id=function_id, name=row[0], description=row[1], kind=row[2],
                input_schema=json.loads(row[3]), output_schema=json.loads(row[4]),
                side_effects=json.loads(row[5]), required_evidence=json.loads(row[6]),
                max_age_seconds=row[7], version=row[8], enabled=bool(row[9]),
            )

    def list_pure_functions(self) -> list[FunctionSpec]:
        """Return only pure/read functions (not mutating actions)."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT function_id, name, description, enabled FROM function_registry WHERE kind = 'function' ORDER BY function_id"
            ).fetchall()
            return [
                FunctionSpec(function_id=r[0], name=r[1], description=r[2], kind="function", enabled=bool(r[3]))
                for r in rows
            ]

    def list_actions(self) -> list[FunctionSpec]:
        """Return only mutating actions."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT function_id, name, description, enabled FROM function_registry WHERE kind = 'action' ORDER BY function_id"
            ).fetchall()
            return [
                FunctionSpec(function_id=r[0], name=r[1], description=r[2], kind="action", enabled=bool(r[3]))
                for r in rows
            ]

    def list_all(self) -> list[FunctionSpec]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT function_id, name, description, kind, enabled FROM function_registry ORDER BY function_id"
            ).fetchall()
            return [
                FunctionSpec(function_id=r[0], name=r[1], description=r[2], kind=r[3], enabled=bool(r[4]))
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
