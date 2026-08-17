"""Ontology type registry.

Provides a central catalog of object types, property types, relation types,
event types, action types, function types, and source types.  The registry
is the authoritative schema for what Egg can represent in its world model.

Proposals and modifications are persisted to SQLite for durability.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any

from egg_companion.world.types import (
    ActionType,
    EventType,
    FunctionType,
    ObjectType,
    OntologyProposal,
    PropertyType,
    RelationType,
    SourceType,
    ValueType,
)


class OntologyRegistry:
    """Thread-safe ontology catalog with SQLite persistence for proposals."""

    def __init__(self, connection: sqlite3.Connection | None = None) -> None:
        self._lock = threading.RLock()
        self._conn = connection
        self._object_types: dict[str, ObjectType] = {}
        self._property_types: dict[str, PropertyType] = {}
        self._relation_types: dict[str, RelationType] = {}
        self._event_types: dict[str, EventType] = {}
        self._action_types: dict[str, ActionType] = {}
        self._function_types: dict[str, FunctionType] = {}
        self._source_types: dict[str, SourceType] = {}
        self._proposals: list[OntologyProposal] = []
        self._version: int = 1
        if self._conn is not None:
            self._ensure_tables()
            self._version = self._load_version()
        self._register_defaults()

    def _ensure_tables(self) -> None:
        with self._lock:
            self._conn.executescript(  # type: ignore[union-attr]
                """
                CREATE TABLE IF NOT EXISTS ontology_proposals (
                    proposal_id TEXT PRIMARY KEY,
                    kind TEXT NOT NULL,
                    definition_json TEXT NOT NULL DEFAULT '{}',
                    source TEXT NOT NULL DEFAULT 'llm_inference',
                    created_at TEXT NOT NULL,
                    validated INTEGER NOT NULL DEFAULT 0,
                    accepted INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS ontology_modifications (
                    modification_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL,
                    type_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    source TEXT NOT NULL DEFAULT 'system',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS ontology_version (
                    version INTEGER NOT NULL DEFAULT 1,
                    updated_at TEXT NOT NULL
                );
                """
            )

    def _load_version(self) -> int:
        with self._lock:
            row = self._conn.execute(  # type: ignore[union-attr]
                "SELECT COALESCE(MAX(version), 1) FROM ontology_version"
            ).fetchone()
            return int(row[0]) if row else 1

    def _save_version(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(  # type: ignore[union-attr]
                "INSERT INTO ontology_version (version, updated_at) VALUES (?, ?)",
                (self._version, now),
            )
            self._conn.commit()  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register_object_type(self, ot: ObjectType) -> None:
        with self._lock:
            self._object_types[ot.id] = ot

    def register_property_type(self, pt: PropertyType) -> None:
        with self._lock:
            self._property_types[pt.id] = pt

    def register_relation_type(self, rt: RelationType) -> None:
        with self._lock:
            self._relation_types[rt.id] = rt

    def register_event_type(self, et: EventType) -> None:
        with self._lock:
            self._event_types[et.id] = et

    def register_action_type(self, at: ActionType) -> None:
        with self._lock:
            self._action_types[at.id] = at

    def register_function_type(self, ft: FunctionType) -> None:
        with self._lock:
            self._function_types[ft.id] = ft

    def register_source_type(self, st: SourceType) -> None:
        with self._lock:
            self._source_types[st.id] = st

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def object_type_exists(self, type_id: str) -> bool:
        return type_id in self._object_types

    def get_object_type(self, type_id: str) -> ObjectType | None:
        return self._object_types.get(type_id)

    def get_property_type(self, type_id: str) -> PropertyType | None:
        return self._property_types.get(type_id)

    def get_relation_type(self, type_id: str) -> RelationType | None:
        return self._relation_types.get(type_id)

    def get_event_type(self, type_id: str) -> EventType | None:
        return self._event_types.get(type_id)

    def get_action_type(self, type_id: str) -> ActionType | None:
        return self._action_types.get(type_id)

    def get_function_type(self, type_id: str) -> FunctionType | None:
        return self._function_types.get(type_id)

    def get_source_type(self, type_id: str) -> SourceType | None:
        return self._source_types.get(type_id)

    def list_object_types(self) -> list[ObjectType]:
        return list(self._object_types.values())

    def list_property_types(self) -> list[PropertyType]:
        return list(self._property_types.values())

    def list_relation_types(self) -> list[RelationType]:
        return list(self._relation_types.values())

    def list_action_types(self) -> list[ActionType]:
        return list(self._action_types.values())

    @property
    def version(self) -> int:
        return self._version

    # ------------------------------------------------------------------
    # Proposals (LLM-generated, never directly applied)
    # ------------------------------------------------------------------

    def submit_proposal(self, proposal: OntologyProposal) -> None:
        with self._lock:
            self._proposals.append(proposal)
            if self._conn is not None:
                self._conn.execute(
                    """INSERT OR REPLACE INTO ontology_proposals
                    (proposal_id, kind, definition_json, source, created_at, validated, accepted)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        proposal.proposal_id,
                        proposal.kind,
                        json.dumps(proposal.definition, default=str),
                        proposal.source,
                        proposal.created_at.isoformat() if isinstance(proposal.created_at, datetime) else str(proposal.created_at),
                        int(proposal.validated),
                        int(proposal.accepted),
                    ),
                )
                self._conn.commit()

    def pending_proposals(self) -> list[OntologyProposal]:
        return [p for p in self._proposals if not p.validated]

    def record_modification(
        self,
        kind: str,
        type_id: str,
        field_name: str,
        old_value: Any,
        new_value: Any,
        source: str = "system",
    ) -> None:
        """Record an ontology modification for audit trail."""
        if self._conn is None:
            return
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                """INSERT INTO ontology_modifications
                (kind, type_id, field_name, old_value, new_value, source, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    kind, type_id, field_name,
                    json.dumps(old_value, default=str),
                    json.dumps(new_value, default=str),
                    source, now,
                ),
            )
            self._conn.commit()

    def describe(self, type_id: str) -> dict[str, Any] | None:
        for registry in (
            self._object_types,
            self._property_types,
            self._relation_types,
            self._event_types,
            self._action_types,
            self._function_types,
            self._source_types,
        ):
            item = registry.get(type_id)
            if item is not None:
                return {"type_id": type_id, **{k: v for k, v in item.__dict__.items()}}
        return None

    # ------------------------------------------------------------------
    # Default ontology
    # ------------------------------------------------------------------

    def _register_defaults(self) -> None:
        for ot in _DEFAULT_OBJECT_TYPES:
            self._object_types[ot.id] = ot
        for pt in _DEFAULT_PROPERTY_TYPES:
            self._property_types[pt.id] = pt
        for rt in _DEFAULT_RELATION_TYPES:
            self._relation_types[rt.id] = rt
        for et in _DEFAULT_EVENT_TYPES:
            self._event_types[et.id] = et
        for at in _DEFAULT_ACTION_TYPES:
            self._action_types[at.id] = at
        for ft in _DEFAULT_FUNCTION_TYPES:
            self._function_types[ft.id] = ft
        for st in _DEFAULT_SOURCE_TYPES:
            self._source_types[st.id] = st


# ======================================================================
# Default ontology definitions
# ======================================================================

_DEFAULT_OBJECT_TYPES = [
    ObjectType(id="person", identity_strategy="persistent_instance", temporal_mode="current_state",
               expected_properties=("label", "preferred_name", "current_location", "last_seen")),
    ObjectType(id="physical_object", identity_strategy="persistent_instance", temporal_mode="current_state",
               expected_properties=("label", "category", "current_location", "last_seen")),
    ObjectType(id="surface", identity_strategy="persistent_instance", temporal_mode="current_state",
               expected_properties=("label", "location")),
    ObjectType(id="place", identity_strategy="named_region", temporal_mode="current_state",
               expected_properties=("label", "bounds")),
    ObjectType(id="camera_view", identity_strategy="fixed_id", temporal_mode="current_state",
               expected_properties=("camera_id", "field_of_view")),
    ObjectType(id="sound_event", identity_strategy="ephemeral", temporal_mode="event_time",
               expected_properties=("source_direction", "label")),
    ObjectType(id="conversation_turn", identity_strategy="ephemeral", temporal_mode="event_time",
               expected_properties=("transcript", "role")),
]

_DEFAULT_PROPERTY_TYPES = [
    PropertyType(id="label", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="preferred_name", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="category", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="current_location", value_type=ValueType.ENTITY_REF, cardinality="one",
                 volatility="dynamic", stale_after=300.0, decay_model="linear"),
    PropertyType(id="last_seen", value_type=ValueType.DATETIME, cardinality="one",
                 volatility="dynamic", stale_after=300.0, decay_model="exponential"),
    PropertyType(id="location", value_type=ValueType.GEOMETRY, cardinality="one",
                 volatility="dynamic", stale_after=300.0, decay_model="linear"),
    PropertyType(id="behavior", value_type=ValueType.STRING, cardinality="one",
                 volatility="dynamic", stale_after=30.0, decay_model="linear"),
    PropertyType(id="bbox", value_type=ValueType.GEOMETRY, cardinality="one",
                 volatility="dynamic", stale_after=5.0, decay_model="linear"),
    PropertyType(id="confidence_score", value_type=ValueType.FLOAT, cardinality="one",
                 stale_after=300.0),
    PropertyType(id="transcript", value_type=ValueType.STRING, cardinality="one",
                 stale_after=600.0),
    PropertyType(id="role", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="source_direction", value_type=ValueType.FLOAT, cardinality="one",
                 unit="degrees", stale_after=300.0),
    PropertyType(id="camera_id", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="field_of_view", value_type=ValueType.JSON, cardinality="one",
                 stale_after=None),
    PropertyType(id="bounds", value_type=ValueType.JSON, cardinality="one",
                 stale_after=None),
    PropertyType(id="label_source", value_type=ValueType.STRING, cardinality="one",
                 stale_after=None),
    PropertyType(id="observability", value_type=ValueType.STRING, cardinality="one",
                 volatility="dynamic", stale_after=300.0, decay_model="linear"),
    PropertyType(id="visible_text", value_type=ValueType.STRING, cardinality="one",
                 volatility="dynamic", stale_after=60.0, decay_model="linear"),
]

_DEFAULT_RELATION_TYPES = [
    RelationType(id="holds", domain=("person",), range=("physical_object",),
                 persistence="observation_dependent", stale_after=5.0),
    RelationType(id="near", domain=(), range=(),
                 symmetric=True, persistence="observation_dependent", stale_after=3.0),
    RelationType(id="inside", domain=(), range=("place", "surface"),
                 persistence="observation_dependent", stale_after=10.0),
    RelationType(id="on_top_of", domain=("physical_object",), range=("surface",),
                 persistence="observation_dependent", stale_after=30.0),
    RelationType(id="visible_from", domain=(), range=("camera_view",),
                 persistence="observation_dependent", stale_after=2.0),
    RelationType(id="located_at", domain=(), range=("place", "surface"),
                 persistence="observation_dependent", stale_after=30.0),
    RelationType(id="speaking_to", domain=("person",), range=("person",),
                 persistence="observation_dependent", stale_after=5.0),
    RelationType(id="owns", domain=("person",), range=("physical_object",),
                 persistence="persistent_until_contradicted"),
    RelationType(id="same_as", domain=(), range=(),
                 persistence="hypothesis"),
    RelationType(id="co_observed_with", domain=(), range=(),
                 symmetric=True, persistence="evidence_association"),
]

_DEFAULT_EVENT_TYPES = [
    EventType(id="appearance", roles={"entity": "entity"}),
    EventType(id="disappearance", roles={"entity": "entity"}),
    EventType(id="movement", roles={"entity": "entity", "from": "location", "to": "location"}),
    EventType(id="speech_utterance", roles={"speaker": "person", "transcript": "string"}),
    EventType(id="object_transfer", roles={"object": "physical_object", "giver": "person",
                                            "receiver": "person"}),
    EventType(id="person_entry", roles={"person": "person", "place": "place"}),
    EventType(id="person_exit", roles={"person": "person", "place": "place"}),
    EventType(id="identity_correction", roles={"entity": "entity", "old_label": "string",
                                                "new_label": "string"}),
    EventType(id="label_correction", roles={"entity": "entity", "old_label": "string",
                                             "new_label": "string"}),
    EventType(id="conversation_turn", roles={"speaker": "person", "transcript": "string"}),
    EventType(id="ocr_detection", roles={"target": "entity", "transcript": "string"}),
]

_DEFAULT_ACTION_TYPES = [
    ActionType(id="speak", actor="agent", parameters={"text": "string"}),
    ActionType(id="ask_clarifying_question", actor="agent", parameters={"text": "string"}),
    ActionType(id="focus_camera", actor="agent", parameters={"camera_id": "string"}),
    ActionType(id="inspect_entity", actor="agent", parameters={"entity_id": "entity_ref"}),
    ActionType(id="ask_identity_clarification", actor="agent",
               parameters={"candidate_entity": "entity_ref"}, approval_required=False),
]

_DEFAULT_FUNCTION_TYPES = [
    FunctionType(id="distance", inputs=("entity_a", "entity_b"), output_type=ValueType.FLOAT),
    FunctionType(id="last_seen", inputs=("entity",), output_type=ValueType.DATETIME),
    FunctionType(id="currently_visible", inputs=("entity",), output_type=ValueType.BOOLEAN),
    FunctionType(id="location_age", inputs=("entity",), output_type=ValueType.DURATION),
    FunctionType(id="people_in_place", inputs=("place",), output_type=ValueType.JSON),
    FunctionType(id="active_conflict_count", inputs=("entity",), output_type=ValueType.INTEGER),
    FunctionType(id="identity_entropy", inputs=("track",), output_type=ValueType.FLOAT),
]

_DEFAULT_SOURCE_TYPES = [
    SourceType(id="camera", authority_class="observation"),
    SourceType(id="detector", authority_class="observation"),
    SourceType(id="face_matcher", authority_class="hypothesis"),
    SourceType(id="object_tracker", authority_class="observation"),
    SourceType(id="ornith_vlm", authority_class="inference"),
    SourceType(id="microphone", authority_class="observation"),
    SourceType(id="asr", authority_class="inference"),
    SourceType(id="respeaker_doa", authority_class="observation"),
    SourceType(id="user_statement", authority_class="claim"),
    SourceType(id="user_correction", authority_class="correction"),
    SourceType(id="system_clock", authority_class="axiom"),
    SourceType(id="runtime", authority_class="axiom"),
    SourceType(id="derived_function", authority_class="derived"),
    SourceType(id="llm_inference", authority_class="inference"),
    SourceType(id="external_api", authority_class="claim"),
]
