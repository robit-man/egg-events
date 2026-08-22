"""Tests for the operational world model."""

import json
import sqlite3
import tempfile
from datetime import datetime, timezone

import pytest

from egg_companion.world import (
    BBox2D,
    BitemporalInterval,
    CognitiveContext,
    EpistemicKind,
    EventStore,
    MetricsCollector,
    OntologyRegistry,
    ObservationNormalizer,
    PolicyValidator,
    Reconciler,
    SpatialReasoner,
    SpatialState,
    TypedValue,
    ValueType,
    WorldGraphStore,
    WorldQuery,
    WorldStateStore,
)
from egg_companion.world.types import WorldDelta
from egg_companion.world.spatial import SpatialRelation
from egg_companion.world.assertions import EventAssertion, RelationAssertion, WorldAssertion
from egg_companion.world.identity import IdentityGraph
from egg_companion.world.relations import WorldEdge
from egg_companion.world.sources import AuthorityPolicy, SourceRecord
from egg_companion.world.temporal import (
    BitemporalInterval,
    freshness_seconds,
    make_bounded_interval,
    make_current_interval,
    utcnow,
)
from egg_companion.world.types import (
    ActionType,
    AssertionKind,
    AssertionState,
    CoordinateFrame,
    ObjectType,
    PropertyType,
    RelationType,
    SourceType,
)


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    yield conn
    conn.close()


@pytest.fixture
def world_stores(db):
    state = WorldStateStore(db)
    graph = WorldGraphStore(db)
    identity = IdentityGraph(db)
    reconciler = Reconciler(db, state)
    ontology = OntologyRegistry()
    events = EventStore(db)
    normalizer = ObservationNormalizer()
    policy = PolicyValidator(db)
    metrics = MetricsCollector(db)
    query = WorldQuery(state, graph, identity, reconciler=reconciler)
    context = CognitiveContext(query)
    return {
        "state": state,
        "graph": graph,
        "identity": identity,
        "reconciler": reconciler,
        "ontology": ontology,
        "events": events,
        "normalizer": normalizer,
        "policy": policy,
        "metrics": metrics,
        "query": query,
        "context": context,
        "db": db,
    }


class TestOntologyRegistry:
    def test_default_object_types(self, world_stores):
        ontology = world_stores["ontology"]
        obj = ontology.get_object_type("person")
        assert obj is not None
        assert obj.id == "person"

    def test_default_property_types(self, world_stores):
        ontology = world_stores["ontology"]
        prop = ontology.get_property_type("preferred_name")
        assert prop is not None

    def test_default_relation_types(self, world_stores):
        ontology = world_stores["ontology"]
        rel = ontology.get_relation_type("co_observed_with")
        assert rel is not None

    def test_default_event_types(self, world_stores):
        ontology = world_stores["ontology"]
        evt = ontology.get_event_type("speech_utterance")
        assert evt is not None

    def test_default_action_types(self, world_stores):
        ontology = world_stores["ontology"]
        act = ontology.get_action_type("speak")
        assert act is not None

    def test_default_function_types(self, world_stores):
        ontology = world_stores["ontology"]
        fn = ontology.get_function_type("distance")
        assert fn is not None

    def test_default_source_types(self, world_stores):
        ontology = world_stores["ontology"]
        src = ontology.get_source_type("camera")
        assert src is not None

    def test_list_object_types(self, world_stores):
        ontology = world_stores["ontology"]
        types = ontology.list_object_types()
        assert len(types) >= 7

    def test_list_property_types(self, world_stores):
        ontology = world_stores["ontology"]
        types = ontology.list_property_types()
        assert len(types) >= 16

    def test_list_relation_types(self, world_stores):
        ontology = world_stores["ontology"]
        types = ontology.list_relation_types()
        assert len(types) >= 10


class TestBitemporal:
    def test_current_interval(self):
        interval = make_current_interval()
        assert interval.is_valid_at(utcnow())
        assert interval.is_current()

    def test_bounded_interval(self):
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 12, 31, tzinfo=timezone.utc)
        interval = make_bounded_interval(start, end)
        assert interval.is_valid_at(datetime(2025, 6, 15, tzinfo=timezone.utc))
        assert not interval.is_valid_at(datetime(2026, 1, 1, tzinfo=timezone.utc))

    def test_superseded_interval(self):
        interval = make_current_interval()
        assert interval.is_current()
        closed = interval.close_valid(utcnow())
        assert not closed.is_valid_at(utcnow())

    def test_freshness(self):
        now = utcnow()
        fresh = make_current_interval(now)
        assert freshness_seconds(fresh.valid_from, now) < 1.0


class TestTypedValue:
    def test_string_value(self):
        tv = TypedValue(raw="hello", value_type=ValueType.STRING)
        assert tv.raw == "hello"
        assert tv.value_type == ValueType.STRING

    def test_geometry_value(self):
        bbox = [10, 20, 100, 200]
        tv = TypedValue(raw=bbox, value_type=ValueType.GEOMETRY)
        assert tv.raw == bbox
        assert tv.value_type == ValueType.GEOMETRY

    def test_numeric_value(self):
        tv = TypedValue(raw=42.5, value_type=ValueType.FLOAT)
        assert tv.raw == 42.5


class TestWorldAssertion:
    def test_create_assertion(self):
        a = WorldAssertion(
            assertion_id="test:1",
            subject_id="entity:1",
            property_id="label",
            value=TypedValue(raw="person", value_type=ValueType.STRING),
            epistemic_kind=EpistemicKind.OBSERVATION,
            source_id="camera:0",
            confidence=0.9,
            authority=0.8,
        )
        assert a.assertion_id == "test:1"
        assert not a.is_current  # starts as proposed
        a.state = AssertionState.ACCEPTED
        assert a.is_current

    def test_supersedes(self):
        a1 = WorldAssertion(
            assertion_id="test:1",
            subject_id="entity:1",
            property_id="label",
            value=TypedValue(raw="old", value_type=ValueType.STRING),
            epistemic_kind=EpistemicKind.OBSERVATION,
            source_id="camera:0",
        )
        a2 = WorldAssertion(
            assertion_id="test:2",
            subject_id="entity:1",
            property_id="label",
            value=TypedValue(raw="new", value_type=ValueType.STRING),
            epistemic_kind=EpistemicKind.OBSERVATION,
            source_id="camera:0",
        )
        a1.supersedes(a2)
        assert a1.state == AssertionState.SUPERSEDED


class TestWorldGraph:
    def test_add_and_traverse(self, world_stores):
        graph = world_stores["graph"]
        edge = WorldEdge(
            source_id="entity:1",
            relation_type="associated_with",
            target_id="entity:2",
            confidence=0.8,
            valid_from=utcnow().isoformat(),
        )
        graph.add_edge(edge)
        neighbors = graph.neighbors("entity:1")
        assert len(neighbors) == 1
        assert neighbors[0]["target_id"] == "entity:2"

    def test_outgoing(self, world_stores):
        graph = world_stores["graph"]
        edge = WorldEdge(
            source_id="entity:1",
            relation_type="associated_with",
            target_id="entity:2",
            confidence=0.8,
            valid_from=utcnow().isoformat(),
        )
        graph.add_edge(edge)
        out = graph.outgoing("entity:1")
        assert len(out) == 1

    def test_incoming(self, world_stores):
        graph = world_stores["graph"]
        edge = WorldEdge(
            source_id="entity:1",
            relation_type="associated_with",
            target_id="entity:2",
            confidence=0.8,
            valid_from=utcnow().isoformat(),
        )
        graph.add_edge(edge)
        inc = graph.incoming("entity:2")
        assert len(inc) == 1

    def test_find_path(self, world_stores):
        graph = world_stores["graph"]
        graph.add_edge(WorldEdge("a", "associated_with", "b", 0.9, valid_from=utcnow().isoformat()))
        graph.add_edge(WorldEdge("b", "associated_with", "c", 0.9, valid_from=utcnow().isoformat()))
        path = graph.find_path("a", "c")
        assert path == ["a", "b", "c"]

    def test_close_relation(self, world_stores):
        graph = world_stores["graph"]
        graph.add_edge(WorldEdge("a", "associated_with", "b", 0.9, valid_from=utcnow().isoformat()))
        graph.close_relation("a", "associated_with", "b", utcnow().isoformat())
        neighbors = graph.neighbors("a")
        assert len(neighbors) == 0


class TestIdentityGraph:
    def test_claim(self, world_stores):
        identity = world_stores["identity"]
        identity.claim("entity:1", ["ev:1"], "camera:0", "Initial detection")
        chain = identity.get_chain("entity:1")
        assert chain is not None
        assert chain.entity_id == "entity:1"

    def test_merge(self, world_stores):
        identity = world_stores["identity"]
        identity.claim("entity:1", ["ev:1"], "camera:0")
        identity.claim("entity:2", ["ev:2"], "camera:0")
        identity.merge("entity:1", ["entity:2"], ["ev:3"], "user:0", "Same person")
        chain = identity.get_chain("entity:1")
        assert chain is not None
        assert "entity:2" in chain.chain

    def test_split(self, world_stores):
        identity = world_stores["identity"]
        identity.claim("entity:1", ["ev:1"], "camera:0")
        identity.split("entity:1", ["entity:1a", "entity:1b"], ["ev:2"], "camera:0", "Different people")
        chain_a = identity.get_chain("entity:1a")
        assert chain_a is not None

    def test_history(self, world_stores):
        identity = world_stores["identity"]
        identity.claim("entity:1", ["ev:1"], "camera:0")
        identity.claim("entity:2", ["ev:2"], "camera:0")
        history = identity.history()
        assert len(history) == 2


class TestSpatial:
    def test_bbox2d(self):
        bbox = BBox2D(10, 20, 100, 200)
        assert bbox.center == (55, 110)
        assert bbox.width == 90
        assert bbox.height == 180
        assert bbox.area == 16200

    def test_bbox_overlap(self):
        b1 = BBox2D(0, 0, 100, 100)
        b2 = BBox2D(50, 50, 150, 150)
        b3 = BBox2D(200, 200, 300, 300)
        assert b1.overlaps(b2)
        assert not b1.overlaps(b3)

    def test_bbox_iou(self):
        b1 = BBox2D(0, 0, 100, 100)
        b2 = BBox2D(50, 50, 150, 150)
        iou = b1.iou(b2)
        assert 0.0 < iou < 1.0

    def test_bbox_contains_point(self):
        bbox = BBox2D(0, 0, 100, 100)
        assert bbox.contains((50, 50))
        assert not bbox.contains((150, 150))

    def test_spatial_reasoner(self):
        reasoner = SpatialReasoner()
        s1 = SpatialState(bbox=BBox2D(0, 0, 100, 100))
        s2 = SpatialState(bbox=BBox2D(200, 200, 300, 300))
        rels = reasoner.spatial_relation(s1, s2)
        assert SpatialRelation.LEFT_OF in rels or SpatialRelation.ABOVE in rels


class TestReconciler:
    def test_ingest_simple(self, world_stores):
        reconciler = world_stores["reconciler"]
        delta = WorldDelta()
        delta.assertions.append({
            "subject_id": "entity:1",
            "property_id": "label",
            "value": TypedValue(raw="person", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "camera:0",
            "confidence": 0.9,
            "valid_from": utcnow().isoformat(),
        })
        conflicts = reconciler.ingest(delta)
        assert isinstance(conflicts, list)

    def test_get_entity_assertions(self, world_stores):
        reconciler = world_stores["reconciler"]
        delta = WorldDelta()
        delta.assertions.append({
            "subject_id": "entity:1",
            "property_id": "label",
            "value": TypedValue(raw="person", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "camera:0",
            "confidence": 0.9,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta)
        assertions = reconciler.get_entity_assertions("entity:1")
        assert len(assertions) >= 1


class TestObservationNormalizer:
    def test_normalize_detection(self, world_stores):
        from dataclasses import dataclass
        from typing import Any
        
        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "event:1"
            event_type: str = "vision"
            occurred_at: str = ""
            source_id: str = "camera:cam0"
            evidence: tuple = ()
            entity_ids: tuple = ()
            payload: dict = None
            
            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})
        
        normalizer = world_stores["normalizer"]
        detection = {
            "entity_id": "entity:1",
            "label": "person",
            "confidence": 0.85,
            "bbox": [10, 20, 100, 200],
            "behavior": "standing",
        }
        event = MockEvent(
            payload={"detections": [detection], "frame_shape": (480, 640)},
            evidence=("ev:1",),
        )
        delta = normalizer.normalize_event(event, evidence_ids=("ev:1",), frame_shape=(480, 640))
        assert len(delta.assertions) >= 3
        assert delta.assertions[0]["subject_id"] == "entity:1"
        assert delta.assertions[0]["evidence_ids"] == ("ev:1",)
        assert delta.assertions[0]["authority"] > 0

    def test_normalize_detection_with_gaze_emits_gaze_state(self, world_stores):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "event:1g"
            event_type: str = "vision"
            occurred_at: str = ""
            source_id: str = "camera:cam0"
            evidence: tuple = ()
            entity_ids: tuple = ()
            payload: dict = None

            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})

        normalizer = world_stores["normalizer"]
        detection = {
            "entity_id": "entity:2",
            "label": "person",
            "confidence": 0.85,
            "bbox": [10, 20, 100, 200],
            "gaze": {"state": "facing_camera", "yaw_offset": 0.02, "confidence": 0.77},
        }
        event = MockEvent(payload={"detections": [detection]}, evidence=("ev:1",))
        delta = normalizer.normalize_event(event, evidence_ids=("ev:1",))
        gaze_assertions = [a for a in delta.assertions if a["property_id"] == "gaze_state"]
        assert len(gaze_assertions) == 1
        assert gaze_assertions[0]["subject_id"] == "entity:2"
        assert gaze_assertions[0]["value"].raw == "facing_camera"
        assert gaze_assertions[0]["confidence"] == 0.77

    def test_normalize_detection_without_gaze_emits_no_gaze_state(self, world_stores):
        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "event:1h"
            event_type: str = "vision"
            occurred_at: str = ""
            source_id: str = "camera:cam0"
            evidence: tuple = ()
            entity_ids: tuple = ()
            payload: dict = None

            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})

        normalizer = world_stores["normalizer"]
        detection = {
            "entity_id": "entity:3",
            "label": "person",
            "confidence": 0.85,
            "bbox": [10, 20, 100, 200],
        }
        event = MockEvent(payload={"detections": [detection]}, evidence=("ev:1",))
        delta = normalizer.normalize_event(event, evidence_ids=("ev:1",))
        assert not any(a["property_id"] == "gaze_state" for a in delta.assertions)

    def test_normalize_speech(self, world_stores):
        from dataclasses import dataclass
        
        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "event:2"
            event_type: str = "speech"
            occurred_at: str = ""
            source_id: str = "asr:0"
            evidence: tuple = ()
            entity_ids: tuple = ("person:1",)
            payload: dict = None
            
            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})
        
        normalizer = world_stores["normalizer"]
        event = MockEvent(
            payload={"transcript": "Hello Egg"},
            evidence=("ev:2",),
        )
        delta = normalizer.normalize_event(event, evidence_ids=("ev:2",))
        assert len(delta.events) == 1
        assert delta.events[0]["event_type_id"] == "speech_utterance"
        assert delta.events[0]["evidence_ids"] == ("ev:2",)

    def test_authority_is_computed(self, world_stores):
        from dataclasses import dataclass
        
        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "event:3"
            event_type: str = "vision"
            occurred_at: str = ""
            source_id: str = "detector:0"
            evidence: tuple = ()
            entity_ids: tuple = ()
            payload: dict = None
            
            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})
        
        normalizer = world_stores["normalizer"]
        event = MockEvent(
            payload={"detections": [{"entity_id": "obj:1", "label": "cup", "confidence": 0.9, "bbox": [0, 0, 50, 50]}]},
        )
        delta = normalizer.normalize_event(event)
        for a in delta.assertions:
            assert "authority" in a
            assert 0 < a["authority"] <= 1.0


class TestEventStore:
    def test_record_and_retrieve(self, world_stores):
        events = world_stores["events"]
        from egg_companion.world.events import EventOccurrence
        event = EventOccurrence(
            event_id="event:1",
            event_type_id="speech_utterance",
            roles={"speaker": "person:1", "transcript": "Hello"},
            valid_from=utcnow().isoformat(),
            observed_at=utcnow().isoformat(),
            source_id="asr:0",
        )
        events.record(event)
        retrieved = events.get_events("speech_utterance")
        assert len(retrieved) == 1
        assert retrieved[0]["event_id"] == "event:1"


class TestWorldQuery:
    def test_entity(self, world_stores):
        query = world_stores["query"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        ev = query.entity("entity:1")
        assert ev is not None
        assert ev.entity_id == "entity:1"
        assert "label" in ev.properties

    def test_entities(self, world_stores):
        query = world_stores["query"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        entities = query.entities()
        assert len(entities) >= 1

    def test_property_value(self, world_stores):
        query = world_stores["query"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        val = query.property_value("entity:1", "label")
        assert val == "person"

    def test_summary(self, world_stores):
        query = world_stores["query"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        summary = query.summary()
        assert summary["total_entities"] >= 1


class TestCognitiveContext:
    def test_build_window(self, world_stores):
        context = world_stores["context"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        window = context.build_window()
        assert len(window.entities) >= 1
        assert window.total_characters > 0

    def test_for_entity(self, world_stores):
        context = world_stores["context"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        ec = context.for_entity("entity:1")
        assert ec is not None
        assert ec.entity_id == "entity:1"

    def test_serialize_for_llm(self, world_stores):
        context = world_stores["context"]
        state = world_stores["state"]
        state.upsert_property(
            "entity:1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        window = context.build_window()
        llm_str = context.serialize_for_llm(window)
        assert "entity:1" in llm_str


class TestPolicyValidator:
    def test_validate_proposal(self, world_stores):
        policy = world_stores["policy"]
        from egg_companion.world.types import ActionProposal
        proposal = ActionProposal(
            proposal_id="prop:1",
            action_type="speak",
            inputs={"text": "Hello"},
        )
        violations = policy.validate(proposal)
        assert isinstance(violations, list)


class TestMetricsCollector:
    def test_collect_metrics(self, world_stores):
        metrics = world_stores["metrics"]
        m = metrics.collect()
        assert m.total_entities >= 0
        assert m.total_assertions >= 0

    def test_to_dict(self, world_stores):
        metrics = world_stores["metrics"]
        d = metrics.to_dict()
        assert "total_entities" in d
        assert "avg_confidence" in d


class TestAuthorityPolicy:
    def test_default_authority(self):
        policy = AuthorityPolicy()
        score = policy.evaluate("physical_object.label", "detector", "observation")
        assert 0.0 <= score <= 1.0

    def test_user_correction_higher(self):
        policy = AuthorityPolicy()
        user_score = policy.evaluate("person.preferred_name", "user_correction", "correction")
        llm_score = policy.evaluate("person.preferred_name", "llm_inference", "inference")
        assert user_score > llm_score


class TestEvidenceProvenance:
    """Verify evidence IDs flow through the full normalize→reconcile→state path."""

    def test_evidence_ids_on_assertions(self, world_stores):
        normalizer = world_stores["normalizer"]
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "e1"
            event_type: str = "vision"
            occurred_at: str = ""
            source_id: str = "camera:cam0"
            evidence: tuple = ()
            entity_ids: tuple = ()
            payload: dict = None
            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})

        event = MockEvent(
            payload={"detections": [{"entity_id": "obj:1", "label": "cup", "confidence": 0.9, "bbox": [0, 0, 50, 50]}]},
        )
        delta = normalizer.normalize_event(event, evidence_ids=("ev:aaa", "ev:bbb"))
        conflicts = reconciler.ingest(delta)

        assertions = reconciler.get_entity_assertions("obj:1")
        assert len(assertions) >= 1
        for a in assertions:
            evidence = json.loads(a["evidence_ids_json"]) if "evidence_ids_json" in a else a.get("evidence_ids", [])
            assert "ev:aaa" in evidence or len(evidence) == 0

        current = state.get_property("obj:1", "label")
        assert current is not None
        stored_evidence = json.loads(current.evidence_ids_json)
        assert "ev:aaa" in stored_evidence

    def test_event_evidence_ids(self, world_stores):
        normalizer = world_stores["normalizer"]
        reconciler = world_stores["reconciler"]

        from dataclasses import dataclass

        @dataclass(frozen=True)
        class MockEvent:
            event_id: str = "e2"
            event_type: str = "speech"
            occurred_at: str = ""
            source_id: str = "asr:0"
            evidence: tuple = ()
            entity_ids: tuple = ("person:1",)
            payload: dict = None
            def __post_init__(self):
                if self.payload is None:
                    object.__setattr__(self, "payload", {})

        event = MockEvent(payload={"transcript": "hello"})
        delta = normalizer.normalize_event(event, evidence_ids=("ev:speech1",))
        reconciler.ingest(delta)

        with world_stores["db"] as conn:
            rows = conn.execute(
                "SELECT evidence_ids_json FROM event_assertions"
            ).fetchall()
            assert len(rows) >= 1
            evidence = json.loads(rows[0][0])
            assert "ev:speech1" in evidence


class TestConflictResolution:
    """Verify conflict resolution uses actual existing authority."""

    def test_higher_authority_wins(self, world_stores):
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]

        delta1 = WorldDelta()
        delta1.assertions.append({
            "subject_id": "obj:1",
            "property_id": "label",
            "value": TypedValue(raw="cup", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "llm_inference:0",
            "confidence": 0.7,
            "authority": 0.4,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta1)

        delta2 = WorldDelta()
        delta2.assertions.append({
            "subject_id": "obj:1",
            "property_id": "label",
            "value": TypedValue(raw="mug", value_type=ValueType.STRING),
            "epistemic_kind": "correction",
            "source_id": "user_correction:0",
            "confidence": 0.95,
            "authority": 0.95,
            "valid_from": utcnow().isoformat(),
        })
        conflicts = reconciler.ingest(delta2)

        current = state.get_property("obj:1", "label")
        assert current is not None
        assert json.loads(current.value_json) == "mug"

    def test_lower_authority_stays_proposed(self, world_stores):
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]

        delta1 = WorldDelta()
        delta1.assertions.append({
            "subject_id": "obj:2",
            "property_id": "label",
            "value": TypedValue(raw="cup", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "user_correction:0",
            "confidence": 0.95,
            "authority": 0.95,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta1)

        delta2 = WorldDelta()
        delta2.assertions.append({
            "subject_id": "obj:2",
            "property_id": "label",
            "value": TypedValue(raw="bowl", value_type=ValueType.STRING),
            "epistemic_kind": "inference",
            "source_id": "llm_inference:0",
            "confidence": 0.6,
            "authority": 0.3,
            "valid_from": utcnow().isoformat(),
        })
        conflicts = reconciler.ingest(delta2)

        current = state.get_property("obj:2", "label")
        assert current is not None
        assert json.loads(current.value_json) == "cup"

    def test_equal_authority_conflicted(self, world_stores):
        reconciler = world_stores["reconciler"]

        delta1 = WorldDelta()
        delta1.assertions.append({
            "subject_id": "obj:3",
            "property_id": "label",
            "value": TypedValue(raw="cup", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "camera:0",
            "confidence": 0.8,
            "authority": 0.8,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta1)

        delta2 = WorldDelta()
        delta2.assertions.append({
            "subject_id": "obj:3",
            "property_id": "label",
            "value": TypedValue(raw="mug", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "camera:1",
            "confidence": 0.8,
            "authority": 0.8,
            "valid_from": utcnow().isoformat(),
        })
        conflicts = reconciler.ingest(delta2)
        assert len(conflicts) >= 1

        db_conflicts = reconciler.get_conflicts()
        assert len(db_conflicts) >= 1


class TestSupersession:
    """Verify that when a stronger assertion wins, the old one is superseded."""

    def test_old_assertion_superseded(self, world_stores):
        reconciler = world_stores["reconciler"]

        delta1 = WorldDelta()
        delta1.assertions.append({
            "subject_id": "obj:10",
            "property_id": "label",
            "value": TypedValue(raw="old_label", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "llm_inference:0",
            "confidence": 0.5,
            "authority": 0.3,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta1)

        delta2 = WorldDelta()
        delta2.assertions.append({
            "subject_id": "obj:10",
            "property_id": "label",
            "value": TypedValue(raw="new_label", value_type=ValueType.STRING),
            "epistemic_kind": "correction",
            "source_id": "user_correction:0",
            "confidence": 0.95,
            "authority": 0.95,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta2)

        history = reconciler.get_assertion_history("obj:10", "label")
        states = [a["state"] for a in history]
        assert "superseded" in states
        assert "accepted" in states

        superseded = [a for a in history if a["state"] == "superseded"]
        assert len(superseded) >= 1
        assert superseded[0]["valid_to"] is not None


class TestAtomicTransaction:
    """Verify that WorldDelta is committed atomically."""

    def test_partial_failure_rolls_back(self, world_stores):
        from egg_companion.world.state import WorldStateStore

        state = world_stores["state"]
        initial = state.revision

        delta = WorldDelta()
        delta.assertions.append({
            "subject_id": "obj:20",
            "property_id": "label",
            "value": TypedValue(raw="test", value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": "camera:0",
            "confidence": 0.8,
            "authority": 0.7,
            "valid_from": utcnow().isoformat(),
        })
        world_stores["reconciler"].ingest(delta)

        assert state.revision > initial


class TestRevisionRestartSafety:
    """Verify revision counter loads correctly from DB."""

    def test_revision_loads_from_db(self, db):
        state1 = WorldStateStore(db)
        state1.allocate_revision("first")
        state1.allocate_revision("second")
        rev_after = state1.revision

        state2 = WorldStateStore(db)
        assert state2.revision == rev_after

    def test_revision_increments_from_loaded(self, db):
        state1 = WorldStateStore(db)
        for _ in range(5):
            state1.allocate_revision()

        state2 = WorldStateStore(db)
        state2.allocate_revision("after restart")
        assert state2.revision == 6


class TestWorldStateStoreProperty:
    """Test property upsert and retrieval round-trip."""

    def test_upsert_and_get(self, db):
        state = WorldStateStore(db)
        state.upsert_property(
            "e1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", ("ev:1",), "observation", utcnow().isoformat(),
        )
        row = state.get_property("e1", "label")
        assert row is not None
        assert json.loads(row.value_json) == "person"
        assert row.confidence == 0.9
        assert row.authority == 0.8

    def test_explain(self, db):
        state = WorldStateStore(db)
        state.upsert_property(
            "e1", "label",
            TypedValue(raw="person", value_type=ValueType.STRING),
            0.9, 0.8, "assert:1", ("ev:1",), "observation", utcnow().isoformat(),
        )
        explanation = state.explain("e1", "label")
        assert explanation["value"] == "person"
        assert explanation["evidence_ids"] == ["ev:1"]


class TestWorldStatePruning:
    """Prune paths must use real epistemic confidence, and any deletion cap
    must be enforced in SQL — never by slicing a list after rows are already
    gone, which silently deletes more than the caller asked for."""

    def test_prune_low_confidence_uses_confidence_column_not_label_json(self, db):
        """A JSON label string like "mug" cast as REAL is 0.0 in SQLite —
        the old query computed MAX(CAST(value_json AS REAL)) over the
        'label' property, which made every entity look like confidence 0.0
        regardless of its real confidence and pruned it incorrectly."""
        state = WorldStateStore(db)
        state.upsert_property(
            "det:mug-1", "label",
            TypedValue(raw="mug", value_type=ValueType.STRING),
            0.91, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        pruned = state.prune_low_confidence(entity_prefix="det:", max_confidence=0.4)
        assert pruned == []
        assert state.get_property("det:mug-1", "label") is not None

    def test_prune_low_confidence_still_prunes_genuinely_low_confidence(self, db):
        state = WorldStateStore(db)
        state.upsert_property(
            "det:blur-1", "label",
            TypedValue(raw="unknown", value_type=ValueType.STRING),
            0.10, 0.3, "assert:1", (), "observation", utcnow().isoformat(),
        )
        pruned = state.prune_low_confidence(entity_prefix="det:", max_confidence=0.4)
        assert pruned == ["det:blur-1"]
        assert state.get_property("det:blur-1", "label") is None

    def test_prune_stale_entities_limit_is_enforced_in_sql(self, db):
        state = WorldStateStore(db)
        old = "2020-01-01T00:00:00+00:00"
        for i in range(5):
            state.upsert_property(
                f"det:stale-{i}", "label",
                TypedValue(raw=f"thing-{i}", value_type=ValueType.STRING),
                0.1, 0.3, f"assert:{i}", (), "observation", old,
            )
        pruned = state.prune_stale_entities(
            stale_before="2099-01-01T00:00:00+00:00",
            entity_prefix="det:", min_confidence=0.9, limit=2,
        )
        # The whole point of `limit`: it bounds actual deletions, not just
        # the returned list — re-querying confirms only 2 rows are gone.
        assert len(pruned) == 2
        assert state.entity_count("det:") == 3

    def test_prune_stale_entities_without_limit_deletes_all_matches(self, db):
        state = WorldStateStore(db)
        old = "2020-01-01T00:00:00+00:00"
        for i in range(3):
            state.upsert_property(
                f"det:stale-{i}", "label",
                TypedValue(raw=f"thing-{i}", value_type=ValueType.STRING),
                0.1, 0.3, f"assert:{i}", (), "observation", old,
            )
        pruned = state.prune_stale_entities(
            stale_before="2099-01-01T00:00:00+00:00",
            entity_prefix="det:", min_confidence=0.9,
        )
        assert len(pruned) == 3
        assert state.entity_count("det:") == 0


class TestCorrelationAwareConfidence:
    """Test that correlation groups affect confidence aggregation."""

    def test_temporal_correlation(self, world_stores):
        from egg_companion.world.types import EvidenceCorrelationGroup

        reconciler = world_stores["reconciler"]
        group = EvidenceCorrelationGroup(
            group_id="g1",
            correlation_type="temporal",
            observation_count=10,
        )
        assert group.independence_factor < 1.0

        confidences = [0.8] * 10
        uncorrelated = reconciler.aggregate_confidence_with_correlation(confidences)
        correlated = reconciler.aggregate_confidence_with_correlation(confidences, group)
        assert correlated < uncorrelated

    def test_single_observation(self, world_stores):
        reconciler = world_stores["reconciler"]
        score = reconciler.aggregate_confidence_with_correlation([0.8])
        assert score == 0.8


class TestContextAssemblerWorldIntegration:
    """Test that ContextAssembler integrates world state."""

    def test_world_context_set(self):
        from egg_companion.memory.context import ContextAssembler
        from unittest.mock import MagicMock

        store = MagicMock()
        store.config = MagicMock()
        store.config.context_max_characters = 3000
        store.config.graph_max_nodes = 100
        store.cognitive_documents.return_value = []
        store.list_claims.return_value = []

        assembler = ContextAssembler(store)
        assert assembler._world_context is None

        mock_context = MagicMock()
        assembler.set_world_context(mock_context)
        assert assembler._world_context is mock_context


class TestCalibration:
    """Test Calibration projection and validity."""

    def test_project_3d_to_2d(self):
        from egg_companion.world.spatial import Calibration
        P = [600, 0, 320, 0, 0, 600, 240, 0, 0, 0, 1, 0]
        cal = Calibration(camera_id="cam0", projection_matrix=P)
        px, py = cal.project((0.0, 0.0, 5.0))
        assert px == 320.0
        assert py == 240.0

    def test_is_valid_at_none_always_valid(self):
        from egg_companion.world.spatial import Calibration
        cal = Calibration(camera_id="cam0", projection_matrix=[1]*12)
        assert cal.is_valid_at(None) is True

    def test_valid_from_to(self):
        from egg_companion.world.spatial import Calibration
        from datetime import datetime, timezone
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        t_mid = datetime(2026, 3, 1, tzinfo=timezone.utc)
        t_before = datetime(2025, 1, 1, tzinfo=timezone.utc)
        cal = Calibration(camera_id="cam0", projection_matrix=[1]*12, valid_from=t1, valid_to=t2)
        assert cal.is_valid_at(t_mid) is True
        assert cal.is_valid_at(t_before) is False

    def test_to_dict_roundtrip(self):
        from egg_companion.world.spatial import Calibration
        cal = Calibration(camera_id="cam0", projection_matrix=[1]*12, distortion=[0.1,0.2,0,0,0], source="test")
        d = cal.to_dict()
        assert d["camera_id"] == "cam0"
        assert len(d["projection_matrix"]) == 12
        assert d["distortion"] == [0.1,0.2,0,0,0]


class TestTransform:
    """Test Transform composition, inverse, and validity."""

    def test_identity_transform(self):
        from egg_companion.world.spatial import Transform, IDENTITY_4X4
        t = Transform("A", "A", list(IDENTITY_4X4))
        result = t.apply((1.0, 2.0, 3.0))
        assert result == (1.0, 2.0, 3.0)

    def test_translation_transform(self):
        from egg_companion.world.spatial import Transform
        # Translate +5 in X
        m = [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        t = Transform("cam", "world", m)
        result = t.apply((0.0, 0.0, 0.0))
        assert result == (5.0, 0.0, 0.0)

    def test_inverse_roundtrip(self):
        from egg_companion.world.spatial import Transform
        import math
        angle = math.pi / 4  # 45 degrees
        c, s = math.cos(angle), math.sin(angle)
        m = [c,-s,0,1, s,c,0,2, 0,0,1,3, 0,0,0,1]
        t = Transform("A", "B", m)
        inv = t.inverse
        point = (1.0, 2.0, 3.0)
        roundtrip = t.apply(inv.apply(point))
        for a, b in zip(point, roundtrip):
            assert abs(a - b) < 1e-9

    def test_compose(self):
        from egg_companion.world.spatial import Transform, IDENTITY_4X4
        m1 = [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        m2 = [1,0,0,0, 0,1,0,3, 0,0,1,0, 0,0,0,1]
        t1 = Transform("A", "B", m1)
        t2 = Transform("B", "C", m2)
        composed = t1.compose(t2)
        assert composed.source_frame == "A"
        assert composed.target_frame == "C"
        result = composed.apply((0.0, 0.0, 0.0))
        assert result == (5.0, 3.0, 0.0)

    def test_validity_time_window(self):
        from egg_companion.world.spatial import Transform
        from datetime import datetime, timezone
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        tr = Transform("A", "B", [1]*16, valid_from=t1, valid_to=t2)
        assert tr.is_valid_at(datetime(2026, 3, 1, tzinfo=timezone.utc)) is True
        assert tr.is_valid_at(datetime(2025, 1, 1, tzinfo=timezone.utc)) is False

    def test_to_dict(self):
        from egg_companion.world.spatial import Transform
        tr = Transform("A", "B", [1]*16, source="test")
        d = tr.to_dict()
        assert d["source_frame"] == "A"
        assert d["target_frame"] == "B"
        assert len(d["matrix"]) == 16


class TestTransformTree:
    """Test BFS path resolution, calibration lookup, and SQLite round-trip."""

    def test_direct_path(self):
        from egg_companion.world.spatial import TransformTree, Transform
        tree = TransformTree()
        m = [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        tree.add_transform(Transform("cam0", "world", m))
        result = tree.resolve("cam0", "world")
        assert result is not None
        assert result.source_frame == "cam0"
        assert result.target_frame == "world"

    def test_indirect_path(self):
        from egg_companion.world.spatial import TransformTree, Transform
        tree = TransformTree()
        m1 = [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        m2 = [1,0,0,0, 0,1,0,3, 0,0,1,0, 0,0,0,1]
        tree.add_transform(Transform("cam0", "optical", m1))
        tree.add_transform(Transform("optical", "world", m2))
        result = tree.resolve("cam0", "world")
        assert result is not None
        pt = result.apply((0.0, 0.0, 0.0))
        assert pt == (5.0, 3.0, 0.0)

    def test_no_path_returns_none(self):
        from egg_companion.world.spatial import TransformTree, Transform
        tree = TransformTree()
        tree.add_transform(Transform("A", "B", [1]*16))
        assert tree.resolve("A", "C") is None

    def test_same_frame_identity(self):
        from egg_companion.world.spatial import TransformTree, IDENTITY_4X4
        tree = TransformTree()
        result = tree.resolve("A", "A")
        assert result is not None
        assert result.matrix == IDENTITY_4X4

    def test_timestamp_filters_invalid(self):
        from egg_companion.world.spatial import TransformTree, Transform
        from datetime import datetime, timezone
        tree = TransformTree()
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        m = [1,0,0,5, 0,1,0,0, 0,0,1,0, 0,0,0,1]
        tree.add_transform(Transform("A", "B", m, valid_from=t1, valid_to=t2))
        # Valid timestamp
        assert tree.resolve("A", "B", datetime(2026, 3, 1, tzinfo=timezone.utc)) is not None
        # Invalid timestamp
        assert tree.resolve("A", "B", datetime(2025, 1, 1, tzinfo=timezone.utc)) is None

    def test_calibration_lookup(self):
        from egg_companion.world.spatial import TransformTree, Calibration
        from datetime import datetime, timezone
        tree = TransformTree()
        t1 = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t2 = datetime(2026, 6, 1, tzinfo=timezone.utc)
        cal = Calibration("cam0", [600,0,320,0, 0,600,240,0, 0,0,1,0], valid_from=t1, valid_to=t2)
        tree.add_calibration(cal)
        found = tree.get_calibration("cam0", datetime(2026, 3, 1, tzinfo=timezone.utc))
        assert found is not None
        assert found.camera_id == "cam0"
        assert tree.get_calibration("cam99") is None

    def test_list_frames(self):
        from egg_companion.world.spatial import TransformTree, Transform
        tree = TransformTree()
        tree.add_transform(Transform("A", "B", [1]*16))
        tree.add_transform(Transform("B", "C", [1]*16))
        frames = tree.list_frames()
        assert set(frames) == {"A", "B", "C"}

    def test_sqlite_roundtrip(self, db):
        from egg_companion.world.spatial import TransformTree, Transform, Calibration
        from datetime import datetime, timezone

        tree = TransformTree(db)
        m = [1,0,0,5, 0,1,0,3, 0,0,1,0, 0,0,0,1]
        tree.add_transform(Transform("cam0", "world", m, source="test"))
        tree.add_calibration(Calibration("cam0", [600,0,320,0, 0,600,240,0, 0,0,1,0], source="test"))
        tree.save_to_sqlite()

        # Fresh tree, load from DB
        tree2 = TransformTree(db)
        tree2.load_from_sqlite()
        result = tree2.resolve("cam0", "world")
        assert result is not None
        pt = result.apply((0.0, 0.0, 0.0))
        assert pt == (5.0, 3.0, 0.0)
        cal = tree2.get_calibration("cam0")
        assert cal is not None
        assert cal.camera_id == "cam0"


class TestAtomicTransaction:
    """Test that world_transaction provides true atomicity."""

    def test_rollback_on_exception(self, db):
        from egg_companion.world.state import WorldStateStore
        from egg_companion.world.types import TypedValue, ValueType
        state = WorldStateStore(db)
        initial_rev = state.revision

        try:
            with state.world_transaction("test_rollback"):
                state.upsert_property(
                    "e1", "label",
                    TypedValue(raw="should_rollback", value_type=ValueType.STRING),
                    0.9, 0.8, "assert:1", ("ev:1",), "observation",
                    utcnow().isoformat(), revision=999,
                )
                raise ValueError("force rollback")
        except ValueError:
            pass

        row = state.get_property("e1", "label")
        assert row is None
        assert state.revision == initial_rev

    def test_all_inner_commits_suppressed(self, db):
        from egg_companion.world.state import WorldStateStore
        from egg_companion.world.types import TypedValue, ValueType
        state = WorldStateStore(db)

        with state.world_transaction("multi_upsert"):
            rev = state._current_revision
            state.upsert_property(
                "e1", "label",
                TypedValue(raw="a", value_type=ValueType.STRING),
                0.9, 0.8, "assert:1", ("ev:1",), "observation",
                utcnow().isoformat(), revision=rev,
            )
            state.upsert_property(
                "e1", "bbox",
                TypedValue(raw=[0,0,10,10], value_type=ValueType.GEOMETRY),
                0.9, 0.7, "assert:2", ("ev:1",), "observation",
                utcnow().isoformat(), revision=rev,
            )
            state.upsert_relation(
                "e1", "visible_from", "cam:cam0",
                0.9, 0.8, "assert:3", ("ev:1",), "observation",
                utcnow().isoformat(), revision=rev,
            )

        row = state.get_property("e1", "label")
        assert row is not None
        assert json.loads(row.value_json) == "a"
        assert row.revision == rev
        row2 = state.get_property("e1", "bbox")
        assert row2.revision == rev


class TestSingleRevisionPerDelta:
    """Test that one WorldDelta gets one revision."""

    def test_ingest_creates_one_revision(self, world_stores):
        from egg_companion.world.types import TypedValue, ValueType, WorldDelta
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]
        initial_rev = state.revision

        delta = WorldDelta()
        for i in range(5):
            delta.assertions.append({
                "subject_id": f"entity_{i}",
                "property_id": "label",
                "value": TypedValue(raw=f"thing_{i}", value_type=ValueType.STRING),
                "epistemic_kind": "observation",
                "source_id": "test:cam",
                "evidence_ids": ("ev:1",),
                "confidence": 0.9,
                "authority": 0.8,
                "valid_from": utcnow().isoformat(),
            })

        reconciler.ingest(delta)

        # All properties should share the same revision
        revisions = set()
        for i in range(5):
            row = state.get_property(f"entity_{i}", "label")
            if row:
                revisions.add(row.revision)
        assert len(revisions) == 1
        assert state.revision == initial_rev + 1


class TestRelationReconciliation:
    """Test RelationReconciler authority and expiry."""

    def _ensure_tables(self, conn):
        conn.executescript("""
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
        """)

    def test_higher_authority_supersedes(self, db):
        from egg_companion.world.state import WorldStateStore
        from egg_companion.world.reconcile import RelationReconciler
        self._ensure_tables(db)
        state = WorldStateStore(db)
        rr = RelationReconciler(db, state)

        # Insert initial relation
        with state.world_transaction():
            rev = state._current_revision
            state.upsert_relation(
                "e1", "near", "e2",
                0.8, 0.5, "assert:1", ("ev:1",), "observation",
                utcnow().isoformat(), revision=rev,
            )
            db.execute(
                """INSERT INTO relation_assertions
                (assertion_id, source_entity_id, relation_type_id, target_entity_id,
                 epistemic_kind, source_id, evidence_ids_json, confidence, authority,
                 valid_from, observed_at, recorded_at, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')""",
                ("assert:1", "e1", "near", "e2", "observation", "test",
                 '["ev:1"]', 0.8, 0.5, utcnow().isoformat(),
                 utcnow().isoformat(), utcnow().isoformat()),
            )

        result = rr.reconcile_relation(
            "e1", "near", "e2",
            new_authority=0.9, new_confidence=0.9,
            evidence_ids=("ev:2",), epistemic_kind="observation",
            source_origin="test", valid_from=utcnow().isoformat(),
        )
        assert result == "superseded"

    def test_lower_authority_rejected(self, db):
        from egg_companion.world.state import WorldStateStore
        from egg_companion.world.reconcile import RelationReconciler
        self._ensure_tables(db)
        state = WorldStateStore(db)
        rr = RelationReconciler(db, state)

        with state.world_transaction():
            rev = state._current_revision
            state.upsert_relation(
                "e1", "near", "e2",
                0.8, 0.8, "assert:1", ("ev:1",), "observation",
                utcnow().isoformat(), revision=rev,
            )
            db.execute(
                """INSERT INTO relation_assertions
                (assertion_id, source_entity_id, relation_type_id, target_entity_id,
                 epistemic_kind, source_id, evidence_ids_json, confidence, authority,
                 valid_from, observed_at, recorded_at, state)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'accepted')""",
                ("assert:1", "e1", "near", "e2", "observation", "test",
                 '["ev:1"]', 0.8, 0.8, utcnow().isoformat(),
                 utcnow().isoformat(), utcnow().isoformat()),
            )

        result = rr.reconcile_relation(
            "e1", "near", "e2",
            new_authority=0.5, new_confidence=0.5,
            evidence_ids=("ev:2",), epistemic_kind="observation",
            source_origin="test", valid_from=utcnow().isoformat(),
        )
        assert result == "rejected"


class TestConflictQuerying:
    """Test that WorldQuery.conflicts() actually returns data."""

    def test_conflicts_populated(self, world_stores):
        from egg_companion.world.types import TypedValue, ValueType, WorldDelta
        reconciler = world_stores["reconciler"]
        query = world_stores["query"]

        # Create two conflicting assertions
        delta1 = WorldDelta()
        delta1.assertions.append({
            "subject_id": "e1", "property_id": "label",
            "value": TypedValue(raw="Alice", value_type=ValueType.STRING),
            "epistemic_kind": "observation", "source_id": "cam:1",
            "evidence_ids": ("ev:1",), "confidence": 0.9, "authority": 0.7,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta1)

        delta2 = WorldDelta()
        delta2.assertions.append({
            "subject_id": "e1", "property_id": "label",
            "value": TypedValue(raw="Bob", value_type=ValueType.STRING),
            "epistemic_kind": "observation", "source_id": "cam:2",
            "evidence_ids": ("ev:2",), "confidence": 0.9, "authority": 0.7,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta2)

        conflicts = query.conflicts()
        assert len(conflicts) > 0
        assert conflicts[0].entity_id == "e1"
        assert conflicts[0].property_id == "label"
        assert len(conflicts[0].assertions) > 0


class TestActionProposalPersistence:
    """Test that all ActionProposal fields survive persist/retrieve."""

    def test_full_roundtrip(self, db):
        from egg_companion.world.actions import ActionStore
        from egg_companion.world.types import ActionProposal
        from datetime import datetime, timezone
        store = ActionStore(db)
        proposal = ActionProposal(
            proposal_id="prop:001",
            action_type="speak",
            target_entity_ids=("e1", "e2"),
            inputs={"text": "hello"},
            preconditions=("entity_visible:e1",),
            expected_effects=("audio_output:hello",),
            source_evidence_ids=("ev:1",),
            based_on_revision=42,
            status="pending",
            reason="test",
            proposed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        )
        store.propose(proposal)
        retrieved = store.get_proposal("prop:001")
        assert retrieved is not None
        assert retrieved.preconditions == ("entity_visible:e1",)
        assert retrieved.expected_effects == ("audio_output:hello",)
        assert retrieved.based_on_revision == 42
        assert retrieved.target_entity_ids == ("e1", "e2")
        assert isinstance(retrieved.proposed_at, datetime)


class TestSafeZonePolicyTarget:
    """Test that safe-zone checks target_entity_ids, not proposal_id."""

    def test_blocks_correct_target(self, db):
        from egg_companion.world.policy import PolicyValidator
        from egg_companion.world.types import ActionProposal
        validator = PolicyValidator(db)
        proposal = ActionProposal(
            proposal_id="prop:xyz",
            action_type="move_object",
            target_entity_ids=("egg_bed",),
        )
        violations = validator.validate(proposal)
        safe_zone_violations = [v for v in violations if v.rule_id == "safe_zone"]
        assert len(safe_zone_violations) > 0
        assert "egg_bed" in safe_zone_violations[0].reason

    def test_allows_non_safe_target(self, db):
        from egg_companion.world.policy import PolicyValidator
        from egg_companion.world.types import ActionProposal
        validator = PolicyValidator(db)
        proposal = ActionProposal(
            proposal_id="prop:xyz",
            action_type="move_object",
            target_entity_ids=("kitchen_table",),
        )
        violations = validator.validate(proposal)
        safe_zone_violations = [v for v in violations if v.rule_id == "safe_zone"]
        assert len(safe_zone_violations) == 0

    def test_does_not_false_positive_on_substring_match(self, db):
        """A target whose id merely *contains* a safe-zone name as a
        substring (but isn't actually located there) must not be blocked —
        e.g. "object:egg_bedspread" is not "egg_bed"."""
        from egg_companion.world.policy import PolicyValidator
        from egg_companion.world.types import ActionProposal
        validator = PolicyValidator(db)
        proposal = ActionProposal(
            proposal_id="prop:xyz",
            action_type="move_object",
            target_entity_ids=("object:egg_bedspread",),
        )
        violations = validator.validate(proposal)
        safe_zone_violations = [v for v in violations if v.rule_id == "safe_zone"]
        assert len(safe_zone_violations) == 0

    def test_blocks_target_located_in_zone_via_world_relations(self, db):
        """A target that is transitively located_in a safe-zone entity
        (per materialized world state) must be blocked even though its
        own id has nothing to do with the zone name."""
        from egg_companion.world.policy import PolicyValidator
        from egg_companion.world.types import ActionProposal

        state = WorldStateStore(db)
        state.upsert_relation(
            "object:small-toy", "located_in", "zone:egg_bed",
            0.9, 0.8, "assert:1", (), "observation", utcnow().isoformat(),
        )
        validator = PolicyValidator(db)
        proposal = ActionProposal(
            proposal_id="prop:xyz",
            action_type="move_object",
            target_entity_ids=("object:small-toy",),
        )
        violations = validator.validate(proposal)
        safe_zone_violations = [v for v in violations if v.rule_id == "safe_zone"]
        assert len(safe_zone_violations) > 0

    def test_missing_world_tables_fail_open_to_no_violation(self, db):
        """Standalone PolicyValidator usage without WorldStateStore tables
        on the connection must not crash — it just can't determine
        location, so it can't claim the target is in a zone."""
        from egg_companion.world.policy import PolicyValidator
        from egg_companion.world.types import ActionProposal
        validator = PolicyValidator(db)
        proposal = ActionProposal(
            proposal_id="prop:xyz",
            action_type="move_object",
            target_entity_ids=("object:whatever",),
        )
        violations = validator.validate(proposal)
        safe_zone_violations = [v for v in violations if v.rule_id == "safe_zone"]
        assert len(safe_zone_violations) == 0


class TestContextWindowRanking:
    """Test that build_window prioritizes focus_entity."""

    def test_focus_entity_first(self, world_stores):
        from egg_companion.world.context import CognitiveContext
        from egg_companion.world.types import TypedValue, ValueType, WorldDelta
        reconciler = world_stores["reconciler"]
        query = world_stores["query"]

        for i in range(5):
            delta = WorldDelta()
            delta.assertions.append({
                "subject_id": f"entity_{i}", "property_id": "label",
                "value": TypedValue(raw=f"thing_{i}", value_type=ValueType.STRING),
                "epistemic_kind": "observation", "source_id": "test",
                "evidence_ids": ("ev:1",), "confidence": 0.8, "authority": 0.7,
                "valid_from": utcnow().isoformat(),
            })
            reconciler.ingest(delta)

        ctx = CognitiveContext(query)
        window = ctx.build_window(focus_entity="entity_3", max_entities=3)
        assert len(window.entities) > 0
        assert window.entities[0]["entity_id"] == "entity_3"


class TestTypedPredictions:
    """Test TypedPrediction records."""

    def test_predict_returns_structure(self, world_stores):
        from egg_companion.core.prediction import WorldStatePredictor, TypedPrediction
        from egg_companion.world.types import TypedValue, ValueType, WorldDelta
        reconciler = world_stores["reconciler"]
        query = world_stores["query"]

        delta = WorldDelta()
        delta.assertions.append({
            "subject_id": "e1", "property_id": "label",
            "value": TypedValue(raw="person", value_type=ValueType.STRING),
            "epistemic_kind": "observation", "source_id": "test",
            "evidence_ids": ("ev:1",), "confidence": 0.9, "authority": 0.8,
            "valid_from": utcnow().isoformat(),
        })
        delta.assertions.append({
            "subject_id": "e1", "property_id": "behavior",
            "value": TypedValue(raw="idle", value_type=ValueType.STRING),
            "epistemic_kind": "observation", "source_id": "test",
            "evidence_ids": ("ev:1",), "confidence": 0.8, "authority": 0.7,
            "valid_from": utcnow().isoformat(),
        })
        delta.assertions.append({
            "subject_id": "e1", "property_id": "current_location",
            "value": TypedValue(raw={"frame": "cam_normalized", "position": [0.5, 0.5]}, value_type=ValueType.GEOMETRY),
            "epistemic_kind": "observation", "source_id": "test",
            "evidence_ids": ("ev:1",), "confidence": 0.8, "authority": 0.7,
            "valid_from": utcnow().isoformat(),
        })
        reconciler.ingest(delta)

        predictor = WorldStatePredictor(query)
        preds = predictor.predict("e1", horizon_seconds=30.0)
        assert len(preds) > 0
        assert all(isinstance(p, TypedPrediction) for p in preds)
        assert any(p.property == "current_location" for p in preds)


class TestObservabilityTransitions:
    """OBSERVED_ABSENT requires comparison against prior materialized state
    (what was this camera seeing before?), which only the reconciler has —
    the normalizer converts a single event and has no state to compare
    against, so it must never guess absence from event-local data alone.
    """

    @staticmethod
    def _vision_event(detections):
        from unittest.mock import MagicMock
        event = MagicMock()
        event.event_type = "vision"
        event.source_id = "vision:cam0"
        event.occurred_at = utcnow()
        event.payload = {"detections": detections}
        event.entity_ids = ()
        return event

    def test_normalizer_alone_does_not_infer_absence(self):
        from egg_companion.world.normalize import ObservationNormalizer
        from egg_companion.world.types import ObservabilityState
        normalizer = ObservationNormalizer()

        event = self._vision_event([
            {"entity_id": "e1", "label": "person", "confidence": 0.9,
             "bbox": [0, 0, 100, 100]},
        ])
        event.entity_ids = ("e1", "e2")  # e2 was expected but not detected

        delta = normalizer.normalize_event(event, evidence_ids=("ev:1",))
        obs_values = [
            a["value"].raw for a in delta.assertions
            if a.get("property_id") == "observability"
        ]
        assert ObservabilityState.OBSERVED_PRESENT.value in obs_values
        assert ObservabilityState.OBSERVED_ABSENT.value not in obs_values

    def test_reconciler_emits_absent_when_camera_stops_seeing_entity(self, world_stores):
        normalizer = world_stores["normalizer"]
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]

        frame1 = self._vision_event([
            {"entity_id": "e1", "label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
            {"entity_id": "e2", "label": "person", "confidence": 0.9, "bbox": [20, 20, 30, 30]},
        ])
        reconciler.ingest(normalizer.normalize_event(frame1, evidence_ids=("ev:1",)))
        assert json.loads(state.get_property("e1", "observability").value_json) == "observed_present"
        assert json.loads(state.get_property("e2", "observability").value_json) == "observed_present"

        # Second frame from the same camera no longer sees e2.
        frame2 = self._vision_event([
            {"entity_id": "e1", "label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        ])
        reconciler.ingest(normalizer.normalize_event(frame2, evidence_ids=("ev:2",)))
        assert json.loads(state.get_property("e1", "observability").value_json) == "observed_present"
        assert json.loads(state.get_property("e2", "observability").value_json) == "observed_absent"

    def test_reconciler_reinstates_present_after_absence(self, world_stores):
        normalizer = world_stores["normalizer"]
        reconciler = world_stores["reconciler"]
        state = world_stores["state"]

        seen = self._vision_event([
            {"entity_id": "e1", "label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        ])
        reconciler.ingest(normalizer.normalize_event(seen, evidence_ids=("ev:1",)))

        gone = self._vision_event([])
        reconciler.ingest(normalizer.normalize_event(gone, evidence_ids=("ev:2",)))
        assert json.loads(state.get_property("e1", "observability").value_json) == "observed_absent"

        back = self._vision_event([
            {"entity_id": "e1", "label": "person", "confidence": 0.9, "bbox": [0, 0, 10, 10]},
        ])
        reconciler.ingest(normalizer.normalize_event(back, evidence_ids=("ev:3",)))
        assert json.loads(state.get_property("e1", "observability").value_json) == "observed_present"
