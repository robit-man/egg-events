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
    query = WorldQuery(state, graph, identity)
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
        normalizer = world_stores["normalizer"]
        detection = {
            "label": "person",
            "confidence": 0.85,
            "bbox": [10, 20, 100, 200],
            "behavior": "standing",
        }
        delta = normalizer.normalize_detection(detection, "cam0", utcnow().isoformat(), (480, 640))
        assert len(delta.assertions) >= 3
        assert delta.assertions[0]["subject_id"] is not None

    def test_normalize_speech(self, world_stores):
        normalizer = world_stores["normalizer"]
        delta = normalizer.normalize_speech("person:1", "Hello Egg", utcnow().isoformat())
        assert len(delta.events) == 1
        assert delta.events[0]["event_type_id"] == "speech_utterance"


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
