"""Operational world model for Egg.

Typed, continuously reconciled world state layered over the existing
evidence-backed memory graph.  The world model never replaces raw
evidence — it interprets it.
"""

from egg_companion.world.types import (
    ActionType,
    AssertionKind,
    AssertionState,
    CoordinateFrame,
    EpistemicKind,
    EvidenceCorrelationGroup,
    ObservabilityState,
    ObjectType,
    PropertyType,
    RelationType,
    SourceType,
    TypedValue,
    Uncertainty,
    ValueType,
)
from egg_companion.world.assertions import EventAssertion, RelationAssertion, WorldAssertion
from egg_companion.world.ontology import OntologyRegistry
from egg_companion.world.sources import AuthorityPolicy, SourceRecord
from egg_companion.world.temporal import BitemporalInterval, utcnow
from egg_companion.world.state import WorldStateStore
from egg_companion.world.normalize import ObservationNormalizer
from egg_companion.world.reconcile import Reconciler
from egg_companion.world.relations import WorldGraphStore
from egg_companion.world.identity import IdentityGraph
from egg_companion.world.spatial import BBox2D, SpatialReasoner, SpatialState
from egg_companion.world.events import EventStore
from egg_companion.world.functions import FunctionRegistry
from egg_companion.world.actions import ActionStore
from egg_companion.world.policy import PolicyValidator
from egg_companion.world.query import WorldQuery
from egg_companion.world.context import CognitiveContext
from egg_companion.world.metrics import MetricsCollector
from egg_companion.world.bridge import KnowledgeGraphBridge

__all__ = [
    "ActionType",
    "AssertionKind",
    "AssertionState",
    "CoordinateFrame",
    "EpistemicKind",
    "EvidenceCorrelationGroup",
    "ObservabilityState",
    "ObjectType",
    "PropertyType",
    "RelationType",
    "SourceType",
    "TypedValue",
    "Uncertainty",
    "ValueType",
    "EventAssertion",
    "RelationAssertion",
    "WorldAssertion",
    "OntologyRegistry",
    "AuthorityPolicy",
    "SourceRecord",
    "BitemporalInterval",
    "utcnow",
    "WorldStateStore",
    "ObservationNormalizer",
    "Reconciler",
    "WorldGraphStore",
    "IdentityGraph",
    "BBox2D",
    "SpatialReasoner",
    "SpatialState",
    "EventStore",
    "FunctionRegistry",
    "ActionStore",
    "PolicyValidator",
    "WorldQuery",
    "CognitiveContext",
    "MetricsCollector",
    "KnowledgeGraphBridge",
]
