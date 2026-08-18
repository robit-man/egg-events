"""Shared typed records and enums for the operational world model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class ValueType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    ENUM = "enum"
    DATETIME = "datetime"
    DURATION = "duration"
    QUANTITY = "quantity"
    VECTOR = "vector"
    GEOMETRY = "geometry"
    ENTITY_REF = "entity_ref"
    JSON = "json"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class TypedValue:
    """A value with explicit type and optional unit."""

    raw: Any
    value_type: ValueType = ValueType.UNKNOWN
    unit: str | None = None

    def __post_init__(self) -> None:
        if self.value_type == ValueType.UNKNOWN and self.raw is not None:
            object.__setattr__(self, "value_type", _infer_type(self.raw))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TypedValue):
            return NotImplemented
        return self.raw == other.raw and self.value_type == other.value_type

    def __hash__(self) -> int:
        return hash((self.raw, self.value_type))


def _infer_type(raw: Any) -> ValueType:
    if isinstance(raw, bool):
        return ValueType.BOOLEAN
    if isinstance(raw, int):
        return ValueType.INTEGER
    if isinstance(raw, float):
        return ValueType.FLOAT
    if isinstance(raw, str):
        return ValueType.STRING
    if isinstance(raw, datetime):
        return ValueType.DATETIME
    if isinstance(raw, (list, tuple)):
        return ValueType.VECTOR
    if isinstance(raw, dict):
        return ValueType.JSON
    return ValueType.UNKNOWN


# ---------------------------------------------------------------------------
# Epistemic kinds
# ---------------------------------------------------------------------------


class EpistemicKind(str, Enum):
    OBSERVATION = "observation"
    CLAIM = "claim"
    HYPOTHESIS = "hypothesis"
    INFERENCE = "inference"
    CORRECTION = "correction"
    DERIVED = "derived"


# ---------------------------------------------------------------------------
# Observability states
# ---------------------------------------------------------------------------


class ObservabilityState(str, Enum):
    """Observability states for object permanence."""
    OBSERVED_PRESENT = "observed_present"
    OBSERVED_ABSENT = "observed_absent"
    NOT_OBSERVED = "not_observed"
    OCCLUDED = "occluded"
    OUTSIDE_COVERAGE = "outside_coverage"
    SENSOR_UNAVAILABLE = "sensor_unavailable"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Uncertainty decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Uncertainty:
    """Decomposed uncertainty for a world model assertion."""
    measurement: float = 0.0
    identity: float = 0.0
    classification: float = 0.0
    spatial: float = 0.0
    temporal: float = 0.0
    source_disagreement: float = 0.0
    staleness: float = 0.0

    @property
    def total(self) -> float:
        """Root mean square of all uncertainty components."""
        components = [
            self.measurement, self.identity, self.classification,
            self.spatial, self.temporal, self.source_disagreement, self.staleness,
        ]
        return (sum(c**2 for c in components) / len(components)) ** 0.5


# ---------------------------------------------------------------------------
# Evidence correlation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvidenceCorrelationGroup:
    """Groups correlated evidence to prevent independent confirmation inflation.
    
    Adjacent video frames, repeated measurements from the same sensor,
    or observations from the same session should be grouped so that
    confidence aggregation applies diminishing returns.
    """
    group_id: str
    session_id: str | None = None
    camera_id: str | None = None
    sensor_id: str | None = None
    start_time: str | None = None
    end_time: str | None = None
    observation_count: int = 1
    correlation_type: str = "temporal"  # temporal, spatial, sensor, session
    
    @property
    def independence_factor(self) -> float:
        """Returns a factor [0, 1] representing effective independent observations.
        
        For N correlated observations, effective independent count is:
        1 + (N-1) * factor where factor < 1 for correlated observations.
        """
        if self.observation_count <= 1:
            return 1.0
        
        # Diminishing returns based on correlation type
        factors = {
            "temporal": 0.1,   # Adjacent frames are highly correlated
            "spatial": 0.3,    # Nearby sensors have some independence
            "sensor": 0.2,     # Same sensor, different time
            "session": 0.5,    # Same session, different sensors
        }
        factor = factors.get(self.correlation_type, 0.1)
        
        effective = 1 + (self.observation_count - 1) * factor
        return effective / self.observation_count


# ---------------------------------------------------------------------------
# Assertion lifecycle
# ---------------------------------------------------------------------------


class AssertionState(str, Enum):
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    SUPERSEDED = "superseded"
    CONFLICTED = "conflicted"
    RETRACTED = "retracted"
    STALE = "stale"


class AssertionKind(str, Enum):
    PROPERTY = "property"
    RELATION = "relation"
    EVENT = "event"
    ACTION = "action"


# ---------------------------------------------------------------------------
# Ontology types
# ---------------------------------------------------------------------------


@dataclass
class ObjectType:
    id: str
    identity_strategy: str = "persistent_instance"
    temporal_mode: str = "current_state"
    expected_properties: tuple[str, ...] = ()
    description: str = ""


@dataclass
class PropertyType:
    id: str
    value_type: ValueType = ValueType.UNKNOWN
    cardinality: str = "one"
    unit: str | None = None
    description: str = ""
    volatility: str = "stable"
    stale_after: float | None = None
    decay_model: str = "none"  # none, linear, exponential


@dataclass
class RelationType:
    id: str
    domain: tuple[str, ...] = ()
    range: tuple[str, ...] = ()
    temporal: bool = True
    symmetric: bool = False
    transitive: bool = False
    persistence: str = "observation_dependent"
    stale_after: float | None = None
    max_count: int | None = None
    description: str = ""


@dataclass
class EventType:
    id: str
    roles: dict[str, str] = field(default_factory=dict)
    description: str = ""


@dataclass
class ActionType:
    id: str
    actor: str = "agent"
    parameters: dict[str, str] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    effects: tuple[str, ...] = ()
    approval_required: bool = False
    description: str = ""


@dataclass
class FunctionType:
    id: str
    inputs: tuple[str, ...] = ()
    output_type: ValueType = ValueType.UNKNOWN
    description: str = ""


@dataclass
class SourceType:
    id: str
    authority_class: str = "observation"
    description: str = ""


# ---------------------------------------------------------------------------
# Coordinate frames
# ---------------------------------------------------------------------------


class CoordinateFrame(str, Enum):
    CAMERA_PIXELS = "camera_pixels"
    CAMERA_NORMALIZED = "camera_normalized"
    CAMERA_OPTICAL = "camera_optical"
    EGG_BASE = "egg_base"
    ROOM_LOCAL = "room_local"
    MAP = "map"
    EARTH = "earth"


@dataclass
class SpatialState:
    position: tuple[float, ...]
    frame_id: str = CoordinateFrame.CAMERA_NORMALIZED
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    transform_confidence: float = 1.0


# ---------------------------------------------------------------------------
# World delta
# ---------------------------------------------------------------------------


@dataclass
class WorldDelta:
    """Output of ObservationNormalizer: the raw semantic changes to reconcile."""

    observations: list[dict[str, Any]] = field(default_factory=list)
    assertions: list[dict[str, Any]] = field(default_factory=list)
    relation_assertions: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    identity_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    # One entry per camera that produced a frame this tick, even with zero
    # detections — lets the reconciler diff "who's visible now" against
    # prior state to emit OBSERVED_ABSENT transitions.
    camera_frames: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Reconciliation output
# ---------------------------------------------------------------------------


@dataclass
class ResolvedState:
    value: Any = None
    confidence: float = 0.0
    assertion_ids: tuple[str, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# Action layer
# ---------------------------------------------------------------------------


@dataclass
class ActionProposal:
    proposal_id: str
    action_type: str
    target_entity_ids: tuple[str, ...] = ()
    inputs: dict[str, Any] = field(default_factory=dict)
    preconditions: tuple[str, ...] = ()
    expected_effects: tuple[str, ...] = ()
    source_evidence_ids: tuple[str, ...] = ()
    based_on_revision: int = 0
    status: str = "pending"  # pending, accepted, rejected
    reason: str = ""
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ActionExecution:
    execution_id: str
    proposal_id: str
    result: Any = None
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    success: bool | None = None
    evidence_ids: tuple[str, ...] = ()


@dataclass
class ActionOutcome:
    outcome_id: str
    execution_id: str
    success: bool
    result: Any = None
    side_effects: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    observed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Ontology proposal (LLM-generated, not directly applied)
# ---------------------------------------------------------------------------


@dataclass
class OntologyProposal:
    proposal_id: str
    kind: str  # "object_type", "property_type", "relation_type"
    definition: dict[str, Any] = field(default_factory=dict)
    source: str = "llm_inference"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    validated: bool = False
    accepted: bool = False
