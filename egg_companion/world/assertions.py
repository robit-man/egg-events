"""Typed propositions: WorldAssertion and RelationAssertion."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from egg_companion.world.temporal import BitemporalInterval, utcnow
from egg_companion.world.types import (
    AssertionKind,
    AssertionState,
    EpistemicKind,
    TypedValue,
)


@dataclass
class WorldAssertion:
    """A typed property assertion about an entity in the world."""

    assertion_id: str
    subject_id: str
    property_id: str
    value: TypedValue
    epistemic_kind: EpistemicKind
    source_id: str
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    authority: float = 0.0
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    observed_at: datetime = field(default_factory=utcnow)
    recorded_at: datetime = field(default_factory=utcnow)
    state: AssertionState = AssertionState.PROPOSED
    revision_of: str | None = None
    ontology_revision: int = 1
    kind: AssertionKind = AssertionKind.PROPERTY

    @property
    def is_current(self) -> bool:
        return self.state in {AssertionState.ACCEPTED, AssertionState.CONFLICTED}

    def supersedes(self, new_assertion: WorldAssertion) -> WorldAssertion:
        """Mark this assertion as superseded by the new one."""
        self.state = AssertionState.SUPERSEDED
        self.valid_to = utcnow()
        return new_assertion


@dataclass
class RelationAssertion:
    """A typed relationship between two entities."""

    assertion_id: str
    source_entity_id: str
    relation_type_id: str
    target_entity_id: str
    epistemic_kind: EpistemicKind
    source_id: str
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    authority: float = 0.0
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    observed_at: datetime = field(default_factory=utcnow)
    recorded_at: datetime = field(default_factory=utcnow)
    state: AssertionState = AssertionState.PROPOSED
    revision_of: str | None = None
    ontology_revision: int = 1
    kind: AssertionKind = AssertionKind.RELATION

    @property
    def is_current(self) -> bool:
        return self.state in {AssertionState.ACCEPTED, AssertionState.CONFLICTED}

    def close(self, at: datetime | None = None) -> None:
        self.valid_to = at or utcnow()
        self.state = AssertionState.SUPERSEDED


@dataclass
class EventAssertion:
    """A typed event occurrence with role bindings."""

    assertion_id: str
    event_type_id: str
    roles: dict[str, str] = field(default_factory=dict)
    epistemic_kind: EpistemicKind = EpistemicKind.OBSERVATION
    source_id: str = ""
    evidence_ids: tuple[str, ...] = ()
    confidence: float = 0.0
    valid_from: datetime = field(default_factory=utcnow)
    valid_to: datetime | None = None
    observed_at: datetime = field(default_factory=utcnow)
    recorded_at: datetime = field(default_factory=utcnow)
    state: AssertionState = AssertionState.PROPOSED
    ontology_revision: int = 1
    kind: AssertionKind = AssertionKind.EVENT
