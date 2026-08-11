from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    bbox: BoundingBox
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    camera_id: str
    timestamp: datetime
    detections: tuple[Detection, ...]
    semantic_labels: tuple[str, ...] = ()
    microphone_direction: float | None = None


@dataclass(frozen=True)
class AttentionTarget:
    track_id: str
    detection: Detection
    novelty: float
    priority: float
    reason: str
    camera_id: str
    timestamp: datetime


@dataclass(frozen=True)
class EvidenceRef:
    evidence_id: str
    modality: str
    captured_at: datetime
    source_type: str
    source_id: str
    media_key: str | None = None
    quality: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PerceptualEvent:
    event_id: str
    event_type: Literal[
        "vision", "speech", "object", "identity", "ocr", "user_correction", "attention"
    ]
    occurred_at: datetime
    source_id: str
    evidence: tuple[EvidenceRef, ...] = ()
    entity_ids: tuple[str, ...] = ()
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EpisodeDraft:
    episode_id: str
    context_key: str
    started_at: datetime
    ended_at: datetime
    evidence: tuple[EvidenceRef, ...]
    entity_ids: tuple[str, ...]
    surprise: dict[str, float] = field(default_factory=dict)
    summary: str | None = None


@dataclass(frozen=True)
class MemoryHit:
    owner_type: Literal["entity", "episode", "claim"]
    owner_id: str
    score: float
    confidence: float
    provenance: tuple[EvidenceRef, ...] = ()
    why: tuple[str, ...] = ()


@dataclass(frozen=True)
class AttentionDecision:
    capture_priority: float
    allow_outward_speech: bool
    components: dict[str, float]
    reason: str
    cooldown_seconds: float


@dataclass(frozen=True)
class InteractionDecision:
    allow_speech: bool
    reason: str
    response_fingerprint: str | None = None
