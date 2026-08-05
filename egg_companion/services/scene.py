from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from egg_companion.core.attention import intersection_over_union
from egg_companion.models import Detection, Observation


@dataclass
class SceneItem:
    track_id: str
    camera_id: str
    detection: Detection
    first_seen: datetime
    last_seen: datetime
    sightings: int = 1
    label_override: str | None = None
    correction_asked: bool = False

    @property
    def label(self) -> str:
        return self.label_override or self.detection.label


class SceneInventory:
    """Tracks physical objects over time instead of counting detector frames."""

    def __init__(self, ttl_seconds: float = 20.0) -> None:
        self.ttl_seconds = ttl_seconds
        self._items: dict[str, SceneItem] = {}
        self._pending_correction: str | None = None

    def update(self, observation: Observation) -> None:
        self._expire(observation.timestamp)
        for detection in observation.detections:
            item = self._match(observation.camera_id, detection)
            if item is None:
                item = SceneItem(str(uuid4()), observation.camera_id, detection, observation.timestamp, observation.timestamp)
                self._items[item.track_id] = item
            else:
                item.detection = detection
                item.last_seen = observation.timestamp
                item.sightings += 1

    def next_uncertain(self) -> SceneItem | None:
        if self._pending_correction is not None:
            return None
        candidates = [
            item
            for item in self._items.values()
            if not item.correction_asked and item.sightings >= 3 and 0.45 <= item.detection.confidence < 0.70
        ]
        if not candidates:
            return None
        candidate = max(candidates, key=lambda item: (item.sightings, item.detection.confidence))
        candidate.correction_asked = True
        self._pending_correction = candidate.track_id
        return candidate

    def pending(self) -> SceneItem | None:
        return self._items.get(self._pending_correction) if self._pending_correction else None

    def resolve_pending(self, decision: str, label: str | None = None) -> SceneItem | None:
        item = self.pending()
        if item is None or decision not in {"confirm", "correct"}:
            return None
        if decision == "correct" and label:
            item.label_override = label
        self._pending_correction = None
        return item

    def dismiss_pending(self) -> SceneItem | None:
        item = self.pending()
        self._pending_correction = None
        return item

    def snapshot(self) -> list[dict[str, object]]:
        counts: dict[str, int] = {}
        for item in self._items.values():
            counts[item.label] = counts.get(item.label, 0) + 1
        return [
            {"label": label, "count": count}
            for label, count in sorted(counts.items(), key=lambda entry: (-entry[1], entry[0]))
        ]

    def _match(self, camera_id: str, detection: Detection) -> SceneItem | None:
        candidates = [
            item
            for item in self._items.values()
            if item.camera_id == camera_id and intersection_over_union(item.detection, detection) >= 0.35
        ]
        return max(candidates, key=lambda item: intersection_over_union(item.detection, detection), default=None)

    def _expire(self, now: datetime) -> None:
        expired = [
            track_id
            for track_id, item in self._items.items()
            if (now - item.last_seen).total_seconds() > self.ttl_seconds
        ]
        for track_id in expired:
            self._items.pop(track_id, None)
            if self._pending_correction == track_id:
                self._pending_correction = None
