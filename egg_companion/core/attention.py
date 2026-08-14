from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from uuid import uuid4

from egg_companion.models import (
    AttentionTarget, Detection, GraphCognitiveSignal, Observation,
)


def intersection_over_union(first: Detection, second: Detection) -> float:
    left = max(first.bbox.x1, second.bbox.x1)
    top = max(first.bbox.y1, second.bbox.y1)
    right = min(first.bbox.x2, second.bbox.x2)
    bottom = min(first.bbox.y2, second.bbox.y2)
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    union = first.bbox.area + second.bbox.area - intersection
    return intersection / union if union else 0.0


@dataclass
class _Track:
    id: str
    detection: Detection
    camera_id: str
    last_seen: datetime
    seen_count: int = 1


class AttentionManager:
    """Session-local target selection without biometric identification or persistence."""

    def __init__(
        self,
        track_ttl_seconds: float,
        min_priority: float,
        max_targets: int = 1,
    ) -> None:
        self.track_ttl_seconds = track_ttl_seconds
        self.min_priority = min_priority
        self.max_targets = max_targets
        self._tracks: dict[str, _Track] = {}

    def select(
        self,
        observation: Observation,
        graph_feedback: dict[str, GraphCognitiveSignal] | None = None,
        observation_policy: dict[str, object] | None = None,
    ) -> list[AttentionTarget]:
        self._expire_tracks(observation.timestamp)
        targets = [
            self._score(
                detection,
                observation,
                graph_feedback or {},
                observation_policy or {},
            )
            for detection in observation.detections
        ]
        selected = [target for target in targets if target.priority >= self.min_priority]
        selected.sort(key=lambda target: target.priority, reverse=True)
        return selected[: self.max_targets]

    def _score(
        self,
        detection: Detection,
        observation: Observation,
        graph_feedback: dict[str, GraphCognitiveSignal],
        observation_policy: dict[str, object],
    ) -> AttentionTarget:
        matched = self._match(detection, observation.camera_id)
        if matched is None:
            track = _Track(str(uuid4()), detection, observation.camera_id, observation.timestamp)
            self._tracks[track.id] = track
            novelty = 1.0
            reason = "new person" if detection.label == "person" else f"new {detection.label}"
        else:
            prior_label = matched.detection.label
            behavior_changed = matched.detection.attributes.get("behavior") != detection.attributes.get("behavior")
            if prior_label != detection.label:
                stable_id = detection.attributes.get("identity_id") or detection.attributes.get("object_id")
                novelty = 0.65 if stable_id else 0.05
                reason = "resolved entity changed" if stable_id else "detector label unstable"
            elif behavior_changed:
                novelty = 0.15
                reason = "behavior changed"
            else:
                novelty = 0.0
                reason = "continuing"
            matched.detection = detection
            matched.last_seen = observation.timestamp
            matched.seen_count += 1
            track = matched

        shape = detection.attributes.get("frame_shape")
        frame_area = float(shape[0] * shape[1]) if isinstance(shape, list) and len(shape) == 2 else 1.0
        size = min(1.0, detection.bbox.area / max(frame_area * 0.2, 1.0))
        stable_id = str(
            detection.attributes.get("identity_id")
            or detection.attributes.get("object_id")
            or ""
        )
        signal = graph_feedback.get(stable_id)
        familiarity = signal.familiarity if signal else 0.0
        effective_novelty = novelty * (1.0 - 0.85 * familiarity)
        focus_entities = {
            str(value) for value in observation_policy.get("focus_entity_ids", []) if value
        }
        policy_relevant = bool(stable_id and stable_id in focus_entities)
        person_bonus = 0.25 * effective_novelty if detection.label == "person" else 0.0
        action_bonus = 0.2 if detection.attributes.get("behavior") in {"waving", "approaching"} else 0.0
        direction_bonus = 0.1 if detection.attributes.get("audio_aligned") is True else 0.0
        policy_bonus = 0.22 if policy_relevant else 0.0
        gap_bonus = 0.06 * signal.knowledge_gap if signal and policy_relevant else 0.0
        priority = min(
            1.0,
            0.45 * effective_novelty
            + 0.2 * size
            + person_bonus
            + action_bonus
            + direction_bonus
            + policy_bonus
            + gap_bonus,
        )
        if policy_relevant:
            reason += "; conversation-relevant"
        return AttentionTarget(
            track_id=track.id,
            detection=detection,
            novelty=novelty,
            priority=priority,
            reason=reason,
            camera_id=observation.camera_id,
            timestamp=observation.timestamp,
        )

    def _match(self, detection: Detection, camera_id: str) -> _Track | None:
        stable_id = detection.attributes.get("identity_id") or detection.attributes.get("object_id")
        candidates = [
            track
            for track in self._tracks.values()
            if track.camera_id == camera_id
            and (
                stable_id
                and stable_id == (
                    track.detection.attributes.get("identity_id")
                    or track.detection.attributes.get("object_id")
                )
                or not stable_id
            )
            and intersection_over_union(detection, track.detection) >= 0.35
        ]
        return max(candidates, key=lambda track: intersection_over_union(detection, track.detection), default=None)

    def _expire_tracks(self, now: datetime) -> None:
        self._tracks = {
            track_id: track
            for track_id, track in self._tracks.items()
            if (now - track.last_seen).total_seconds() <= self.track_ttl_seconds
        }
