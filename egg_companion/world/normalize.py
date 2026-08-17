"""ObservationNormalizer: converts PerceptualEvent into WorldDelta."""

from __future__ import annotations

from typing import Any

from egg_companion.world.types import TypedValue, ValueType, WorldDelta


class ObservationNormalizer:
    """Converts raw perceptual events into structured WorldDelta."""

    def normalize_detection(
        self,
        detection: dict[str, Any],
        camera_id: str,
        observed_at: str,
        frame_shape: tuple[int, int] | None = None,
    ) -> WorldDelta:
        """Normalize a single detection into a WorldDelta."""
        delta = WorldDelta()
        entity_id = (
            detection.get("identity_id")
            or detection.get("object_id")
            or f"detected:{detection.get('label', 'unknown')}"
        )
        label = detection.get("label", "unknown")
        confidence = float(detection.get("confidence", 0.0))
        bbox = detection.get("bbox")
        behavior = detection.get("behavior")

        delta.observations.append({
            "entity_id": entity_id,
            "label": label,
            "confidence": confidence,
            "camera_id": camera_id,
            "observed_at": observed_at,
        })

        delta.assertions.append({
            "subject_id": entity_id,
            "property_id": "label",
            "value": TypedValue(raw=label, value_type=ValueType.STRING),
            "epistemic_kind": "observation",
            "source_id": f"camera:{camera_id}",
            "confidence": confidence,
            "valid_from": observed_at,
        })

        if bbox:
            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "bbox",
                "value": TypedValue(raw=bbox, value_type=ValueType.GEOMETRY),
                "epistemic_kind": "observation",
                "source_id": f"camera:{camera_id}",
                "confidence": confidence,
                "valid_from": observed_at,
            })

        if behavior:
            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "behavior",
                "value": TypedValue(raw=behavior, value_type=ValueType.STRING),
                "epistemic_kind": "observation",
                "source_id": f"camera:{camera_id}",
                "confidence": confidence,
                "valid_from": observed_at,
            })

        if bbox and frame_shape:
            h, w = frame_shape
            center_x = ((bbox[0] + bbox[2]) / 2) / w
            center_y = ((bbox[1] + bbox[3]) / 2) / h
            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "current_location",
                "value": TypedValue(
                    raw={"frame": f"{camera_id}_normalized", "position": [round(center_x, 4), round(center_y, 4)]},
                    value_type=ValueType.GEOMETRY,
                ),
                "epistemic_kind": "observation",
                "source_id": f"camera:{camera_id}",
                "confidence": confidence * 0.8,
                "valid_from": observed_at,
            })

        delta.relation_assertions.append({
            "source_entity_id": entity_id,
            "relation_type_id": "visible_from",
            "target_entity_id": f"camera_view:{camera_id}",
            "confidence": confidence,
            "source_id": f"camera:{camera_id}",
            "valid_from": observed_at,
        })

        return delta

    def normalize_speech(
        self,
        speaker_id: str | None,
        transcript: str,
        observed_at: str,
        source_id: str = "asr",
    ) -> WorldDelta:
        """Normalize a speech utterance into a WorldDelta."""
        delta = WorldDelta()
        turn_id = f"turn:{observed_at}"
        delta.events.append({
            "event_type_id": "speech_utterance",
            "roles": {"speaker": speaker_id or "unknown", "transcript": transcript},
            "source_id": source_id,
            "observed_at": observed_at,
        })
        if speaker_id:
            delta.relation_assertions.append({
                "source_entity_id": speaker_id,
                "relation_type_id": "speaking_to",
                "target_entity_id": "agent:egg",
                "confidence": 0.8,
                "source_id": source_id,
                "valid_from": observed_at,
            })
        return delta

    def merge_deltas(self, *deltas: WorldDelta) -> WorldDelta:
        merged = WorldDelta()
        for d in deltas:
            merged.observations.extend(d.observations)
            merged.assertions.extend(d.assertions)
            merged.relation_assertions.extend(d.relation_assertions)
            merged.events.extend(d.events)
            merged.identity_hypotheses.extend(d.identity_hypotheses)
        return merged
