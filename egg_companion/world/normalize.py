"""ObservationNormalizer: converts PerceptualEvent into WorldDelta.

All normalization of raw perceptual events into structured world assertions
flows through this module.  The pipeline should never manually construct
WorldDelta with label/bbox/behavior semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from egg_companion.world.ontology import OntologyRegistry
from egg_companion.world.sources import AuthorityPolicy
from egg_companion.world.types import (
    EpistemicKind,
    ObservabilityState,
    TypedValue,
    ValueType,
    WorldDelta,
)


class ObservationNormalizer:
    """Converts raw perceptual events into structured WorldDelta.

    Uses AuthorityPolicy to compute authority for each assertion.
    Accepts evidence_ids from the caller so that provenance is never lost.
    """

    def __init__(
        self,
        authority_policy: AuthorityPolicy | None = None,
        ontology: OntologyRegistry | None = None,
        min_confidence: float = 0.45,
    ) -> None:
        self._authority = authority_policy or AuthorityPolicy()
        self._ontology = ontology or OntologyRegistry()
        self._min_confidence = min_confidence

    def normalize_event(
        self,
        event: Any,
        *,
        evidence_ids: tuple[str, ...] = (),
        confidences: dict[str, float] | None = None,
        frame_shape: tuple[int, int] | None = None,
    ) -> WorldDelta:
        """Normalize a PerceptualEvent into a WorldDelta.

        This is the single entry point for all normalization.  The pipeline
        should call this and nothing else.
        """
        confidences = confidences or {}
        source_id = getattr(event, "source_id", "unknown")
        source_type = self._source_type_from_id(source_id)
        occurred_at = getattr(event, "occurred_at", datetime.now(timezone.utc))
        if isinstance(occurred_at, datetime):
            occurred_at = occurred_at.isoformat()
        event_type = getattr(event, "event_type", "")
        payload = getattr(event, "payload", {})
        entity_ids = getattr(event, "entity_ids", ())

        if event_type in ("vision", "object", "identity"):
            return self._normalize_visual_event(
                payload, source_id, source_type, occurred_at,
                evidence_ids=evidence_ids, confidences=confidences,
                frame_shape=frame_shape, entity_ids=entity_ids,
            )
        elif event_type == "speech":
            return self._normalize_speech_event(
                payload, source_id, source_type, occurred_at,
                evidence_ids=evidence_ids, confidences=confidences,
                entity_ids=entity_ids,
            )
        elif event_type == "ocr":
            return self._normalize_ocr_event(
                payload, source_id, source_type, occurred_at,
                evidence_ids=evidence_ids, confidences=confidences,
                entity_ids=entity_ids,
            )
        else:
            return WorldDelta()

    def _normalize_visual_event(
        self,
        payload: dict[str, Any],
        source_id: str,
        source_type: str,
        observed_at: str,
        *,
        evidence_ids: tuple[str, ...] = (),
        confidences: dict[str, float],
        frame_shape: tuple[int, int] | None,
        entity_ids: tuple[str, ...] = (),
    ) -> WorldDelta:
        delta = WorldDelta()
        detections = payload.get("detections", [])
        if not isinstance(detections, (list, tuple)):
            return delta

        camera_id = source_id.split(":")[-1] if ":" in source_id else source_id
        visible_entity_ids: set[str] = set()

        for detection in detections:
            if not isinstance(detection, dict):
                continue

            label = detection.get("label", "unknown")
            entity_id = (
                detection.get("entity_id")
                or detection.get("object_id")
                or detection.get("identity_id")
                or (f"det:{label}" if label and label != "unknown" else None)
            )
            if not entity_id:
                continue

            confidence = float(detection.get("confidence", 0.0))
            bbox = detection.get("bbox")
            behavior = detection.get("behavior")

            # Filter: reject detections below confidence threshold
            if confidence < self._min_confidence:
                continue

            authority = self._authority.evaluate(
                property_type=f"{self._entity_type_from_label(label)}.label",
                source_type=source_type,
                epistemic_kind=EpistemicKind.OBSERVATION.value,
            )

            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "label",
                "value": TypedValue(raw=label, value_type=ValueType.STRING),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "authority": authority,
                "valid_from": observed_at,
            })

            if bbox:
                if isinstance(bbox, dict):
                    bbox = [bbox.get("x1", 0), bbox.get("y1", 0), bbox.get("x2", 0), bbox.get("y2", 0)]
                bbox_authority = self._authority.evaluate(
                    property_type="*.bbox",
                    source_type=source_type,
                    epistemic_kind=EpistemicKind.OBSERVATION.value,
                )
                delta.assertions.append({
                    "subject_id": entity_id,
                    "property_id": "bbox",
                    "value": TypedValue(raw=bbox, value_type=ValueType.GEOMETRY),
                    "epistemic_kind": EpistemicKind.OBSERVATION.value,
                    "source_id": source_id,
                    "evidence_ids": evidence_ids,
                    "confidence": confidence,
                    "authority": bbox_authority,
                    "valid_from": observed_at,
                })

            if behavior:
                behavior_authority = self._authority.evaluate(
                    property_type="*.behavior",
                    source_type=source_type,
                    epistemic_kind=EpistemicKind.OBSERVATION.value,
                )
                delta.assertions.append({
                    "subject_id": entity_id,
                    "property_id": "behavior",
                    "value": TypedValue(raw=behavior, value_type=ValueType.STRING),
                    "epistemic_kind": EpistemicKind.OBSERVATION.value,
                    "source_id": source_id,
                    "evidence_ids": evidence_ids,
                    "confidence": confidence,
                    "authority": behavior_authority,
                    "valid_from": observed_at,
                })

            if bbox and frame_shape:
                h, w = frame_shape
                center_x = ((bbox[0] + bbox[2]) / 2) / w
                center_y = ((bbox[1] + bbox[3]) / 2) / h
                loc_authority = self._authority.evaluate(
                    property_type="*.current_location",
                    source_type=source_type,
                    epistemic_kind=EpistemicKind.OBSERVATION.value,
                )
                delta.assertions.append({
                    "subject_id": entity_id,
                    "property_id": "current_location",
                    "value": TypedValue(
                        raw={"frame": f"{source_id.split(':')[-1]}_normalized",
                             "position": [round(center_x, 4), round(center_y, 4)]},
                        value_type=ValueType.GEOMETRY,
                    ),
                    "epistemic_kind": EpistemicKind.OBSERVATION.value,
                    "source_id": source_id,
                    "evidence_ids": evidence_ids,
                    "confidence": confidence * 0.8,
                    "authority": loc_authority,
                    "valid_from": observed_at,
                })

            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "last_seen",
                "value": TypedValue(raw=observed_at, value_type=ValueType.DATETIME),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "authority": 0.9,
                "valid_from": observed_at,
            })

            delta.assertions.append({
                "subject_id": entity_id,
                "property_id": "observability",
                "value": TypedValue(
                    raw=ObservabilityState.OBSERVED_PRESENT.value,
                    value_type=ValueType.ENUM,
                ),
                "epistemic_kind": EpistemicKind.OBSERVATION.value,
                "source_id": source_id,
                "evidence_ids": evidence_ids,
                "confidence": confidence,
                "authority": 0.9,
                "valid_from": observed_at,
            })
            visible_entity_ids.add(entity_id)

            delta.relation_assertions.append({
                "source_entity_id": entity_id,
                "relation_type_id": "visible_from",
                "target_entity_id": f"camera_view:{camera_id}",
                "confidence": confidence,
                "authority": self._authority.evaluate(
                    property_type="*.visible_from",
                    source_type=source_type,
                    epistemic_kind=EpistemicKind.OBSERVATION.value,
                ),
                "source_id": source_id,
                "evidence_ids": evidence_ids,
                "valid_from": observed_at,
            })

        # Record that this camera produced a frame this tick — even with
        # zero detections — so the Reconciler can diff against prior state
        # to emit OBSERVED_ABSENT.  This must not be derived from
        # relation_assertions alone: a frame with zero detections has none,
        # but still means "the camera looked and saw nobody", which is
        # exactly the case absence-tracking needs to catch.
        #
        # OBSERVED_ABSENT is intentionally not emitted here.  This method is
        # stateless — it only sees entities present in *this* event — so it
        # cannot correctly tell "previously visible, now missing" apart from
        # "never visible to begin with".  That diff requires querying prior
        # world state, which only the Reconciler has access to; see
        # Reconciler._emit_absence_transitions.
        delta.camera_frames.append({
            "camera_id": f"camera_view:{camera_id}",
            "visible_entity_ids": sorted(visible_entity_ids),
            "source_id": source_id,
            "evidence_ids": evidence_ids,
            "valid_from": observed_at,
        })

        return delta

    def _normalize_speech_event(
        self,
        payload: dict[str, Any],
        source_id: str,
        source_type: str,
        observed_at: str,
        *,
        evidence_ids: tuple[str, ...] = (),
        confidences: dict[str, float],
        entity_ids: tuple[str, ...] = (),
    ) -> WorldDelta:
        delta = WorldDelta()
        transcript = payload.get("transcript", "")
        if not transcript:
            return delta

        speaker_id = entity_ids[0] if entity_ids else payload.get("speaker", "unknown")
        authority = self._authority.evaluate(
            property_type="conversation_turn.transcript",
            source_type=source_type,
            epistemic_kind=EpistemicKind.OBSERVATION.value,
        )

        delta.events.append({
            "event_type_id": "speech_utterance",
            "roles": {"speaker": speaker_id, "transcript": str(transcript)[:500]},
            "source_id": source_id,
            "evidence_ids": evidence_ids,
            "confidence": confidences.get("transcript", 0.8),
            "observed_at": observed_at,
        })

        if speaker_id and speaker_id != "unknown":
            delta.relation_assertions.append({
                "source_entity_id": speaker_id,
                "relation_type_id": "speaking_to",
                "target_entity_id": "agent:egg",
                "confidence": 0.8,
                "authority": authority,
                "source_id": source_id,
                "evidence_ids": evidence_ids,
                "valid_from": observed_at,
            })

        return delta

    def _normalize_ocr_event(
        self,
        payload: dict[str, Any],
        source_id: str,
        source_type: str,
        observed_at: str,
        *,
        evidence_ids: tuple[str, ...] = (),
        confidences: dict[str, float],
        entity_ids: tuple[str, ...] = (),
    ) -> WorldDelta:
        delta = WorldDelta()
        text = payload.get("text", "")
        if not text:
            return delta

        target_id = str(
            payload.get("target_id")
            or (entity_ids[0] if entity_ids else "unknown")
        )
        authority = self._authority.evaluate(
            property_type="physical_object.label",
            source_type=source_type,
            epistemic_kind=EpistemicKind.OBSERVATION.value,
        )

        # Determine text type: static (signs, books) vs dynamic (screens, clocks)
        text_type = payload.get("text_type", "static")
        is_dynamic = text_type == "dynamic"

        raw_ocr_confidence = payload.get("ocr_confidence")
        text_confidence = (
            float(raw_ocr_confidence)
            if isinstance(raw_ocr_confidence, (int, float))
            else confidences.get("text", 0.6)
        )

        delta.events.append({
            "event_type_id": "ocr_detection",
            "roles": {
                "target": target_id,
                "transcript": str(text)[:500],
            },
            "source_id": source_id,
            "evidence_ids": evidence_ids,
            "confidence": text_confidence,
            "observed_at": observed_at,
        })

        # Static text → visible_text (persistent until contradicted)
        # Dynamic text → displays_text (stale_after 30s from ontology)
        property_id = "displays_text" if is_dynamic else "visible_text"
        valid_for_seconds = payload.get("valid_for_seconds")
        if is_dynamic and valid_for_seconds is None:
            valid_for_seconds = 30.0

        assertion: dict[str, Any] = {
            "subject_id": target_id,
            "property_id": property_id,
            "value": TypedValue(raw=text, value_type=ValueType.STRING),
            "epistemic_kind": EpistemicKind.OBSERVATION.value,
            "source_id": source_id,
            "evidence_ids": evidence_ids,
            "confidence": text_confidence,
            "authority": authority,
            "valid_from": observed_at,
        }
        if valid_for_seconds is not None:
            assertion["valid_for_seconds"] = valid_for_seconds

        delta.assertions.append(assertion)

        return delta

    @staticmethod
    def _source_type_from_id(source_id: str) -> str:
        if ":" not in source_id:
            return source_id
        return source_id.split(":", 1)[0]

    @staticmethod
    def _entity_type_from_label(label: str) -> str:
        label_lower = label.lower()
        if label_lower in ("person", "man", "woman", "child", "egg"):
            return "person"
        return "physical_object"

    def merge_deltas(self, *deltas: WorldDelta) -> WorldDelta:
        merged = WorldDelta()
        for d in deltas:
            merged.observations.extend(d.observations)
            merged.assertions.extend(d.assertions)
            merged.relation_assertions.extend(d.relation_assertions)
            merged.events.extend(d.events)
            merged.identity_hypotheses.extend(d.identity_hypotheses)
            merged.camera_frames.extend(d.camera_frames)
        return merged
