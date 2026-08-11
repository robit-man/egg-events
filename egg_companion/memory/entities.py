from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any

import numpy as np

from egg_companion.memory.store import MemoryStore
from egg_companion.models import PerceptualEvent


class EntityResolver:
    """Conservative bridge from live profile IDs into graph entities and claims."""

    def __init__(self, store: MemoryStore) -> None:
        self.store = store

    def ensure_event_entities(self, event: PerceptualEvent) -> dict[str, float]:
        confidences: dict[str, float] = {}
        descriptors = event.payload.get("entities", ())
        if not isinstance(descriptors, (list, tuple)):
            return confidences
        for descriptor in descriptors:
            if not isinstance(descriptor, dict):
                continue
            entity_id = descriptor.get("id")
            entity_type = descriptor.get("type")
            if not isinstance(entity_id, str) or not entity_id or entity_type not in {
                "person", "face_observation", "appearance_track", "object", "object_category", "content"
            }:
                continue
            confidence = self._bounded_confidence(descriptor.get("confidence"))
            display_name = descriptor.get("label")
            if not isinstance(display_name, str) or not display_name.strip() or display_name == entity_id:
                display_name = None
            metadata = {
                key: value for key, value in descriptor.items()
                if key not in {"id", "type", "label", "confidence"} and self._json_scalar(value)
            }
            self.store.upsert_entity(entity_type, display_name, metadata, entity_id, now=event.occurred_at)
            if display_name:
                self.store.assert_claim_once(
                    entity_id, "has_alias", display_name, confidence, event.occurred_at,
                    source=str(descriptor.get("source") or "runtime"),
                )
            confidences[entity_id] = confidence
        return confidences

    def sync_identity_profile(
        self, profile: dict[str, object], evidence_id: str | None = None
    ) -> str:
        profile_id = str(profile["profile_id"])
        face_embedding = profile.get("face_embedding")
        entity_type = "person" if isinstance(face_embedding, np.ndarray) else "appearance_track"
        name = profile.get("name")
        display_name = str(name) if isinstance(name, str) and name.strip() else None
        last_seen = self._datetime(profile["last_seen"])
        metadata = {
            "source_system": "identity-library",
            "source_profile_id": profile_id,
            "identity_kind": str(profile.get("kind") or "appearance"),
            "face_confirmed": entity_type == "person",
            "samples": int(profile.get("samples") or 0),
            "sightings": int(profile.get("sightings") or 0),
            "last_camera": profile.get("last_camera"),
            "confidence": self._bounded_confidence(profile.get("confidence")),
        }
        self.store.upsert_entity(entity_type, display_name, metadata, profile_id, now=last_seen)
        if display_name:
            self.store.assert_claim_once(
                profile_id, "has_alias", display_name, 1.0, last_seen,
                source="user", evidence_id=evidence_id,
            )
        clip_embedding = profile.get("clip_embedding")
        if isinstance(clip_embedding, np.ndarray):
            self.store.add_embedding(
                "entity", profile_id, "appearance", "open-clip", clip_embedding,
                self._bounded_confidence(profile.get("confidence")), last_seen,
                embedding_id=f"legacy:identity:{profile_id}:clip",
            )
        if isinstance(face_embedding, np.ndarray):
            self.store.add_embedding(
                "entity", profile_id, "face", "opencv-sface", face_embedding,
                self._bounded_confidence(profile.get("confidence")), last_seen,
                embedding_id=f"legacy:identity:{profile_id}:face",
            )
        return profile_id

    def sync_object_profile(
        self, profile: dict[str, object], evidence_id: str | None = None
    ) -> str:
        profile_id = str(profile["profile_id"])
        label = str(profile["label"])
        last_seen = self._datetime(profile["last_seen"])
        label_source = str(profile.get("label_source") or "legacy")
        metadata = {
            "source_system": "object-library",
            "source_profile_id": profile_id,
            "samples": int(profile.get("samples") or 0),
            "detector_confidence": self._bounded_confidence(profile.get("confidence")),
            "label_source": label_source,
            "label_confidence": self._bounded_confidence(profile.get("label_confidence")),
            "review_state": str(profile.get("review_state") or "pending"),
            "label_provenance": profile.get("label_provenance")
            if isinstance(profile.get("label_provenance"), dict) else {},
        }
        self.store.upsert_entity("object", label, metadata, profile_id, now=last_seen)
        history = profile.get("label_history")
        if isinstance(history, list):
            for index, previous in enumerate(history):
                if not isinstance(previous, dict) or not isinstance(previous.get("label"), str):
                    continue
                revised_at = self._datetime(previous.get("revised_at") or last_seen)
                self.store.assert_claim(
                    profile_id, "has_label", str(previous["label"]),
                    self._bounded_confidence(previous.get("confidence")), revised_at,
                    claim_id=f"legacy:object:{profile_id}:label-history:{index}",
                    source=str(previous.get("source") or "legacy"), evidence_id=evidence_id,
                    metadata={"revised_by": previous.get("revised_by")}, state="superseded",
                )
        for claim in self.store.list_claims(profile_id, state="active", limit=self.configured_claim_limit()):
            if claim["predicate"] == "has_label" and claim["object_id_or_text"].casefold() != label.casefold():
                self.store.revise_claim(
                    str(claim["claim_id"]), "correct", label_source, label,
                    evidence_id, last_seen,
                )
        self.store.assert_claim_once(
            profile_id, "has_label", label,
            self._bounded_confidence(profile.get("label_confidence")), last_seen,
            source=label_source, evidence_id=evidence_id,
            metadata={"review_state": metadata["review_state"]},
        )
        embedding = profile.get("embedding")
        if isinstance(embedding, np.ndarray):
            self.store.add_embedding(
                "entity", profile_id, "masked-object", "open-clip", embedding,
                self._bounded_confidence(profile.get("confidence")), last_seen,
                embedding_id=f"legacy:object:{profile_id}:clip",
            )
        return profile_id

    def configured_claim_limit(self) -> int:
        return max(20, self.store.config.retrieval_limit)

    @staticmethod
    def media_checksum(payload: bytes) -> str:
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _bounded_confidence(value: object) -> float:
        try:
            return max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _datetime(value: object) -> datetime:
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        raise TypeError(f"expected datetime, received {type(value).__name__}")

    @staticmethod
    def _json_scalar(value: Any) -> bool:
        return value is None or isinstance(value, (str, int, float, bool))
