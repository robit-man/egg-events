from __future__ import annotations

import json
import shutil
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from egg_companion.adapters.vision import SegmentedObject, VisionEngine
from egg_companion.config import ObjectLearningConfig


@dataclass
class ObjectProfile:
    profile_id: str
    label: str
    embedding: np.ndarray
    confidence: float
    samples: int
    first_seen: datetime
    last_seen: datetime
    thumbnail: bytes | None = None
    label_source: str = "user"
    label_confidence: float = 1.0
    review_state: str = "pending"
    label_history: list[dict[str, object]] = field(default_factory=list)
    label_provenance: dict[str, object] = field(default_factory=dict)
    last_match_state: str | None = None
    last_reviewed_at: datetime | None = None
    audit_state: str | None = None
    audit_notes: str | None = None
    appearance_description: str | None = None
    adjudication_history: list[dict[str, object]] = field(default_factory=list)
    merged_into: str | None = None


class ObjectLibrary:
    """On-device CLIP library of user-labelled, SAM-segmented objects."""

    def __init__(self, config: ObjectLearningConfig) -> None:
        self.config = config
        self._directory = Path(config.storage_dir)
        self._lock = threading.Lock()
        self._profiles: dict[str, ObjectProfile] = {}
        self._dashboard_snapshot_cache: list[dict[str, object]] = []
        if config.enabled:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._load()

    def dashboard_snapshot(self) -> list[dict[str, object]]:
        """Return the last coherent card projection while a profile is updating."""
        if not self._lock.acquire(blocking=False):
            return [dict(item) for item in self._dashboard_snapshot_cache]
        try:
            rows = [
                {
                    "id": profile.profile_id,
                    "label": profile.label,
                    "confidence": round(profile.confidence, 3),
                    "samples": profile.samples,
                    "first_seen": profile.first_seen.isoformat(),
                    "last_seen": profile.last_seen.isoformat(),
                    "thumbnail_url": f"/api/objects/{profile.profile_id}/mask.png",
                    "label_source": profile.label_source,
                    "label_confidence": round(profile.label_confidence, 3),
                    "review_state": profile.review_state,
                    "last_match_state": profile.last_match_state,
                    "label_history_count": len(profile.label_history),
                    "appearance_description": str(
                        profile.appearance_description or ""
                    )[:240],
                }
                for profile in sorted(
                    self._profiles.values(), key=lambda item: item.last_seen, reverse=True
                )
                if profile.merged_into is None and profile.review_state != "rejected"
            ]
            self._dashboard_snapshot_cache = [dict(item) for item in rows]
            return rows
        finally:
            self._lock.release()

    def learn(
        self, label: str, segmented: SegmentedObject, vision: VisionEngine, label_source: str = "user",
        label_confidence: float = 1.0, label_provenance: dict[str, object] | None = None,
        *, force_new: bool = False, appearance_description: str | None = None,
        adjudication: dict[str, object] | None = None,
    ) -> ObjectProfile | None:
        normalized_label = " ".join(label.strip().split())
        if not self.config.enabled or not normalized_label or len(normalized_label) > 64:
            return None
        embedding = vision.embed_image(self._masked_image(segmented.image, segmented.mask))
        now = datetime.now(timezone.utc)
        with self._lock:
            existing = next(
                (
                    item
                    for item in self._profiles.values()
                    if not force_new
                    and item.merged_into is None
                    and item.review_state != "rejected"
                    and item.label.casefold() == normalized_label.casefold()
                ),
                None,
            )
            if existing is None:
                profile_id = f"object-{max(self._profile_numbers(), default=0) + 1:03d}"
                profile = ObjectProfile(
                    profile_id, normalized_label, embedding, segmented.confidence, 1, now, now,
                    label_source=label_source, label_confidence=label_confidence,
                    review_state=self._verified_state(label_source),
                    label_provenance=dict(label_provenance or {}),
                    last_reviewed_at=now if label_source in {"user", "ornith-vlm"} else None,
                    appearance_description=self._description(appearance_description),
                    adjudication_history=(
                        [dict(adjudication)] if isinstance(adjudication, dict) else []
                    ),
                )
                self._profiles[profile_id] = profile
            else:
                profile = existing
                profile.embedding = self._normalized((profile.embedding * profile.samples) + embedding)
                profile.samples += 1
                profile.confidence = max(profile.confidence, segmented.confidence)
                profile.last_seen = now
                profile.label_source = label_source
                profile.label_confidence = label_confidence
                profile.label_provenance = dict(label_provenance or profile.label_provenance)
                if label_source in {"user", "ornith-vlm"}:
                    profile.review_state = self._verified_state(label_source)
                    profile.last_reviewed_at = now
                if self._description(appearance_description):
                    profile.appearance_description = self._description(
                        appearance_description
                    )
                if isinstance(adjudication, dict):
                    profile.adjudication_history.append(dict(adjudication))
                    profile.adjudication_history = profile.adjudication_history[-32:]
            self._save(profile, segmented.image, segmented.mask)
            return profile

    def confirm_match(
        self,
        profile_id: str,
        label: str,
        segmented: SegmentedObject,
        vision: VisionEngine,
        confidence: float,
        *,
        model_id: str,
        appearance_description: str | None,
        provenance: dict[str, object] | None = None,
        adjudication: dict[str, object] | None = None,
    ) -> ObjectProfile | None:
        """Update a profile only after a pixel-grounded identity adjudication."""

        normalized_label = " ".join(label.strip().split())
        if not normalized_label or len(normalized_label) > 64:
            return None
        embedding = vision.embed_image(
            self._masked_image(segmented.image, segmented.mask)
        )
        now = datetime.now(timezone.utc)
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None or profile.merged_into is not None:
                return None
            if profile.label.casefold() != normalized_label.casefold():
                profile.label_history.append(
                    {
                        "label": profile.label,
                        "source": profile.label_source,
                        "confidence": profile.label_confidence,
                        "revised_at": now.isoformat(),
                        "revised_by": model_id,
                    }
                )
                profile.label = normalized_label
            profile.embedding = self._normalized(
                (profile.embedding * profile.samples) + embedding
            )
            profile.samples += 1
            profile.confidence = max(profile.confidence, segmented.confidence)
            profile.last_seen = now
            profile.label_source = "ornith-vlm"
            profile.label_confidence = max(0.0, min(1.0, float(confidence)))
            profile.review_state = "vlm_verified"
            profile.last_reviewed_at = now
            profile.last_match_state = "vlm_confirmed"
            profile.label_provenance = {
                **dict(provenance or {}),
                "model_id": model_id,
                "accepted_at": now.isoformat(),
            }
            description = self._description(appearance_description)
            if description:
                profile.appearance_description = description
            if isinstance(adjudication, dict):
                profile.adjudication_history.append(dict(adjudication))
                profile.adjudication_history = profile.adjudication_history[-32:]
            self._save(profile, segmented.image, segmented.mask)
            return profile

    def match(self, segmented: SegmentedObject, vision: VisionEngine) -> tuple[ObjectProfile, float] | None:
        if not self.config.enabled:
            return None
        embedding = vision.embed_image(self._masked_image(segmented.image, segmented.mask))
        with self._lock:
            candidates = [
                profile
                for profile in self._profiles.values()
                if profile.merged_into is None and profile.review_state != "rejected"
            ]
            if not candidates:
                return None
            profile, similarity = max(
                ((item, float(np.dot(embedding, item.embedding))) for item in candidates),
                key=lambda item: item[1],
            )
            if similarity < self.config.similarity_threshold:
                return None
            # A similarity hit is a proposal, not a sighting.  Only
            # ``confirm_match`` may advance durable encounter state.
            profile.last_match_state = "clip_proposal"
            return profile, similarity

    def relabel(
        self, profile_id: str, label: str, confidence: float, source: str, model_id: str,
        provenance: dict[str, object] | None = None,
        *, appearance_description: str | None = None,
        adjudication: dict[str, object] | None = None,
    ) -> ObjectProfile | None:
        normalized = " ".join(label.strip().split())
        if not normalized or len(normalized) > 64:
            return None
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return None
            if profile.label.casefold() != normalized.casefold():
                profile.label_history.append(
                    {
                        "label": profile.label,
                        "source": profile.label_source,
                        "confidence": profile.label_confidence,
                        "revised_at": datetime.now(timezone.utc).isoformat(),
                        "revised_by": model_id,
                    }
                )
                profile.label = normalized
            profile.label_source = source
            profile.label_confidence = confidence
            profile.label_provenance = {
                **dict(provenance or {}),
                "model_id": model_id,
                "accepted_at": datetime.now(timezone.utc).isoformat(),
            }
            profile.review_state = self._verified_state(source)
            profile.last_reviewed_at = datetime.now(timezone.utc)
            description = self._description(appearance_description)
            if description:
                profile.appearance_description = description
            if isinstance(adjudication, dict):
                profile.adjudication_history.append(dict(adjudication))
                profile.adjudication_history = profile.adjudication_history[-32:]
            self._save_metadata(profile)
            return profile

    def mark_review_failed(self, profile_id: str) -> None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile:
                profile.review_state = "failed"
                profile.last_reviewed_at = datetime.now(timezone.utc)
                self._save_metadata(profile)

    def reject_profile(
        self, profile_id: str, reason: str, adjudication: dict[str, object] | None = None
    ) -> ObjectProfile | None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return None
            profile.review_state = "rejected"
            profile.label_confidence = 0.0
            profile.last_match_state = "vlm_rejected"
            profile.last_reviewed_at = datetime.now(timezone.utc)
            profile.audit_state = "pixel_evidence_rejected"
            profile.audit_notes = " ".join(str(reason).split())[:400]
            if isinstance(adjudication, dict):
                profile.adjudication_history.append(dict(adjudication))
                profile.adjudication_history = profile.adjudication_history[-32:]
            self._save_metadata(profile)
            return profile

    def duplicate_candidates(
        self, minimum_similarity: float, limit: int = 8
    ) -> list[tuple[str, str, float]]:
        """Return embedding proposals; a VLA must decide physical identity."""

        with self._lock:
            profiles = [
                item
                for item in self._profiles.values()
                if item.merged_into is None and item.review_state != "rejected"
            ]
            proposals = [
                (left.profile_id, right.profile_id, float(np.dot(left.embedding, right.embedding)))
                for index, left in enumerate(profiles)
                for right in profiles[index + 1 :]
                if not self._pair_already_adjudicated(left, right)
            ]
        proposals = [item for item in proposals if item[2] >= minimum_similarity]
        proposals.sort(key=lambda item: item[2], reverse=True)
        return proposals[: max(0, int(limit))]

    def record_pair_adjudication(
        self,
        left_id: str,
        right_id: str,
        analysis: dict[str, object],
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            left = self._profiles.get(left_id)
            right = self._profiles.get(right_id)
            if left is None or right is None:
                return
            left_record = {
                **dict(analysis),
                "comparison_profile_id": right_id,
                "adjudicated_at": now,
            }
            right_record = {
                **dict(analysis),
                "comparison_profile_id": left_id,
                "adjudicated_at": now,
            }
            left.adjudication_history.append(left_record)
            right.adjudication_history.append(right_record)
            left.adjudication_history = left.adjudication_history[-32:]
            right.adjudication_history = right.adjudication_history[-32:]
            self._save_metadata(left)
            self._save_metadata(right)

    def merge_profiles(
        self,
        canonical_id: str,
        alias_id: str,
        similarity: float,
        adjudication: dict[str, object],
    ) -> ObjectProfile | None:
        """Coalesce two profiles after a VLA confirms the same physical item."""

        if canonical_id == alias_id:
            return None
        now = datetime.now(timezone.utc)
        with self._lock:
            canonical = self._profiles.get(canonical_id)
            alias = self._profiles.get(alias_id)
            if (
                canonical is None
                or alias is None
                or canonical.merged_into is not None
                or alias.merged_into is not None
            ):
                return None
            total = max(1, canonical.samples + alias.samples)
            canonical.embedding = self._normalized(
                canonical.embedding * canonical.samples
                + alias.embedding * alias.samples
            )
            canonical.samples = total
            canonical.first_seen = min(canonical.first_seen, alias.first_seen)
            canonical.last_seen = max(canonical.last_seen, alias.last_seen)
            canonical.confidence = max(canonical.confidence, alias.confidence)
            canonical.label_confidence = max(
                canonical.label_confidence, alias.label_confidence
            )
            if not canonical.appearance_description and alias.appearance_description:
                canonical.appearance_description = alias.appearance_description
            canonical.adjudication_history.extend(alias.adjudication_history)
            canonical.adjudication_history.append(dict(adjudication))
            canonical.adjudication_history = canonical.adjudication_history[-32:]
            canonical.last_match_state = "vlm_coalesced"
            canonical.last_reviewed_at = now
            alias.merged_into = canonical_id
            alias.review_state = "merged"
            alias.last_match_state = "vlm_coalesced_alias"
            alias.audit_state = "same_physical_object"
            alias.audit_notes = f"VLA-confirmed merge into {canonical_id} at {similarity:.3f}"
            alias.last_reviewed_at = now
            canonical_directory = self._directory / canonical.profile_id
            canonical_directory.mkdir(parents=True, exist_ok=True)
            np.save(canonical_directory / "embedding.npy", canonical.embedding)
            self._save_metadata(canonical)
            self._save_metadata(alias)
            return canonical

    def mark_audited(self, profile_id: str, audit_state: str, notes: str | None = None) -> ObjectProfile | None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            if profile is None:
                return None
            profile.audit_state = audit_state
            profile.audit_notes = notes
            profile.last_reviewed_at = datetime.now(timezone.utc)
            self._save_metadata(profile)
            return profile

    def profiles_due_for_review(self, stale_after_seconds: float) -> list[tuple[str, str, float]]:
        now = datetime.now(timezone.utc)
        with self._lock:
            due = []
            for profile in self._profiles.values():
                if (
                    not profile.thumbnail
                    or profile.merged_into is not None
                    or profile.review_state == "rejected"
                ):
                    continue
                if profile.label_source == "user" and profile.review_state == "user_corrected":
                    continue
                if profile.review_state in {"pending", "failed"}:
                    due.append(profile)
                    continue
                if profile.last_reviewed_at is None:
                    due.append(profile)
                    continue
                age = (now - profile.last_reviewed_at).total_seconds()
                if age >= stale_after_seconds:
                    due.append(profile)
            due.sort(key=lambda item: item.last_reviewed_at or datetime.min.replace(tzinfo=timezone.utc))
            return [(profile.profile_id, profile.label, profile.label_confidence) for profile in due]

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "id": profile.profile_id,
                    "label": profile.label,
                    "confidence": round(profile.confidence, 3),
                    "samples": profile.samples,
                    "first_seen": profile.first_seen.isoformat(),
                    "last_seen": profile.last_seen.isoformat(),
                    "thumbnail_url": f"/api/objects/{profile.profile_id}/mask.png",
                    "label_source": profile.label_source,
                    "label_confidence": round(profile.label_confidence, 3),
                    "review_state": profile.review_state,
                    "last_match_state": profile.last_match_state,
                    "label_history": list(profile.label_history),
                    "label_provenance": dict(profile.label_provenance),
                    "appearance_description": profile.appearance_description,
                    "adjudication_history": list(profile.adjudication_history),
                    "merged_into": profile.merged_into,
                }
                for profile in sorted(self._profiles.values(), key=lambda item: item.last_seen, reverse=True)
                if profile.merged_into is None and profile.review_state != "rejected"
            ]

    def summary_snapshot(self) -> list[dict[str, object]]:
        """Return the card-grid projection without full adjudication ledgers."""
        with self._lock:
            return [
                {
                    "id": profile.profile_id,
                    "label": profile.label,
                    "confidence": round(profile.confidence, 3),
                    "samples": profile.samples,
                    "first_seen": profile.first_seen.isoformat(),
                    "last_seen": profile.last_seen.isoformat(),
                    "thumbnail_url": f"/api/objects/{profile.profile_id}/mask.png",
                    "label_source": profile.label_source,
                    "label_confidence": round(profile.label_confidence, 3),
                    "review_state": profile.review_state,
                    "last_match_state": profile.last_match_state,
                    "label_history_count": len(profile.label_history),
                    "appearance_description": str(
                        profile.appearance_description or ""
                    )[:240],
                }
                for profile in sorted(
                    self._profiles.values(), key=lambda item: item.last_seen, reverse=True
                )
                if profile.merged_into is None and profile.review_state != "rejected"
            ]

    def thumbnail(self, profile_id: str) -> bytes | None:
        with self._lock:
            profile = self._profiles.get(profile_id)
            return profile.thumbnail if profile else None

    def migration_profiles(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "profile_id": profile.profile_id,
                    "label": profile.label,
                    "embedding": profile.embedding.copy(),
                    "confidence": profile.confidence,
                    "samples": profile.samples,
                    "first_seen": profile.first_seen,
                    "last_seen": profile.last_seen,
                    "thumbnail": bytes(profile.thumbnail) if profile.thumbnail else None,
                    "label_source": profile.label_source,
                    "label_confidence": profile.label_confidence,
                    "review_state": profile.review_state,
                    "label_history": list(profile.label_history),
                    "label_provenance": dict(profile.label_provenance),
                    "appearance_description": profile.appearance_description,
                    "adjudication_history": list(profile.adjudication_history),
                    "merged_into": profile.merged_into,
                }
                for profile in self._profiles.values()
            ]

    def profile_record(self, profile_id: str) -> dict[str, object] | None:
        return next(
            (profile for profile in self.migration_profiles() if profile["profile_id"] == profile_id),
            None,
        )

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            profile = self._profiles.pop(profile_id, None)
            if profile is None:
                return False
            shutil.rmtree(self._directory / profile_id, ignore_errors=True)
            return True

    def segmented_profile(self, profile_id: str) -> SegmentedObject | None:
        import cv2

        with self._lock:
            profile = self._profiles.get(profile_id)
            thumbnail = profile.thumbnail if profile else None
            confidence = profile.confidence if profile else 0.0
        if not thumbnail:
            return None
        decoded = cv2.imdecode(np.frombuffer(thumbnail, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 4:
            return None
        return SegmentedObject(decoded[:, :, :3], decoded[:, :, 3], confidence)

    def _load(self) -> None:
        for directory in sorted(self._directory.glob("object-*")):
            try:
                metadata = json.loads((directory / "profile.json").read_text(encoding="utf-8"))
                profile = ObjectProfile(
                    profile_id=str(metadata["id"]),
                    label=str(metadata["label"]),
                    embedding=self._normalized(np.load(directory / "embedding.npy")),
                    confidence=float(metadata["confidence"]),
                    samples=int(metadata["samples"]),
                    first_seen=datetime.fromisoformat(metadata["first_seen"]),
                    last_seen=datetime.fromisoformat(metadata["last_seen"]),
                    thumbnail=(directory / "mask.png").read_bytes(),
                    label_source=str(metadata.get("label_source", "legacy")),
                    label_confidence=float(metadata.get("label_confidence", metadata.get("confidence", 0.0))),
                    review_state=str(metadata.get("review_state", "pending")),
                    label_history=list(metadata.get("label_history", [])),
                    label_provenance=dict(metadata.get("label_provenance", {})),
                    last_reviewed_at=(
                        datetime.fromisoformat(metadata["last_reviewed_at"])
                        if metadata.get("last_reviewed_at")
                        else None
                    ),
                    audit_state=metadata.get("audit_state"),
                    audit_notes=metadata.get("audit_notes"),
                    appearance_description=metadata.get("appearance_description"),
                    adjudication_history=list(metadata.get("adjudication_history", [])),
                    merged_into=metadata.get("merged_into"),
                )
            except (KeyError, OSError, TypeError, ValueError):
                continue
            self._profiles[profile.profile_id] = profile

    def _save(self, profile: ObjectProfile, image: np.ndarray, mask: np.ndarray) -> None:
        import cv2

        directory = self._directory / profile.profile_id
        directory.mkdir(parents=True, exist_ok=True)
        alpha_image = np.dstack((image, mask))
        success, encoded = cv2.imencode(".png", alpha_image)
        if not success:
            raise RuntimeError("failed to encode segmented object crop")
        profile.thumbnail = encoded.tobytes()
        (directory / "mask.png").write_bytes(profile.thumbnail)
        np.save(directory / "embedding.npy", profile.embedding)
        self._save_metadata(profile)

    def _save_metadata(self, profile: ObjectProfile) -> None:
        directory = self._directory / profile.profile_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "profile.json").write_text(
            json.dumps(
                {
                    "id": profile.profile_id,
                    "label": profile.label,
                    "confidence": profile.confidence,
                    "samples": profile.samples,
                    "first_seen": profile.first_seen.isoformat(),
                    "last_seen": profile.last_seen.isoformat(),
                    "label_source": profile.label_source,
                    "label_confidence": profile.label_confidence,
                    "review_state": profile.review_state,
                    "label_history": profile.label_history,
                    "label_provenance": profile.label_provenance,
                    "last_reviewed_at": profile.last_reviewed_at.isoformat() if profile.last_reviewed_at else None,
                    "audit_state": profile.audit_state,
                    "audit_notes": profile.audit_notes,
                    "appearance_description": profile.appearance_description,
                    "adjudication_history": profile.adjudication_history,
                    "merged_into": profile.merged_into,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def _profile_numbers(self) -> list[int]:
        return [int(profile_id.removeprefix("object-")) for profile_id in self._profiles if profile_id.removeprefix("object-").isdigit()]

    @staticmethod
    def _verified_state(source: str) -> str:
        if source == "ornith-vlm":
            return "vlm_verified"
        if source == "user":
            return "user_corrected"
        return "pending"

    @staticmethod
    def _masked_image(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        import cv2

        return cv2.bitwise_and(image, image, mask=mask)

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        normalized = vector.astype(np.float32)
        norm = float(np.linalg.norm(normalized))
        return normalized / norm if norm > 0 else normalized

    @staticmethod
    def _description(value: str | None) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = " ".join(value.strip().split())
        return normalized[:600] if normalized else None

    @staticmethod
    def _pair_already_adjudicated(left: ObjectProfile, right: ObjectProfile) -> bool:
        return any(
            isinstance(item, dict)
            and item.get("comparison_profile_id") == right.profile_id
            for item in left.adjudication_history
        )
