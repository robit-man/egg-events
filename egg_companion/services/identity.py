from __future__ import annotations

import sqlite3
import threading
from hashlib import sha256
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from egg_companion.adapters.vision import FaceCrop, VisionEngine
from egg_companion.config import IdentityConfig
from egg_companion.memory.fusion import EvidenceFusion
from egg_companion.models import BoundingBox, Detection


@dataclass
class IdentityProfile:
    profile_id: str
    clip_embedding: np.ndarray
    face_embedding: np.ndarray | None
    samples: int
    sightings: int
    first_seen: datetime
    last_seen: datetime
    last_sample_at: datetime
    last_camera: str | None = None
    name: str | None = None
    confidence: float = 0.0
    thumbnail: bytes | None = None
    kind: str = "face"


@dataclass
class _PersonTrack:
    """Short-lived, camera-local continuity; deliberately not a person identity."""

    track_id: str
    camera_id: str
    bbox: BoundingBox
    first_seen: datetime
    last_seen: datetime
    observations: int = 1
    profile_id: str | None = None
    kind: str = "appearance-track"
    face_embeddings: list[np.ndarray] = field(default_factory=list)
    mask_polygon: tuple[tuple[float, float], ...] = ()
    frame_shape: tuple[int, int] | None = None
    last_crop_png: bytes | None = None
    last_vlm_comparison_at: datetime | None = None


class IdentityLibrary:
    """Persistent on-device identity profiles with face-first anonymous recall."""

    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self._directory = Path(config.storage_dir)
        self._lock = threading.RLock()
        self._profiles: dict[str, IdentityProfile] = {}
        # Aliases remain intact as source profiles.  This map only changes the
        # canonical identity used for recall and presentation.
        self._aliases: dict[str, tuple[str, float, str]] = {}
        self._database: sqlite3.Connection | None = None
        self._tracks: dict[str, list[_PersonTrack]] = {}
        self._next_track_number = 1
        self._last_match_components: dict[str, float] = {}
        self._last_match_outcome = "new"
        if config.enabled:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._database = sqlite3.connect(self._directory / "identities.sqlite3", check_same_thread=False)
            self._database.row_factory = sqlite3.Row
            self._create_schema()
            self._load()
            self._seed_face_galleries()

    def observe(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: tuple[Detection, ...],
        vision: VisionEngine,
    ) -> dict[int, dict[str, object]]:
        if not self.config.enabled:
            return {}
        matches: dict[int, dict[str, object]] = {}
        now = datetime.now(timezone.utc)
        used_tracks: set[str] = set()
        for index, detection in enumerate(detections):
            if detection.label != "person":
                continue
            track, new_track, association = self._associate_track(
                camera_id, detection, now, used_tracks
            )
            used_tracks.add(track.track_id)
            prior_crop = track.last_crop_png
            current_crop = self._segmented_person_png(frame, detection, vision)
            comparison = self._temporal_comparison_candidate(
                track,
                detection,
                association,
                prior_crop,
                current_crop,
                now,
                new_track,
            )
            self._advance_track(track, detection, current_crop, now, new_track)

            def attach(match: dict[str, object]) -> dict[str, object]:
                match["temporal_association"] = {
                    key: value
                    for key, value in association.items()
                    if key != "score"
                }
                if comparison is not None:
                    comparison["entity_id"] = str(match["id"])
                    match["_temporal_comparison"] = comparison
                return match

            faces = vision.face_crops(frame, detection)
            if not faces:
                matches[index] = attach(
                    self._track_match(track, detection.confidence, new_track)
                )
                continue
            face = max(faces, key=lambda item: item.image.shape[0] * item.image.shape[1])
            face_embedding = (
                self._normalized(face.face_embedding)
                if face.face_embedding is not None
                and face.confidence >= self.config.minimum_face_quality
                else None
            )
            if track.profile_id is not None:
                with self._lock:
                    profile = self._profiles.get(track.profile_id)
                if profile is not None:
                    self._update_profile(profile, camera_id, vision, face, now, similarity=1.0)
                    matches[index] = attach(
                        self._profile_match(
                            profile, False, 1.0, "track_continuity"
                        )
                    )
                    continue
                track.profile_id = None
            if face_embedding is None:
                track.kind = face.kind if face.kind != "face" else "low-quality-face-track"
                matches[index] = attach(
                    self._track_match(track, face.confidence, new_track)
                )
                continue

            # CLIP is retained as descriptive metadata for memory search, but it
            # is not consulted when deciding whether two observations are one
            # person. Face-specific evidence owns persistent identity.
            clip_embedding = self._normalized(vision.embed_image(face.image))
            profile, created, similarity = self._match_or_create(
                clip_embedding, face_embedding, allow_create=False
            )
            if profile is None:
                self._accumulate_face_candidate(track, face_embedding)
                track.kind = "face-candidate-track"
                if len(track.face_embeddings) < self.config.enrollment_min_face_observations:
                    matches[index] = attach(
                        self._track_match(track, face.confidence, new_track)
                    )
                    continue
                stable_embedding = self._normalized(np.sum(track.face_embeddings, axis=0))
                profile, created, similarity = self._match_or_create(
                    clip_embedding,
                    stable_embedding,
                    allow_create=True,
                    initial_samples=len(track.face_embeddings),
                )
            assert profile is not None
            track.profile_id = profile.profile_id
            with self._lock:
                if not created and new_track:
                    profile.sightings += 1
            self._update_profile(profile, camera_id, vision, face, now, similarity)
            matches[index] = attach(
                self._profile_match(
                    profile,
                    created,
                    similarity,
                    "new" if created else self._last_match_outcome,
                )
            )
        return matches

    def name_most_recent(self, name: str) -> IdentityProfile | None:
        normalized_name = " ".join(name.strip().split())
        if not normalized_name or len(normalized_name) > 64:
            return None
        with self._lock:
            unnamed = [
                profile for profile in self._profiles.values()
                if profile.name is None and profile.profile_id not in self._aliases
            ]
            if not unnamed:
                return None
            profile = max(unnamed, key=lambda item: item.last_seen)
            profile.name = normalized_name
            self._store(profile)
            return profile

    def name_profile(self, profile_id: str, name: str) -> IdentityProfile | None:
        normalized_name = " ".join(name.strip().split())
        if not normalized_name or len(normalized_name) > 64:
            return None
        with self._lock:
            profile = self._profiles.get(self._canonical_id(profile_id))
            if profile is None:
                return None
            profile.name = normalized_name
            self._store(profile)
            return profile

    def delete(self, profile_id: str) -> bool:
        with self._lock:
            profile = self._profiles.pop(profile_id, None)
            if profile is None:
                return False
            if self._database is not None:
                self._database.execute("DELETE FROM face_samples WHERE profile_id=?", (profile_id,))
                self._database.execute(
                    "DELETE FROM profile_aliases WHERE alias_id=? OR canonical_id=?",
                    (profile_id, profile_id),
                )
                self._database.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
                self._database.commit()
            self._aliases = {
                alias_id: mapping for alias_id, mapping in self._aliases.items()
                if alias_id != profile_id and mapping[0] != profile_id
            }
            return True

    def snapshot(self) -> list[dict[str, object]]:
        """Dashboard projection which never presents raw fragments as known people."""

        with self._lock:
            profiles = list(self._profiles.values())
            face_groups = self._face_groups()
            recurrent_faces = [
                group for group in face_groups.values()
                if any(profile.name is not None for profile in group)
                or sum(profile.samples for profile in group) >= 2
            ]
            pending_faces = [
                group[0] for group in face_groups.values()
                if all(profile.name is None for profile in group)
                and sum(profile.samples for profile in group) < 2
            ]
            appearance = [profile for profile in profiles if profile.face_embedding is None]
            rows = [self._snapshot_group(group) for group in recurrent_faces]
            pending = self._snapshot_stack(
                pending_faces,
                "stack:face-candidates",
                "Face candidates awaiting another sighting",
                "face-candidate-stack",
            )
            if pending is not None:
                rows.append(pending)
            by_camera: dict[str, list[IdentityProfile]] = {}
            for profile in appearance:
                by_camera.setdefault(profile.last_camera or "unknown camera", []).append(profile)
            for camera, camera_profiles in by_camera.items():
                stack = self._snapshot_stack(
                    camera_profiles,
                    f"stack:appearance:{camera}",
                    f"Unconfirmed observations · {camera}",
                    "appearance-observation-stack",
                )
                if stack is not None:
                    rows.append(stack)
            return sorted(rows, key=lambda item: str(item["last_seen"]), reverse=True)

    def summary(self) -> dict[str, object]:
        with self._lock:
            profiles = list(self._profiles.values())
            face_groups = self._face_groups()
            gallery = (
                self._database.execute(
                    """SELECT COUNT(*) AS samples, COUNT(DISTINCT profile_id) AS profiles
                    FROM face_samples WHERE validity != 'rejected'"""
                ).fetchone()
                if self._database is not None
                else None
            )
            return {
                "canonical_people": len(face_groups),
                "coalesced_aliases": len(self._aliases),
                "named_people": sum(
                    any(profile.name is not None for profile in group)
                    for group in face_groups.values()
                ),
                "recurrent_face_profiles": sum(
                    all(profile.name is None for profile in group)
                    and sum(profile.samples for profile in group) >= 2
                    for group in face_groups.values()
                ),
                "provisional_face_profiles": sum(
                    all(profile.name is None for profile in group)
                    and sum(profile.samples for profile in group) < 2
                    for group in face_groups.values()
                ),
                "legacy_appearance_fragments": sum(
                    profile.face_embedding is None for profile in profiles
                ),
                "active_observation_tracks": sum(len(tracks) for tracks in self._tracks.values()),
                "retained_face_samples": int(gallery["samples"]) if gallery else 0,
                "profiles_with_face_gallery": int(gallery["profiles"]) if gallery else 0,
                "quarantined_face_samples": int(
                    self._database.execute(
                        "SELECT COUNT(*) FROM face_samples WHERE validity='rejected'"
                    ).fetchone()[0]
                ) if self._database is not None else 0,
                "persistent_identity_authority": "face-specific multi-view templates",
                "appearance_policy": "temporal track only",
                "coalescing_policy": "two-of-three face-model template consensus with repeated/spatial co-observation veto",
            }

    def coalesce_profiles(
        self, conflicting_pairs: set[tuple[str, str]] | None = None
    ) -> list[dict[str, object]]:
        """Create conservative, reversible canonical aliases for legacy face fragments.

        Profiles are never deleted or rewritten.  An alias must directly clear
        the configured similarity threshold against a higher-evidence canonical
        profile, and profiles observed together are never coalesced.
        """

        if not self.config.enabled or not self.config.retroactive_coalescing_enabled:
            return []
        conflicts = {
            tuple(sorted((str(left), str(right))))
            for left, right in (conflicting_pairs or set()) if left != right
        }
        with self._lock:
            valid_profile_ids = self._valid_face_profile_ids()
            faces = [
                profile for profile in self._profiles.values()
                if profile.face_embedding is not None
                and profile.profile_id in valid_profile_ids
                and profile.profile_id not in self._aliases
            ]
            faces.sort(
                key=lambda profile: (
                    profile.name is not None,
                    profile.samples * max(1, profile.sightings),
                    profile.samples,
                    -int(profile.profile_id.removeprefix("person-") or 0)
                    if profile.profile_id.removeprefix("person-").isdigit() else 0,
                ),
                reverse=True,
            )
            assigned: set[str] = set(self._aliases)
            for canonical in faces:
                if canonical.profile_id in assigned:
                    continue
                for candidate in faces:
                    if candidate.profile_id == canonical.profile_id or candidate.profile_id in assigned:
                        continue
                    pair = tuple(sorted((canonical.profile_id, candidate.profile_id)))
                    if pair in conflicts or not self._names_compatible(canonical, candidate):
                        continue
                    similarity = float(np.dot(canonical.face_embedding, candidate.face_embedding))
                    if similarity < self.config.retroactive_merge_similarity:
                        continue
                    reason = "direct_face_similarity_no_coobservation_conflict"
                    self._aliases[candidate.profile_id] = (
                        canonical.profile_id, similarity, reason
                    )
                    assigned.add(candidate.profile_id)
                    self._store_alias(candidate.profile_id, canonical.profile_id, similarity, reason)
            return self.alias_mappings()

    def alias_mappings(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
                    "alias_id": alias_id,
                    "canonical_id": canonical_id,
                    "similarity": round(similarity, 6),
                    "reason": reason,
                }
                for alias_id, (canonical_id, similarity, reason) in sorted(self._aliases.items())
            ]

    def create_alias(
        self,
        alias_id: str,
        canonical_id: str,
        similarity: float,
        reason: str,
        conflicting_pairs: set[tuple[str, str]] | None = None,
    ) -> dict[str, object] | None:
        """Apply one reversible dream merge after rechecking identity constraints."""

        conflicts = {
            tuple(sorted((str(left), str(right))))
            for left, right in (conflicting_pairs or set()) if left != right
        }
        with self._lock:
            alias_id = self._canonical_id(alias_id)
            canonical_id = self._canonical_id(canonical_id)
            if alias_id == canonical_id or alias_id in self._aliases:
                return None
            alias = self._profiles.get(alias_id)
            canonical = self._profiles.get(canonical_id)
            if alias is None or canonical is None:
                return None
            pair = tuple(sorted((alias_id, canonical_id)))
            if pair in conflicts or not self._names_compatible(alias, canonical):
                return None
            bounded_similarity = max(0.0, min(1.0, float(similarity)))
            self._aliases[alias_id] = (canonical_id, bounded_similarity, reason)
            self._store_alias(alias_id, canonical_id, bounded_similarity, reason)
            return {
                "alias_id": alias_id,
                "canonical_id": canonical_id,
                "similarity": round(bounded_similarity, 6),
                "reason": reason,
            }

    def face_sample_snapshot(self) -> list[dict[str, object]]:
        """Return bounded retained face evidence for idle-time template aggregation."""

        if self._database is None:
            return []
        with self._lock:
            rows = self._database.execute(
                """SELECT sample_id, profile_id, captured_at, camera_id, quality,
                sface_embedding, image_jpeg, validity FROM face_samples
                WHERE validity != 'rejected' ORDER BY captured_at"""
            ).fetchall()
            return [
                {
                    "sample_id": str(row["sample_id"]),
                    "profile_id": str(row["profile_id"]),
                    "captured_at": str(row["captured_at"]),
                    "camera_id": row["camera_id"],
                    "quality": float(row["quality"]),
                    "sface_embedding": self._vector(row["sface_embedding"]),
                    "image_jpeg": bytes(row["image_jpeg"]),
                    "validity": str(row["validity"]),
                }
                for row in rows
            ]

    def unvalidated_face_samples(self) -> list[dict[str, object]]:
        """Return retained legacy samples that still need visual validation."""

        if self._database is None:
            return []
        with self._lock:
            return [
                {
                    "sample_id": str(row["sample_id"]),
                    "profile_id": str(row["profile_id"]),
                    "image_jpeg": bytes(row["image_jpeg"]),
                }
                for row in self._database.execute(
                    """SELECT sample_id, profile_id, image_jpeg FROM face_samples
                    WHERE validity='pending' ORDER BY captured_at"""
                )
            ]

    def apply_face_validation(
        self, results: dict[str, bool], model_id: str
    ) -> dict[str, int]:
        """Persist reversible sample-level acceptance without deleting evidence."""

        if self._database is None or not results:
            return {"accepted": 0, "rejected": 0}
        accepted = sum(bool(value) for value in results.values())
        rejected = len(results) - accepted
        validated_at = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._database.executemany(
                """UPDATE face_samples SET validity=?, validation_model=?, validated_at=?
                WHERE sample_id=?""",
                [
                    (
                        "accepted" if is_valid else "rejected",
                        model_id,
                        validated_at,
                        sample_id,
                    )
                    for sample_id, is_valid in results.items()
                ],
            )
            self._database.commit()
        return {"accepted": accepted, "rejected": rejected}

    def identity_timeline_source(self, profile_id: str) -> dict[str, object] | None:
        """Canonical identity metadata and retained crops across every source alias."""

        with self._lock:
            canonical_id = self._canonical_id(profile_id)
            canonical = self._profiles.get(canonical_id)
            if canonical is None or canonical.face_embedding is None:
                return None
            valid_profile_ids = self._valid_face_profile_ids()
            members = [
                profile for profile in self._profiles.values()
                if profile.face_embedding is not None
                and profile.profile_id in valid_profile_ids
                and self._canonical_id(profile.profile_id) == canonical_id
            ]
            if not members:
                return None
            member_ids = sorted(profile.profile_id for profile in members)
            samples: list[dict[str, object]] = []
            if self._database is not None and member_ids:
                placeholders = ",".join("?" for _ in member_ids)
                rows = self._database.execute(
                    f"""SELECT sample_id, profile_id, captured_at, camera_id, quality
                    FROM face_samples WHERE profile_id IN ({placeholders})
                    AND validity != 'rejected'
                    ORDER BY captured_at DESC""",
                    member_ids,
                ).fetchall()
                samples = [
                    {
                        "sample_id": str(row["sample_id"]),
                        "source_profile_id": str(row["profile_id"]),
                        "captured_at": str(row["captured_at"]),
                        "camera_id": row["camera_id"],
                        "quality": round(float(row["quality"]), 4),
                        "artifact_url": (
                            f"/api/identities/{canonical_id}/samples/{row['sample_id']}.jpg"
                        ),
                    }
                    for row in rows
                ]
            representative = max(members, key=lambda item: item.last_seen)
            name = next((profile.name for profile in members if profile.name), None)
            return {
                "id": canonical_id,
                "label": name or canonical_id,
                "alias_ids": [item for item in member_ids if item != canonical_id],
                "source_profile_ids": member_ids,
                "first_seen": min(profile.first_seen for profile in members).isoformat(),
                "last_seen": representative.last_seen.isoformat(),
                "last_camera": representative.last_camera,
                "samples": sum(profile.samples for profile in members),
                "sightings": sum(profile.sightings for profile in members),
                "retained_face_samples": samples,
                "thumbnail_url": f"/api/identities/{representative.profile_id}/face.jpg",
            }

    def face_sample(self, profile_id: str, sample_id: str) -> bytes | None:
        if self._database is None:
            return None
        with self._lock:
            canonical_id = self._canonical_id(profile_id)
            row = self._database.execute(
                "SELECT profile_id, image_jpeg FROM face_samples WHERE sample_id=?",
                (sample_id,),
            ).fetchone()
            if row is None or self._canonical_id(str(row["profile_id"])) != canonical_id:
                return None
            return bytes(row["image_jpeg"])

    def thumbnail(self, profile_id: str) -> bytes | None:
        with self._lock:
            if self._database is not None:
                row = self._database.execute(
                    """SELECT image_jpeg FROM face_samples
                    WHERE profile_id=? AND validity != 'rejected'
                    ORDER BY quality DESC, captured_at DESC LIMIT 1""",
                    (profile_id,),
                ).fetchone()
                if row is not None:
                    return bytes(row["image_jpeg"])
            profile = self._profiles.get(profile_id)
            return (
                profile.thumbnail
                if profile is not None and profile_id in self._valid_face_profile_ids()
                else None
            )

    def migration_profiles(self) -> list[dict[str, object]]:
        with self._lock:
            valid_sample_counts: dict[str, int] = {}
            if self._database is not None:
                valid_sample_counts = {
                    str(row["profile_id"]): int(row["samples"])
                    for row in self._database.execute(
                        """SELECT profile_id, COUNT(*) AS samples FROM face_samples
                        WHERE validity != 'rejected' GROUP BY profile_id"""
                    )
                }
            return [
                {
                    "profile_id": profile.profile_id,
                    "name": profile.name,
                    "kind": profile.kind,
                    "confidence": profile.confidence,
                    "samples": profile.samples,
                    "sightings": profile.sightings,
                    "first_seen": profile.first_seen,
                    "last_seen": profile.last_seen,
                    "last_camera": profile.last_camera,
                    "clip_embedding": profile.clip_embedding.copy(),
                    "face_embedding": profile.face_embedding.copy() if profile.face_embedding is not None else None,
                    "thumbnail": bytes(profile.thumbnail) if profile.thumbnail else None,
                    "valid_face_samples": valid_sample_counts.get(profile.profile_id, 0),
                }
                for profile in self._profiles.values()
            ]

    def profile_record(self, profile_id: str) -> dict[str, object] | None:
        profile_id = self._canonical_id(profile_id)
        return next(
            (profile for profile in self.migration_profiles() if profile["profile_id"] == profile_id),
            None,
        )

    def _match_or_create(
        self,
        clip_embedding: np.ndarray,
        face_embedding: np.ndarray,
        *,
        allow_create: bool,
        initial_samples: int = 1,
    ) -> tuple[IdentityProfile | None, bool, float]:
        with self._lock:
            valid_profile_ids = self._valid_face_profile_ids()
            candidates = [
                profile for profile in self._profiles.values()
                if profile.face_embedding is not None
                and profile.profile_id in valid_profile_ids
            ]
            best_score = 0.0
            if candidates:
                # Rank distinct canonical people rather than raw source rows.
                # Otherwise two old fragments of one face become each other's
                # runner-up and defeat the match-margin safety check.
                grouped: dict[str, tuple[IdentityProfile, float]] = {}
                for item in candidates:
                    canonical_id = self._canonical_id(item.profile_id)
                    score = float(np.dot(face_embedding, item.face_embedding))
                    previous = grouped.get(canonical_id)
                    if previous is None or score > previous[1]:
                        grouped[canonical_id] = (item, score)
                ranked = sorted(grouped.items(), key=lambda item: item[1][1], reverse=True)
                canonical_id, (matched_source, score) = ranked[0]
                profile = self._profiles[canonical_id]
                best_score = score
                runner_up = ranked[1][1][1] if len(ranked) > 1 else -1.0
                separated = len(ranked) == 1 or score - runner_up >= self.config.face_match_margin
                if score >= self.config.face_similarity_threshold and separated:
                    clip_score = float(np.dot(clip_embedding, matched_source.clip_embedding))
                    fusion = EvidenceFusion.person(score, clip_score)
                    self._last_match_components = {
                        **fusion.components,
                        "runner_up_similarity": max(0.0, runner_up),
                        "match_margin": max(0.0, score - runner_up),
                    }
                    if matched_source.profile_id != canonical_id:
                        self._last_match_components["matched_alias"] = matched_source.profile_id
                        self._last_match_outcome = "coalesced_recall"
                    else:
                        self._last_match_outcome = "recalled"
                    return profile, False, score
            if not allow_create:
                self._last_match_components = {
                    "face_similarity": max(0.0, best_score),
                    "enrollment_progress": 0.0,
                }
                self._last_match_outcome = "face_candidate"
                return None, False, best_score
            now = datetime.now(timezone.utc)
            profile_id = f"person-{max(self._profile_numbers(), default=0) + 1:03d}"
            profile = IdentityProfile(
                profile_id=profile_id,
                clip_embedding=self._normalized(clip_embedding),
                face_embedding=self._normalized(face_embedding),
                samples=max(1, initial_samples),
                sightings=1,
                first_seen=now,
                last_seen=now,
                last_sample_at=now,
                kind="face",
            )
            self._profiles[profile_id] = profile
            fusion = EvidenceFusion.person(
                1.0,
                1.0,
            )
            self._last_match_components = fusion.components
            self._last_match_outcome = "new"
            return profile, True, 1.0

    def _accumulate_face_candidate(
        self, track: _PersonTrack, face_embedding: np.ndarray
    ) -> None:
        if track.face_embeddings:
            centroid = self._normalized(np.sum(track.face_embeddings, axis=0))
            if float(np.dot(centroid, face_embedding)) < self.config.enrollment_face_consistency:
                track.face_embeddings.clear()
        track.face_embeddings.append(face_embedding)
        keep = self.config.enrollment_min_face_observations
        if len(track.face_embeddings) > keep:
            track.face_embeddings = track.face_embeddings[-keep:]

    def _associate_track(
        self,
        camera_id: str,
        detection: Detection,
        now: datetime,
        used_tracks: set[str],
    ) -> tuple[_PersonTrack, bool, dict[str, object]]:
        with self._lock:
            tracks = [
                track for track in self._tracks.get(camera_id, [])
                if (now - track.last_seen).total_seconds() <= self.config.track_ttl_seconds
            ]
            self._tracks[camera_id] = tracks
            ranked = sorted(
                (
                    (self._track_affinity(track, detection, now), track)
                    for track in tracks if track.track_id not in used_tracks
                ),
                key=lambda item: float(item[0]["score"]),
                reverse=True,
            )
            if ranked and float(ranked[0][0]["score"]) > 0:
                association = ranked[0][0]
                track = ranked[0][1]
                return track, False, association
            polygon, frame_shape = self._detection_mask_geometry(detection)
            track = _PersonTrack(
                f"track-{self._next_track_number:06d}",
                camera_id,
                detection.bbox,
                now,
                now,
                mask_polygon=polygon,
                frame_shape=frame_shape,
            )
            self._next_track_number += 1
            tracks.append(track)
            return track, True, {
                "basis": "new_track",
                "elapsed_seconds": 0.0,
                "bbox_iou": 0.0,
                "mask_iou": 0.0,
                "mask_containment": 0.0,
                "center_displacement": 0.0,
                "score": 0.0,
            }

    def _track_affinity(
        self, track: _PersonTrack, detection: Detection, now: datetime
    ) -> dict[str, object]:
        prior = track.bbox
        current = detection.bbox
        intersection_width = max(0.0, min(prior.x2, current.x2) - max(prior.x1, current.x1))
        intersection_height = max(0.0, min(prior.y2, current.y2) - max(prior.y1, current.y1))
        intersection = intersection_width * intersection_height
        union = prior.area + current.area - intersection
        bbox_iou = intersection / union if union > 0 else 0.0
        prior_width = max(1.0, prior.x2 - prior.x1)
        prior_height = max(1.0, prior.y2 - prior.y1)
        prior_x = (prior.x1 + prior.x2) / 2
        prior_y = (prior.y1 + prior.y2) / 2
        current_x = (current.x1 + current.x2) / 2
        current_y = (current.y1 + current.y2) / 2
        normalized_distance = (
            ((current_x - prior_x) / prior_width) ** 2
            + ((current_y - prior_y) / prior_height) ** 2
        ) ** 0.5
        elapsed = max(0.0, (now - track.last_seen).total_seconds())
        polygon, frame_shape = self._detection_mask_geometry(detection)
        mask_iou, mask_containment = (0.0, 0.0)
        if elapsed <= self.config.track_mask_max_gap_seconds:
            mask_iou, mask_containment = self._mask_overlap(
                track.mask_polygon,
                polygon,
                track.frame_shape,
                frame_shape,
            )
        if (
            mask_iou >= self.config.track_mask_iou_threshold
            or mask_containment >= self.config.track_mask_containment_threshold
        ):
            basis = "mask_overlap"
            score = 3.0 + max(mask_iou, mask_containment)
        elif bbox_iou >= self.config.track_iou_threshold:
            basis = "bbox_iou"
            score = 2.0 + bbox_iou
        elif normalized_distance <= self.config.track_center_distance:
            basis = "center_continuity"
            score = 1.0 - normalized_distance
        else:
            basis = "none"
            score = 0.0
        return {
            "basis": basis,
            "elapsed_seconds": round(elapsed, 4),
            "bbox_iou": round(bbox_iou, 6),
            "mask_iou": round(mask_iou, 6),
            "mask_containment": round(mask_containment, 6),
            "center_displacement": round(normalized_distance, 6),
            "prior_bbox": self._bbox_dict(prior),
            "current_bbox": self._bbox_dict(current),
            "score": score,
        }

    def _advance_track(
        self,
        track: _PersonTrack,
        detection: Detection,
        current_crop: bytes | None,
        now: datetime,
        new_track: bool,
    ) -> None:
        if not new_track:
            track.last_seen = now
            track.observations += 1
        track.bbox = detection.bbox
        track.mask_polygon, track.frame_shape = self._detection_mask_geometry(detection)
        if current_crop is not None:
            track.last_crop_png = current_crop

    def _temporal_comparison_candidate(
        self,
        track: _PersonTrack,
        detection: Detection,
        association: dict[str, object],
        prior_crop: bytes | None,
        current_crop: bytes | None,
        now: datetime,
        new_track: bool,
    ) -> dict[str, object] | None:
        if (
            new_track
            or not self.config.temporal_vlm_comparison_enabled
            or association.get("basis") != "mask_overlap"
            or prior_crop is None
            or current_crop is None
        ):
            return None
        if (
            track.last_vlm_comparison_at is not None
            and (now - track.last_vlm_comparison_at).total_seconds()
            < self.config.temporal_vlm_cooldown_seconds
        ):
            return None
        track.last_vlm_comparison_at = now
        prior_center = (
            (track.bbox.x1 + track.bbox.x2) / 2,
            (track.bbox.y1 + track.bbox.y2) / 2,
        )
        current_center = (
            (detection.bbox.x1 + detection.bbox.x2) / 2,
            (detection.bbox.y1 + detection.bbox.y2) / 2,
        )
        candidate_id = "temporal-person-" + sha256(
            prior_crop + current_crop + track.track_id.encode()
        ).hexdigest()[:20]
        return {
            "candidate_id": candidate_id,
            "track_id": track.track_id,
            "prior_entity_id": track.profile_id or track.track_id,
            "camera_id": track.camera_id,
            "captured_at": now.isoformat(),
            "prior_png": prior_crop,
            "current_png": current_crop,
            "geometry": {
                **{key: value for key, value in association.items() if key != "score"},
                "centroid_dx_pixels": round(current_center[0] - prior_center[0], 2),
                "centroid_dy_pixels": round(current_center[1] - prior_center[1], 2),
                "simultaneous_distinct_mask_conflict": False,
            },
        }

    @staticmethod
    def _segmented_person_png(
        frame: np.ndarray, detection: Detection, vision: VisionEngine
    ) -> bytes | None:
        try:
            segmented = vision.segment_detection(frame, detection)
            return (
                vision.encode_segmented_object(segmented, 384)
                if segmented is not None else None
            )
        except (AttributeError, RuntimeError, ValueError):
            return None

    @staticmethod
    def _detection_mask_geometry(
        detection: Detection,
    ) -> tuple[tuple[tuple[float, float], ...], tuple[int, int] | None]:
        polygon = detection.attributes.get("mask_polygon")
        shape = detection.attributes.get("frame_shape")
        points: tuple[tuple[float, float], ...] = ()
        if isinstance(polygon, list) and len(polygon) >= 3:
            try:
                points = tuple((float(point[0]), float(point[1])) for point in polygon)
            except (IndexError, TypeError, ValueError):
                points = ()
        frame_shape = None
        if isinstance(shape, (list, tuple)) and len(shape) >= 2:
            try:
                frame_shape = (int(shape[0]), int(shape[1]))
            except (TypeError, ValueError):
                frame_shape = None
        return points, frame_shape

    @staticmethod
    def _mask_overlap(
        prior: tuple[tuple[float, float], ...],
        current: tuple[tuple[float, float], ...],
        prior_shape: tuple[int, int] | None,
        current_shape: tuple[int, int] | None,
    ) -> tuple[float, float]:
        if len(prior) < 3 or len(current) < 3 or prior_shape != current_shape:
            return 0.0, 0.0
        import cv2

        prior_points = np.asarray(prior, dtype=np.float32)
        current_points = np.asarray(current, dtype=np.float32)
        all_points = np.vstack((prior_points, current_points))
        minimum = np.floor(all_points.min(axis=0))
        maximum = np.ceil(all_points.max(axis=0))
        width = max(1, int(maximum[0] - minimum[0] + 3))
        height = max(1, int(maximum[1] - minimum[1] + 3))
        scale = min(1.0, 512.0 / max(width, height))
        canvas_width = max(2, int(np.ceil(width * scale)))
        canvas_height = max(2, int(np.ceil(height * scale)))

        def shifted(points: np.ndarray) -> np.ndarray:
            return np.round((points - minimum + 1) * scale).astype(np.int32)

        prior_mask = np.zeros((canvas_height, canvas_width), dtype=np.uint8)
        current_mask = np.zeros_like(prior_mask)
        cv2.fillPoly(prior_mask, [shifted(prior_points)], 1)
        cv2.fillPoly(current_mask, [shifted(current_points)], 1)
        prior_area = int(np.count_nonzero(prior_mask))
        current_area = int(np.count_nonzero(current_mask))
        intersection = int(np.count_nonzero(prior_mask & current_mask))
        union = prior_area + current_area - intersection
        return (
            intersection / union if union else 0.0,
            intersection / min(prior_area, current_area)
            if min(prior_area, current_area) else 0.0,
        )

    @staticmethod
    def _bbox_dict(bbox: BoundingBox) -> dict[str, float]:
        return {
            "x1": round(bbox.x1, 2),
            "y1": round(bbox.y1, 2),
            "x2": round(bbox.x2, 2),
            "y2": round(bbox.y2, 2),
        }

    def _track_match(
        self, track: _PersonTrack, confidence: float, new_track: bool
    ) -> dict[str, object]:
        continuity = min(1.0, track.observations / 3)
        fusion = EvidenceFusion.person(None, None, continuity)
        return {
            "id": track.track_id,
            "label": "Unconfirmed person",
            "confidence": round(float(confidence) * continuity, 3),
            "needs_name": False,
            "new": new_track,
            "recalled": not new_track,
            "kind": track.kind,
            "sightings": track.observations,
            "resolver_outcome": "appearance_track",
            "confidence_components": fusion.components,
            "persistent": False,
            "enrollment_observations": len(track.face_embeddings),
            "enrollment_required": self.config.enrollment_min_face_observations,
        }

    def _profile_match(
        self, profile: IdentityProfile, created: bool, similarity: float, outcome: str
    ) -> dict[str, object]:
        return {
            "id": profile.profile_id,
            "label": profile.name or profile.profile_id,
            "confidence": round(profile.confidence, 3),
            "needs_name": profile.name is None,
            "new": created,
            "recalled": not created,
            "kind": profile.kind,
            "sightings": profile.sightings,
            "resolver_outcome": outcome,
            "confidence_components": dict(self._last_match_components),
            "persistent": True,
        }

    def _update_profile(
        self,
        profile: IdentityProfile,
        camera_id: str,
        vision: VisionEngine,
        face: FaceCrop,
        now: datetime,
        similarity: float,
    ) -> None:
        with self._lock:
            profile.last_seen = now
            profile.last_camera = camera_id
            profile.confidence = face.confidence if similarity >= 1 else min(face.confidence, similarity)
            if profile.thumbnail is None:
                self._save(profile, face.image, face.face_embedding, face.confidence, camera_id, now)
            elif (now - profile.last_sample_at).total_seconds() >= self.config.sample_interval_seconds:
                profile.clip_embedding = self._normalized(
                    (profile.clip_embedding * profile.samples) + vision.embed_image(face.image)
                )
                if face.face_embedding is not None and profile.face_embedding is not None:
                    profile.face_embedding = self._normalized(
                        (profile.face_embedding * profile.samples) + face.face_embedding
                    )
                profile.samples += 1
                profile.last_sample_at = now
                self._save(profile, face.image, face.face_embedding, face.confidence, camera_id, now)
            else:
                self._store(profile)

    @staticmethod
    def _snapshot_profile(profile: IdentityProfile, status: str) -> dict[str, object]:
        return {
            "id": profile.profile_id,
            "label": profile.name or profile.profile_id,
            "samples": profile.samples,
            "sightings": profile.sightings,
            "confidence": round(profile.confidence, 3),
            "first_seen": profile.first_seen.isoformat(),
            "last_seen": profile.last_seen.isoformat(),
            "last_camera": profile.last_camera,
            "thumbnail_url": f"/api/identities/{profile.profile_id}/face.jpg",
            "kind": profile.kind,
            "status": status,
            "stack_count": 1,
        }

    def _snapshot_group(self, profiles: list[IdentityProfile]) -> dict[str, object]:
        canonical_id = self._canonical_id(profiles[0].profile_id)
        canonical = self._profiles[canonical_id]
        representative = max(profiles, key=lambda profile: profile.last_seen)
        named = next((profile.name for profile in profiles if profile.name), None)
        return {
            "id": canonical_id,
            "label": named or canonical_id,
            "samples": sum(profile.samples for profile in profiles),
            "sightings": sum(profile.sightings for profile in profiles),
            "confidence": round(max(profile.confidence for profile in profiles), 3),
            "first_seen": min(profile.first_seen for profile in profiles).isoformat(),
            "last_seen": representative.last_seen.isoformat(),
            "last_camera": representative.last_camera,
            "thumbnail_url": f"/api/identities/{representative.profile_id}/face.jpg",
            "kind": canonical.kind,
            "status": "coalesced" if len(profiles) > 1 else ("named" if named else "recurrent"),
            "stack_count": len(profiles),
            "alias_ids": sorted(
                profile.profile_id for profile in profiles if profile.profile_id != canonical_id
            ),
            "representative_id": representative.profile_id,
        }

    def _face_groups(self) -> dict[str, list[IdentityProfile]]:
        groups: dict[str, list[IdentityProfile]] = {}
        valid_profile_ids = self._valid_face_profile_ids()
        for profile in self._profiles.values():
            if (
                profile.face_embedding is None
                or profile.profile_id not in valid_profile_ids
            ):
                continue
            groups.setdefault(self._canonical_id(profile.profile_id), []).append(profile)
        return groups

    def _valid_face_profile_ids(self) -> set[str]:
        if self._database is None:
            return {
                profile.profile_id for profile in self._profiles.values()
                if profile.face_embedding is not None
            }
        return {
            str(row["profile_id"])
            for row in self._database.execute(
                """SELECT DISTINCT profile_id FROM face_samples
                WHERE validity != 'rejected'"""
            )
        }

    def _canonical_id(self, profile_id: str) -> str:
        seen: set[str] = set()
        while profile_id not in seen:
            seen.add(profile_id)
            mapping = self._aliases.get(profile_id)
            if mapping is None or mapping[0] not in self._profiles:
                break
            profile_id = mapping[0]
        return profile_id

    @staticmethod
    def _names_compatible(left: IdentityProfile, right: IdentityProfile) -> bool:
        return not (
            left.name and right.name and left.name.strip().casefold() != right.name.strip().casefold()
        )

    @classmethod
    def _snapshot_stack(
        cls,
        profiles: list[IdentityProfile],
        stack_id: str,
        label: str,
        kind: str,
    ) -> dict[str, object] | None:
        if not profiles:
            return None
        representative = max(profiles, key=lambda profile: profile.last_seen)
        return {
            "id": stack_id,
            "label": label,
            "samples": sum(profile.samples for profile in profiles),
            "sightings": sum(profile.sightings for profile in profiles),
            "confidence": round(max(profile.confidence for profile in profiles), 3),
            "first_seen": min(profile.first_seen for profile in profiles).isoformat(),
            "last_seen": representative.last_seen.isoformat(),
            "last_camera": representative.last_camera,
            "thumbnail_url": f"/api/identities/{representative.profile_id}/face.jpg",
            "kind": kind,
            "status": "observation-stack",
            "stack_count": len(profiles),
            "representative_id": representative.profile_id,
        }

    def _create_schema(self) -> None:
        if self._database is None:
            return
        self._database.executescript(
            """
            CREATE TABLE IF NOT EXISTS profiles (
                profile_id TEXT PRIMARY KEY,
                name TEXT,
                kind TEXT NOT NULL,
                confidence REAL NOT NULL,
                samples INTEGER NOT NULL,
                sightings INTEGER NOT NULL,
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                last_sample_at TEXT NOT NULL,
                last_camera TEXT,
                clip_embedding BLOB NOT NULL,
                face_embedding BLOB,
                thumbnail BLOB
            );
            CREATE INDEX IF NOT EXISTS profiles_last_seen ON profiles(last_seen DESC);
            CREATE TABLE IF NOT EXISTS profile_aliases (
                alias_id TEXT PRIMARY KEY REFERENCES profiles(profile_id) ON DELETE CASCADE,
                canonical_id TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                similarity REAL NOT NULL,
                reason TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS profile_aliases_canonical ON profile_aliases(canonical_id);
            CREATE TABLE IF NOT EXISTS face_samples (
                sample_id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL REFERENCES profiles(profile_id) ON DELETE CASCADE,
                captured_at TEXT NOT NULL,
                camera_id TEXT,
                quality REAL NOT NULL,
                sface_embedding BLOB NOT NULL,
                image_jpeg BLOB NOT NULL,
                validity TEXT NOT NULL DEFAULT 'accepted',
                validation_model TEXT,
                validated_at TEXT
            );
            CREATE INDEX IF NOT EXISTS face_samples_profile ON face_samples(profile_id, captured_at DESC);
            """
        )
        columns = {
            str(row[1]) for row in self._database.execute("PRAGMA table_info(face_samples)")
        }
        if "validity" not in columns:
            self._database.execute(
                "ALTER TABLE face_samples ADD COLUMN validity TEXT NOT NULL DEFAULT 'pending'"
            )
        if "validation_model" not in columns:
            self._database.execute(
                "ALTER TABLE face_samples ADD COLUMN validation_model TEXT"
            )
        if "validated_at" not in columns:
            self._database.execute(
                "ALTER TABLE face_samples ADD COLUMN validated_at TEXT"
            )
        self._database.commit()

    def _load(self) -> None:
        if self._database is None:
            return
        for row in self._database.execute("SELECT * FROM profiles"):
            face_bytes = row["face_embedding"]
            face_embedding = self._vector(face_bytes) if face_bytes is not None else None
            profile = IdentityProfile(
                profile_id=row["profile_id"],
                clip_embedding=self._vector(row["clip_embedding"]),
                face_embedding=face_embedding,
                samples=int(row["samples"]),
                sightings=int(row["sightings"]),
                first_seen=datetime.fromisoformat(row["first_seen"]),
                last_seen=datetime.fromisoformat(row["last_seen"]),
                last_sample_at=datetime.fromisoformat(row["last_sample_at"]),
                last_camera=row["last_camera"],
                name=row["name"],
                confidence=float(row["confidence"]),
                thumbnail=row["thumbnail"],
                kind=row["kind"],
            )
            self._profiles[profile.profile_id] = profile
        for row in self._database.execute("SELECT * FROM profile_aliases"):
            if row["alias_id"] in self._profiles and row["canonical_id"] in self._profiles:
                self._aliases[str(row["alias_id"])] = (
                    str(row["canonical_id"]), float(row["similarity"]), str(row["reason"])
                )

    def _store_alias(
        self, alias_id: str, canonical_id: str, similarity: float, reason: str
    ) -> None:
        if self._database is None:
            return
        self._database.execute(
            """INSERT INTO profile_aliases
            (alias_id, canonical_id, similarity, reason, created_at) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(alias_id) DO UPDATE SET canonical_id=excluded.canonical_id,
            similarity=excluded.similarity, reason=excluded.reason""",
            (alias_id, canonical_id, similarity, reason, datetime.now(timezone.utc).isoformat()),
        )
        self._database.commit()

    def _save(
        self,
        profile: IdentityProfile,
        crop: np.ndarray,
        face_embedding: np.ndarray | None = None,
        quality: float | None = None,
        camera_id: str | None = None,
        captured_at: datetime | None = None,
    ) -> None:
        import cv2

        success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("failed to encode identity crop")
        profile.thumbnail = encoded.tobytes()
        self._store(profile)
        retained_embedding = face_embedding if face_embedding is not None else profile.face_embedding
        if retained_embedding is not None:
            self._store_face_sample(
                profile.profile_id,
                profile.thumbnail,
                retained_embedding,
                quality if quality is not None else profile.confidence,
                camera_id or profile.last_camera,
                captured_at or profile.last_sample_at,
            )

    def _seed_face_galleries(self) -> None:
        if self._database is None:
            return
        for profile in self._profiles.values():
            if profile.face_embedding is None or profile.thumbnail is None:
                continue
            existing = self._database.execute(
                "SELECT 1 FROM face_samples WHERE profile_id=? LIMIT 1", (profile.profile_id,)
            ).fetchone()
            if existing is None:
                self._store_face_sample(
                    profile.profile_id,
                    profile.thumbnail,
                    profile.face_embedding,
                    profile.confidence,
                    profile.last_camera,
                    profile.last_sample_at,
                )

    def _store_face_sample(
        self,
        profile_id: str,
        image_jpeg: bytes,
        sface_embedding: np.ndarray,
        quality: float,
        camera_id: str | None,
        captured_at: datetime,
    ) -> None:
        if self._database is None:
            return
        normalized = self._normalized(sface_embedding)
        sample_id = sha256(profile_id.encode("utf-8") + b"\0" + image_jpeg).hexdigest()
        if self._database.execute(
            "SELECT 1 FROM face_samples WHERE sample_id=?", (sample_id,)
        ).fetchone() is not None:
            return
        existing = self._database.execute(
            "SELECT quality, sface_embedding FROM face_samples WHERE profile_id=?",
            (profile_id,),
        ).fetchall()
        if existing:
            similarities = [
                float(np.dot(normalized, self._vector(row["sface_embedding"]))) for row in existing
            ]
            if (
                min(similarities) >= self.config.gallery_diversity_similarity
                and max(float(row["quality"]) for row in existing) >= float(quality)
            ):
                return
        self._database.execute(
            """INSERT INTO face_samples
            (sample_id, profile_id, captured_at, camera_id, quality, sface_embedding, image_jpeg,
            validity, validation_model, validated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'accepted', 'live-yunet-clip-consensus', ?)""",
            (
                sample_id,
                profile_id,
                captured_at.isoformat(),
                camera_id,
                max(0.0, min(1.0, float(quality))),
                normalized.astype(np.float32).tobytes(),
                image_jpeg,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        overflow = self._database.execute(
            """SELECT sample_id FROM face_samples WHERE profile_id=?
            ORDER BY quality DESC, captured_at DESC LIMIT -1 OFFSET ?""",
            (profile_id, self.config.gallery_max_samples),
        ).fetchall()
        if overflow:
            self._database.executemany(
                "DELETE FROM face_samples WHERE sample_id=?",
                [(str(row["sample_id"]),) for row in overflow],
            )
        self._database.commit()

    def _store(self, profile: IdentityProfile) -> None:
        if self._database is None:
            return
        self._database.execute(
            """
            INSERT INTO profiles (
                profile_id, name, kind, confidence, samples, sightings, first_seen, last_seen, last_sample_at,
                last_camera, clip_embedding, face_embedding, thumbnail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(profile_id) DO UPDATE SET
                name=excluded.name, kind=excluded.kind, confidence=excluded.confidence, samples=excluded.samples,
                sightings=excluded.sightings, last_seen=excluded.last_seen, last_sample_at=excluded.last_sample_at,
                last_camera=excluded.last_camera, clip_embedding=excluded.clip_embedding,
                face_embedding=excluded.face_embedding, thumbnail=excluded.thumbnail
            """,
            (
                profile.profile_id,
                profile.name,
                profile.kind,
                profile.confidence,
                profile.samples,
                profile.sightings,
                profile.first_seen.isoformat(),
                profile.last_seen.isoformat(),
                profile.last_sample_at.isoformat(),
                profile.last_camera,
                profile.clip_embedding.astype(np.float32).tobytes(),
                profile.face_embedding.astype(np.float32).tobytes() if profile.face_embedding is not None else None,
                profile.thumbnail,
            ),
        )
        self._database.commit()

    def _profile_numbers(self) -> list[int]:
        return [
            int(profile_id.removeprefix("person-"))
            for profile_id in self._profiles
            if profile_id.removeprefix("person-").isdigit()
        ]

    @staticmethod
    def _vector(payload: bytes) -> np.ndarray:
        return IdentityLibrary._normalized(np.frombuffer(payload, dtype=np.float32).copy())

    @staticmethod
    def _normalized(vector: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vector)
        if norm == 0:
            raise ValueError("identity embedding must not be empty")
        return vector.astype(np.float32) / norm
