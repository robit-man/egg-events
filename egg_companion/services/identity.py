from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from egg_companion.adapters.vision import FaceCrop, VisionEngine
from egg_companion.config import IdentityConfig
from egg_companion.memory.fusion import EvidenceFusion
from egg_companion.models import Detection


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


class IdentityLibrary:
    """Persistent on-device identity profiles with face-first anonymous recall."""

    def __init__(self, config: IdentityConfig) -> None:
        self.config = config
        self._directory = Path(config.storage_dir)
        self._lock = threading.RLock()
        self._profiles: dict[str, IdentityProfile] = {}
        self._database: sqlite3.Connection | None = None
        self._last_match_components: dict[str, float] = {}
        self._last_match_outcome = "new"
        if config.enabled:
            self._directory.mkdir(parents=True, exist_ok=True)
            self._database = sqlite3.connect(self._directory / "identities.sqlite3", check_same_thread=False)
            self._database.row_factory = sqlite3.Row
            self._create_schema()
            self._load()

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
        for index, detection in enumerate(detections):
            if detection.label != "person":
                continue
            faces = vision.face_crops(frame, detection)
            if not faces:
                continue
            face = max(faces, key=lambda item: item.image.shape[0] * item.image.shape[1])
            clip_embedding = self._normalized(vision.embed_image(face.image))
            face_embedding = self._normalized(face.face_embedding) if face.face_embedding is not None else None
            profile, created, similarity = self._match_or_create(clip_embedding, face_embedding, face.kind)
            now = datetime.now(timezone.utc)
            with self._lock:
                profile.last_seen = now
                profile.last_camera = camera_id
                if not created:
                    profile.sightings += 1
                profile.confidence = face.confidence if created else min(face.confidence, similarity)
                if created:
                    self._save(profile, face.image)
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
                    self._save(profile, face.image)
                else:
                    self._store(profile)
            matches[index] = {
                "id": profile.profile_id,
                "label": profile.name or profile.profile_id,
                "confidence": round(profile.confidence, 3),
                "needs_name": profile.name is None,
                "new": created,
                "recalled": not created,
                "kind": profile.kind,
                "sightings": profile.sightings,
                "resolver_outcome": "new" if created else self._last_match_outcome,
                "confidence_components": dict(self._last_match_components),
            }
        return matches

    def name_most_recent(self, name: str) -> IdentityProfile | None:
        normalized_name = " ".join(name.strip().split())
        if not normalized_name or len(normalized_name) > 64:
            return None
        with self._lock:
            unnamed = [profile for profile in self._profiles.values() if profile.name is None]
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
            profile = self._profiles.get(profile_id)
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
                self._database.execute("DELETE FROM profiles WHERE profile_id=?", (profile_id,))
                self._database.commit()
            return True

    def snapshot(self) -> list[dict[str, object]]:
        with self._lock:
            return [
                {
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
                }
                for profile in sorted(self._profiles.values(), key=lambda item: item.last_seen, reverse=True)
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
                }
                for profile in self._profiles.values()
            ]

    def profile_record(self, profile_id: str) -> dict[str, object] | None:
        return next(
            (profile for profile in self.migration_profiles() if profile["profile_id"] == profile_id),
            None,
        )

    def _match_or_create(
        self, clip_embedding: np.ndarray, face_embedding: np.ndarray | None, kind: str
    ) -> tuple[IdentityProfile, bool, float]:
        with self._lock:
            if face_embedding is not None:
                candidates = [profile for profile in self._profiles.values() if profile.face_embedding is not None]
                if candidates:
                    profile, score = max(
                        ((item, float(np.dot(face_embedding, item.face_embedding))) for item in candidates),
                        key=lambda item: item[1],
                    )
                    if score >= self.config.face_similarity_threshold:
                        clip_score = float(np.dot(clip_embedding, profile.clip_embedding))
                        fusion = EvidenceFusion.person(score, clip_score)
                        self._last_match_components = fusion.components
                        self._last_match_outcome = "recalled"
                        return profile, False, score
            else:
                candidates = [profile for profile in self._profiles.values() if profile.kind == kind]
                if candidates:
                    profile, score = max(
                        ((item, float(np.dot(clip_embedding, item.clip_embedding))) for item in candidates),
                        key=lambda item: item[1],
                    )
                    if score >= self.config.similarity_threshold:
                        fusion = EvidenceFusion.person(None, score)
                        self._last_match_components = fusion.components
                        self._last_match_outcome = fusion.outcome
                        return profile, False, score
            now = datetime.now(timezone.utc)
            profile_id = f"person-{max(self._profile_numbers(), default=0) + 1:03d}"
            profile = IdentityProfile(
                profile_id=profile_id,
                clip_embedding=self._normalized(clip_embedding),
                face_embedding=self._normalized(face_embedding) if face_embedding is not None else None,
                samples=1,
                sightings=1,
                first_seen=now,
                last_seen=now,
                last_sample_at=now,
                kind=kind,
            )
            self._profiles[profile_id] = profile
            fusion = EvidenceFusion.person(
                1.0 if face_embedding is not None else None,
                1.0,
            )
            self._last_match_components = fusion.components
            self._last_match_outcome = "new"
            return profile, True, 1.0

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
            """
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

    def _save(self, profile: IdentityProfile, crop: np.ndarray) -> None:
        import cv2

        success, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not success:
            raise RuntimeError("failed to encode identity crop")
        profile.thumbnail = encoded.tobytes()
        self._store(profile)

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
