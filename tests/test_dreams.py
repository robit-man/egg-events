from __future__ import annotations

import numpy as np
from pathlib import Path

from egg_companion.adapters.vision import FaceCrop
from egg_companion.config import DreamsConfig, IdentityConfig
from egg_companion.models import BoundingBox, Detection
from egg_companion.services.dreams import IdentityDreamEngine
from egg_companion.services.identity import IdentityLibrary


class _Vision:
    def __init__(self, embedding: np.ndarray) -> None:
        self.embedding = embedding

    def face_crops(self, frame, detection):
        return (FaceCrop(frame, 0.95, "face", self.embedding),)

    @staticmethod
    def embed_image(frame):
        return np.array((0.2, 0.4, 0.6), dtype=np.float32)


class _ModernEmbedder:
    device = "test"
    model_path = Path("/nonexistent-test-model")

    @staticmethod
    def embed(images: list[bytes]) -> np.ndarray:
        return np.repeat(np.array([[1.0, 0.0]], dtype=np.float32), len(images), axis=0)


class _ComparisonEmbedder(_ModernEmbedder):
    available = True


def _enroll(library: IdentityLibrary, camera: str, embedding: np.ndarray, value: int) -> str:
    frame = np.full((96, 96, 3), value, dtype=np.uint8)
    detection = Detection("person", 0.95, BoundingBox(0, 0, 96, 96))
    for _ in range(3):
        match = library.observe(camera, frame, (detection,), _Vision(embedding))[0]
    return str(match["id"])


def _engine(tmp_path, library: IdentityLibrary) -> IdentityDreamEngine:
    engine = IdentityDreamEngine(
        DreamsConfig(
            model_path=str(tmp_path / "unused"),
            modern_merge_similarity=0.5,
            legacy_merge_similarity=0.45,
        ),
        library,
    )
    engine._embedder = _ModernEmbedder()
    return engine


def test_dream_coalesces_mutual_multimodel_match_and_persists_audit(tmp_path) -> None:
    library = IdentityLibrary(
        IdentityConfig(
            storage_dir=str(tmp_path / "identities"),
            face_similarity_threshold=0.9,
            retroactive_merge_similarity=0.99,
        )
    )
    left = _enroll(library, "front", np.array((1.0, 0.0, 0.0), dtype=np.float32), 40)
    right = _enroll(
        library, "side", np.array((0.82, 0.57236, 0.0), dtype=np.float32), 80
    )

    result = _engine(tmp_path, library).run(set(), "test")

    assert result["merges"] == 1
    assert {result["aliases"][0]["alias_id"], result["aliases"][0]["canonical_id"]} == {
        left,
        right,
    }
    assert library.summary()["canonical_people"] == 1


def test_dream_never_crosses_coobservation_constraint(tmp_path) -> None:
    library = IdentityLibrary(
        IdentityConfig(
            storage_dir=str(tmp_path / "identities"),
            face_similarity_threshold=0.9,
            retroactive_merge_similarity=0.99,
        )
    )
    left = _enroll(library, "front", np.array((1.0, 0.0, 0.0), dtype=np.float32), 40)
    right = _enroll(
        library, "side", np.array((0.82, 0.57236, 0.0), dtype=np.float32), 80
    )

    engine = _engine(tmp_path, library)
    result = engine.run({tuple(sorted((left, right)))}, "test")

    assert result["merges"] == 0
    assert result["conflicts_blocked"] == 1
    assert engine.snapshot()["candidates"][0]["reason"] == "co_observation_conflict"
    assert library.summary()["canonical_people"] == 2


def test_dream_consolidates_dense_reciprocal_cluster_in_one_automatic_pass(tmp_path) -> None:
    library = IdentityLibrary(
        IdentityConfig(
            storage_dir=str(tmp_path / "identities"),
            face_similarity_threshold=0.9999,
            retroactive_merge_similarity=0.9999,
        )
    )
    ids = {
        _enroll(library, "front", np.array((1.0, 0.0, 0.0), dtype=np.float32), 40),
        _enroll(library, "side", np.array((0.82, 0.57236, 0.0), dtype=np.float32), 80),
        _enroll(library, "rear", np.array((0.78, 0.62578, 0.0), dtype=np.float32), 120),
    }

    result = _engine(tmp_path, library).run(set(), "scheduler")

    assert len(ids) == 3
    assert result["merges"] == 2
    assert library.summary()["canonical_people"] == 1


def test_dream_accepts_two_of_three_models_without_reciprocal_margin(tmp_path) -> None:
    library = IdentityLibrary(
        IdentityConfig(
            storage_dir=str(tmp_path / "identities"),
            face_similarity_threshold=0.9999,
        )
    )
    _enroll(library, "front", np.array((1.0, 0.0), dtype=np.float32), 40)
    _enroll(library, "side", np.array((0.1, 0.995), dtype=np.float32), 80)
    engine = IdentityDreamEngine(
        DreamsConfig(
            model_path=str(tmp_path / "unused"),
            modern_merge_similarity=0.5,
            legacy_merge_similarity=0.99,
            comparison_model_path=str(tmp_path / "comparison.onnx"),
            comparison_merge_similarity=0.5,
        ),
        library,
    )
    engine._embedder = _ModernEmbedder()
    engine._comparison_embedder = _ComparisonEmbedder()

    result = engine.run(set(), "test")

    assert result["merges"] == 1
    candidate = engine.snapshot()["candidates"][0]
    assert candidate["comparison_similarity"] == 1.0
    assert candidate["reason"] == "quality_aggregated_multimodel_consensus"


def test_dream_marks_process_orphans_interrupted_on_reopen(tmp_path) -> None:
    library = IdentityLibrary(IdentityConfig(storage_dir=str(tmp_path / "identities")))
    engine = _engine(tmp_path, library)
    engine._database.execute(
        """INSERT INTO dream_runs
        (run_id, requested_by, state, model_id, model_revision, started_at)
        VALUES ('orphan', 'test', 'running', 'test', 'test',
        '2026-01-01T00:00:00+00:00')"""
    )
    engine._database.commit()

    reopened = _engine(tmp_path, library)
    orphan = next(run for run in reopened.snapshot()["runs"] if run["run_id"] == "orphan")

    assert orphan["state"] == "interrupted"
    assert orphan["completed_at"] is not None
    assert "process stopped" in orphan["error"]
