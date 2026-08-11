from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from egg_companion.adapters.vision import FaceCrop, SegmentedObject
from egg_companion.config import (
    EventSegmentationConfig, IdentityConfig, MemoryConfig, ObjectLearningConfig, PrivacyConfig,
)
from egg_companion.memory.migrate_legacy import LegacyMemoryMigrator
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import BoundingBox, Detection, EvidenceRef, PerceptualEvent
from egg_companion.services.identity import IdentityLibrary
from egg_companion.services.object_library import ObjectLibrary


class MigrationVision:
    @staticmethod
    def face_crops(frame, detection):
        return (FaceCrop(frame, 0.96, "face", np.array([1.0, 0.0], dtype=np.float32)),)

    @staticmethod
    def embed_image(image):
        vector = np.array([float(image.mean()) + 1.0, 1.0], dtype=np.float32)
        return vector / np.linalg.norm(vector)


def test_legacy_profile_migration_is_idempotent_and_preserves_provenance(tmp_path) -> None:
    vision = MigrationVision()
    identities = IdentityLibrary(IdentityConfig(storage_dir=str(tmp_path / "identities")))
    frame = np.full((48, 48, 3), 100, dtype=np.uint8)
    for _ in range(3):
        identities.observe(
            "camera-0",
            frame,
            (Detection("person", 0.96, BoundingBox(0, 0, 48, 48)),),
            vision,
        )
    identities.name_most_recent("Ada")

    objects = ObjectLibrary(ObjectLearningConfig(storage_dir=str(tmp_path / "objects")))
    segmented = SegmentedObject(frame, np.full((48, 48), 255, dtype=np.uint8), 0.72)
    profile = objects.learn("wrong label", segmented, vision, "detector", 0.72)
    assert profile is not None
    objects.relabel(profile.profile_id, "ceramic mug", 0.94, "ornith-vlm", "ornith-test")

    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    migrator = LegacyMemoryMigrator(store, identities, objects)
    assert migrator.run() == {"identities": 1, "objects": 1, "media": 2}
    assert migrator.run() == {"identities": 1, "objects": 1, "media": 2}

    entities = store.list_entities(limit=20)
    assert {item["entity_id"] for item in entities} == {"person-001", "object-001"}
    assert next(item for item in entities if item["entity_id"] == "person-001")["entity_type"] == "person"
    object_claims = store.list_claims("object-001", state=None, limit=20)
    assert {(item["object_id_or_text"], item["state"]) for item in object_claims} == {
        ("wrong label", "superseded"),
        ("ceramic mug", "active"),
    }
    assert len(store.embedding_metadata(limit=20)) == 3
    assert len(store.search_evidence(["transparent_mask"], limit=20)) == 1
    assert "vector_blob" not in store.entity_detail("object-001")["embeddings"][0]


def test_pipeline_persists_only_accepted_event_and_links_entities(tmp_path) -> None:
    memory = MemoryConfig(storage_dir=str(tmp_path / "memory"), episode_max_seconds=60)
    config = SimpleNamespace(
        memory=memory, event_segmentation=EventSegmentationConfig(), privacy=PrivacyConfig()
    )
    store = MemoryStore(memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)

    def event(event_id: str, at: datetime) -> PerceptualEvent:
        evidence = EvidenceRef(
            f"evidence-{event_id}", "vision", at, "camera", "camera-0", quality=0.9,
            metadata={"detections": [{"label": "mug"}]},
        )
        return PerceptualEvent(
            event_id, "vision", at, "camera-0", (evidence,), ("object-001",),
            {
                "labels": ["mug"],
                "entities": [
                    {
                        "id": "object-001", "type": "object", "label": "mug",
                        "confidence": 0.9, "source": "object-library",
                    }
                ],
            },
        )

    assert pipeline.ingest(event("episode-1", now)) == (True, 0)
    assert pipeline.ingest(event("repeated-frame", now + timedelta(milliseconds=100))) == (False, 0)
    detail = store.entity_detail("object-001")
    assert detail is not None
    assert [item["evidence_id"] for item in detail["evidence"]] == ["evidence-episode-1"]
    assert detail["episodes"][0]["episode_id"] == "episode-1"
