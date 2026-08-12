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


def test_pipeline_projects_temporal_track_alias_into_one_person(tmp_path) -> None:
    memory = MemoryConfig(storage_dir=str(tmp_path / "memory"))
    config = SimpleNamespace(
        memory=memory,
        event_segmentation=EventSegmentationConfig(),
        privacy=PrivacyConfig(),
    )
    store = MemoryStore(memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    evidence = EvidenceRef(
        "temporal-comparison-1",
        "vision",
        now,
        "ornith-temporal-person",
        "front",
        quality=0.91,
        metadata={
            "analysis": "Visible clothing remains consistent.",
            "displacement_analysis": "The mask moves 18 px right.",
        },
    )
    event = PerceptualEvent(
        "temporal-event-1",
        "identity",
        now,
        "temporal-person-continuity",
        (evidence,),
        ("track-000001", "person-001"),
        {
            "labels": ["temporal person continuity"],
            "entities": [
                {
                    "id": "track-000001",
                    "type": "appearance_track",
                    "confidence": 0.91,
                },
                {
                    "id": "person-001",
                    "type": "person",
                    "confidence": 0.91,
                },
            ],
            "identity_alias": {
                "alias_id": "track-000001",
                "canonical_id": "person-001",
                "similarity": 0.91,
                "reason": "adjacent_mask_overlap_vlm_confirmed",
            },
            "skip_pairwise_co_observation": True,
        },
    )

    assert pipeline.ingest(event) == (True, 0)
    alias = store.entity_detail("track-000001")
    canonical = store.entity_detail("person-001")

    assert alias is not None
    assert alias["entity"]["merged_into"] == "person-001"
    assert canonical is not None
    assert [item["evidence_id"] for item in canonical["evidence"]] == [
        "temporal-comparison-1"
    ]


def test_ornith_relabel_persists_crop_and_supersedes_automatic_memory_labels(tmp_path) -> None:
    memory = MemoryConfig(storage_dir=str(tmp_path / "memory"))
    config = SimpleNamespace(
        memory=memory, event_segmentation=EventSegmentationConfig(), privacy=PrivacyConfig()
    )
    store = MemoryStore(memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    embedding = np.array([1.0, 0.0], dtype=np.float32)
    base = {
        "profile_id": "object-001",
        "label": "tv genre",
        "embedding": embedding,
        "confidence": 0.62,
        "label_confidence": 0.62,
        "label_source": "detector",
        "review_state": "pending",
        "last_seen": now,
        "thumbnail": b"segmented-png-evidence",
        "label_history": [],
        "label_provenance": {},
    }
    pipeline.sync_object_profile(base)
    store.assert_claim_once(
        "object-001", "has_alias", "tv genre", 0.62, now, source="runtime"
    )

    corrected = {
        **base,
        "label": "computer monitor",
        "label_confidence": 0.94,
        "label_source": "ornith-vlm",
        "review_state": "vlm_verified",
        "label_history": [
            {
                "label": "tv genre",
                "source": "detector",
                "confidence": 0.62,
                "revised_at": now.isoformat(),
                "revised_by": "robit/ornith-vision:9b",
            }
        ],
        "label_provenance": {"model_id": "robit/ornith-vision:9b"},
    }
    pipeline.sync_object_profile(corrected)

    detail = store.entity_detail("object-001")
    assert detail is not None
    assert detail["entity"]["display_name"] == "computer monitor"
    assert len(detail["evidence"]) == 2
    assert all(item.get("media_key") for item in detail["evidence"])
    active = store.list_claims("object-001", state="active", limit=20)
    assert {(item["predicate"], item["object_id_or_text"]) for item in active} == {
        ("has_label", "computer monitor")
    }
    assert store.memory_stats()["revisions"] >= 2
