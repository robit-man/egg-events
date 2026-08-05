from datetime import datetime, timedelta, timezone

from egg_companion.config import MemoryConfig, PrivacyConfig
from egg_companion.memory.consolidation import MemoryConsolidator
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def test_consolidation_summarizes_flags_conflicts_and_expires_media(tmp_path) -> None:
    memory = MemoryConfig(
        storage_dir=str(tmp_path / "memory"), raw_media_retention_hours=1,
        consolidation_batch_size=10,
    )
    store = MemoryStore(memory)
    now = datetime.now(timezone.utc)
    observed_at = now - timedelta(hours=2)
    store.upsert_entity("object", "mug", entity_id="object-001", now=observed_at)
    media_key, checksum = store.persist_media("episodes/mug.jpg", b"real-image-bytes")
    evidence = EvidenceRef(
        "evidence-1", "vision", observed_at, "camera", "camera-0", media_key, 0.9,
        {"detections": [{"label": "mug"}]},
    )
    store.append_evidence(evidence, checksum=checksum)
    store.open_episode(observed_at, 0.8, "episode-1")
    store.append_episode_evidence("episode-1", evidence.evidence_id)
    store.link_entity_evidence("object-001", evidence.evidence_id)
    store.link_episode_entity("episode-1", "object-001", confidence=0.9)
    store.close_episode("episode-1", observed_at + timedelta(seconds=4))
    store.assert_claim("object-001", "has_label", "mug", 0.8, observed_at)
    store.assert_claim("object-001", "has_label", "cup", 0.7, observed_at)

    result = MemoryConsolidator(store, PrivacyConfig(evidence_retention_days=30)).run_once()

    assert result["summarized_episodes"] == 1
    assert result["expired_media"] == 1
    assert result["expired_evidence"] == 0
    assert result["claim_conflicts"][0]["subject_id"] == "object-001"
    assert store.episode_detail("episode-1")["episode"]["summary"] == "Observed: mug"
    assert store.entity_detail("object-001")["evidence"][0]["media_key"] is None
    assert store.list_jobs()[0]["state"] == "complete"
