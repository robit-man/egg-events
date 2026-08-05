from datetime import datetime, timezone

import numpy as np
import pytest

from egg_companion.config import MemoryConfig
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def test_store_persists_evidence_graph_and_claim_revisions(tmp_path) -> None:
    config = MemoryConfig(storage_dir=str(tmp_path / "memory"))
    store = MemoryStore(config)
    now = datetime.now(timezone.utc)
    entity = store.upsert_entity("person", entity_id="person-1", display_name="Anonymous")
    evidence = EvidenceRef("evidence-1", "vision", now, "camera", "camera-0", quality=0.9)
    store.append_evidence(evidence)
    episode = store.open_episode(now, novelty=0.6, episode_id="episode-1")
    store.append_episode_evidence(episode, evidence.evidence_id)
    store.link_entity_evidence(entity, evidence.evidence_id)
    claim = store.assert_claim(entity, "has_alias", "Ada", 1.0, now, claim_id="claim-1")
    store.revise_claim(claim, "retract", "user", at=now)
    store.add_embedding("entity", entity, "face", "sface", np.array([1.0, 0.0], dtype=np.float32), 0.9, now)
    store.close_episode(episode, now, "person observed")
    store.close()

    reopened = MemoryStore(config)
    detail = reopened.entity_detail(entity)
    assert detail is not None
    assert detail["claims"][0]["state"] == "retracted"
    assert detail["evidence"][0]["evidence_id"] == evidence.evidence_id
    assert reopened.recent_episodes()[0]["state"] == "closed"


def test_store_rolls_back_invalid_entity_link(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    evidence = EvidenceRef("evidence-1", "vision", datetime.now(timezone.utc), "camera", "camera-0")
    store.append_evidence(evidence)
    with pytest.raises(Exception):
        store.link_entity_evidence("missing", evidence.evidence_id)
    assert store.entity_detail("missing") is None


def test_store_rejects_absolute_media_key(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    evidence = EvidenceRef("evidence-1", "vision", datetime.now(timezone.utc), "camera", "camera-0", media_key="/tmp/frame.jpg")
    with pytest.raises(ValueError):
        store.append_evidence(evidence)


def test_store_integrity_report_detects_no_orphans_or_duplicate_sources(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    store.upsert_entity(
        "object", entity_id="object-001",
        metadata={"source_system": "object-library", "source_profile_id": "object-001"},
    )
    report = store.integrity_report()
    assert report["sqlite_integrity"] == "ok"
    assert report["foreign_key_violations"] == []
    assert report["orphan_entity_embeddings"] == 0
    assert report["duplicate_legacy_sources"] == []


def test_store_persists_media_with_relative_configured_root(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    store = MemoryStore(MemoryConfig(storage_dir="data/memory"))
    media_key, checksum = store.persist_media("legacy/object.png", b"mask")
    assert media_key == "media/legacy/object.png"
    assert (tmp_path / "data/memory" / media_key).read_bytes() == b"mask"
    assert len(checksum) == 64
