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


def test_graph_node_detail_exposes_locally_retained_audio_artifact(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    media_key, checksum = store.persist_media("audio/utterance.wav", b"RIFFaudio")
    evidence = EvidenceRef(
        "audio-1",
        "audio",
        datetime.now(timezone.utc),
        "respeaker",
        "respeaker-asr",
        media_key=media_key,
        quality=0.9,
        metadata={"transcript": "hello"},
    )
    store.append_evidence(evidence, checksum=checksum)

    detail = store.graph_node_detail("evidence", "audio-1")

    assert detail is not None
    assert detail["evidence"]["artifact_url"].endswith("/audio-1/media")
    assert store.evidence_media("audio-1") == (b"RIFFaudio", ".wav")


def test_identity_coalescing_preserves_alias_and_projects_evidence_to_canonical(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path)))
    now = datetime.now(timezone.utc)
    store.upsert_entity("person", entity_id="person-001", now=now)
    store.upsert_entity("person", entity_id="person-002", now=now)
    evidence = EvidenceRef(
        "face-2", "vision", now, "camera", "front", None, 0.9, {"face": True}
    )
    store.append_evidence(evidence)
    store.link_entity_evidence("person-002", "face-2")
    store.open_episode(now, episode_id="episode-face-2")
    store.link_episode_entity("episode-face-2", "person-002", confidence=0.9)

    result = store.coalesce_identity_evidence(
        [{
            "alias_id": "person-002",
            "canonical_id": "person-001",
            "similarity": 0.84,
            "reason": "test",
        }]
    )

    assert result == {"aliases": 1, "evidence_links_copied": 1, "episode_links_copied": 1}
    detail = store.entity_detail("person-001")
    assert detail is not None
    assert [item["evidence_id"] for item in detail["evidence"]] == ["face-2"]
    alias = store.entity_detail("person-002")
    assert alias is not None
    assert alias["entity"]["merged_into"] == "person-001"
    graph = store.knowledge_graph_snapshot()
    assert "entity:person-001" in {node["id"] for node in graph["nodes"]}
    assert "entity:person-002" not in {node["id"] for node in graph["nodes"]}


def test_identity_coobservation_conflicts_include_shared_evidence_and_episode(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path)))
    now = datetime.now(timezone.utc)
    for profile_id in ("person-001", "person-002", "person-003"):
        store.upsert_entity("person", entity_id=profile_id, now=now)
    evidence = EvidenceRef("shared", "vision", now, "camera", "front", None, 0.9, {})
    store.append_evidence(evidence)
    store.link_entity_evidence("person-001", "shared")
    store.link_entity_evidence("person-002", "shared")
    store.open_episode(now, episode_id="shared-episode")
    store.link_episode_entity("shared-episode", "person-002")
    store.link_episode_entity("shared-episode", "person-003")

    conflicts = store.identity_coobservation_conflicts(
        ["person-001", "person-002", "person-003"]
    )

    assert conflicts == {("person-001", "person-002"), ("person-002", "person-003")}


def test_strong_identity_conflicts_require_repetition_or_distinct_boxes(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path)))
    now = datetime.now(timezone.utc)
    for profile_id in ("person-001", "person-002", "person-003"):
        store.upsert_entity("person", entity_id=profile_id, now=now)
    for index in range(3):
        evidence = EvidenceRef(
            f"shared-{index}", "vision", now, "camera", "front", None, 0.9, {}
        )
        store.append_evidence(evidence)
        store.link_entity_evidence("person-001", evidence.evidence_id)
        store.link_entity_evidence("person-002", evidence.evidence_id)
        if index < 2:
            assert store.identity_strong_coobservation_conflicts(
                ["person-001", "person-002"], 3
            ) == set()
    spatial = EvidenceRef(
        "spatial", "vision", now, "camera", "front", None, 0.9,
        {
            "detections": [
                {"identity_id": "person-001", "bbox": {"x1": 0, "y1": 0, "x2": 80, "y2": 160}},
                {"identity_id": "person-003", "bbox": {"x1": 220, "y1": 0, "x2": 300, "y2": 160}},
            ]
        },
    )
    store.append_evidence(spatial)
    store.link_entity_evidence("person-001", spatial.evidence_id)
    store.link_entity_evidence("person-003", spatial.evidence_id)

    assert store.identity_strong_coobservation_conflicts(
        ["person-001", "person-002", "person-003"], 3
    ) == {("person-001", "person-002"), ("person-001", "person-003")}


def test_store_exposes_bounded_multimodal_graph_snapshot(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    now = datetime.now(timezone.utc)
    store.upsert_entity("object", "television", entity_id="object-tv", now=now)
    store.upsert_entity("content", "Channel 8", entity_id="content-channel", now=now)
    evidence = EvidenceRef(
        "evidence-ocr",
        "ocr",
        now,
        "camera-advanced-ocr",
        "camera-0",
        quality=0.91,
        metadata={"text": "Channel 8"},
    )
    store.append_evidence(evidence)
    store.link_entity_evidence("content-channel", evidence.evidence_id)
    store.link_entities_once(
        "object-tv", "contains_text", "content-channel", 0.91, now, evidence_id=evidence.evidence_id
    )

    graph = store.knowledge_graph_snapshot(50)

    assert {node["id"] for node in graph["nodes"]} >= {
        "entity:object-tv",
        "entity:content-channel",
        "evidence:evidence-ocr",
    }
    relation = next(link for link in graph["links"] if link["relation"] == "contains_text")
    assert relation["source"] == "entity:object-tv"
    assert relation["target"] == "entity:content-channel"
    assert relation["confidence"] == pytest.approx(0.91)
    assert graph["counts"]["links"] >= 2


def test_dashboard_summaries_do_not_inline_large_narrative_documents(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    now = datetime.now(timezone.utc)
    large_ledger = [{"summary": "retained evidence " * 100}] * 50
    store.upsert_entity(
        "daily_narrative",
        "Daily story",
        {
            "document_kind": "daily-narrative",
            "revision": 4,
            "local_date": "2026-08-14",
            "abstract_summary": "A grounded day.",
            "timeline": large_ledger,
            "conversation_ledger": large_ledger,
        },
        "daily-narrative:2026-08-14",
        now=now,
    )

    summaries = store.list_entity_summaries(limit=24)
    graph = store.knowledge_graph_snapshot(50)
    narrative = next(
        node for node in graph["nodes"]
        if node["id"] == "entity:daily-narrative:2026-08-14"
    )

    assert "metadata" not in summaries[0]
    assert narrative["metadata"] == {
        "document_kind": "daily-narrative",
        "revision": 4,
        "local_date": "2026-08-14",
        "abstract_summary": "A grounded day.",
    }
    assert store.entity_detail("daily-narrative:2026-08-14")["entity"][
        "metadata"
    ]["timeline"] == large_ledger
