from datetime import datetime, timedelta, timezone

from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.config import DefaultModeConfig, MemoryConfig
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))


def test_default_mode_replays_provenance_without_manufacturing_semantics(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    store.upsert_entity("object", "amber mug", entity_id="object-1", now=now)
    for index in range(3):
        evidence = EvidenceRef(
            f"vision-{index}", "vision", now, "camera", "camera-0", quality=0.9,
            metadata={"label": "amber mug"},
        )
        store.append_evidence(evidence)
        store.link_entity_evidence("object-1", evidence.evidence_id)
    network = DefaultModeNetwork(store, DefaultModeConfig(replay_limit=8))

    first = network.run_once()

    assert first["phase"] == "complete"
    assert first["replayed_entity_ids"] == ["object-1"]
    assert first["reflections_created"] == 0
    assert first["reflection_ids"] == []
    assert first["curiosity_candidates"] == []

    store.assert_claim_once(
        "object-1", "used_for", "morning coffee", 1.0, now, source="human-answer"
    )
    second = network.run_once()
    assert second["curiosity_candidates"] == []
    store.close()


def test_conversation_history_is_chronological_and_marks_suppression(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    store.append_evidence(
        EvidenceRef(
            "heard-1", "audio", now, "respeaker", "asr", quality=1.0,
            metadata={"transcript": "What am I holding?"},
        )
    )
    store.append_evidence(
        EvidenceRef(
            "reply-1", "action", now + timedelta(seconds=1), "policy", "speech", quality=1.0,
            metadata={
                "candidate_response": "You are holding an amber mug.",
                "spoken": True,
                "reason": "fresh vision",
            },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "reply-2", "action", now + timedelta(seconds=2), "policy", "speech", quality=0.8,
            metadata={
                "candidate_response": "A stale response.",
                "spoken": False,
                "reason": "superseded",
            },
        )
    )

    history = store.conversation_history()

    assert [item["role"] for item in history] == ["heard", "agent", "agent"]
    assert history[1]["status"] == "spoken"
    assert history[2]["status"] == "suppressed"
    store.close()


def test_conversation_history_coalesces_late_memory_and_tool_metadata(tmp_path) -> None:
    store = _store(tmp_path)
    now = datetime.now(timezone.utc)
    store.upsert_entity("person", "Troy", entity_id="person-1", now=now)
    store.upsert_entity("object", "amber mug", entity_id="object-1", now=now)
    store.append_evidence(
        EvidenceRef(
            "heard-context", "audio", now, "respeaker", "asr", quality=1.0,
            metadata={
                "transcript": "This is my amber mug.",
                "context_id": "utterance-1",
                "asr_model": "dual",
                "doa": 91.2,
            },
        )
    )
    store.link_entity_evidence("person-1", "heard-context")
    store.append_evidence(
        EvidenceRef(
            "late-correction", "speech", now + timedelta(milliseconds=200),
            "user-correction", "asr", quality=1.0,
            metadata={
                "transcript": "This is my amber mug.",
                "context_id": "utterance-1",
                "corrected_label": "amber mug",
                "object_id": "object-1",
            },
        )
    )
    store.link_entity_evidence("object-1", "late-correction")
    store.append_evidence(
        EvidenceRef(
            "semantic-context", "audio_semantics", now + timedelta(milliseconds=400),
            "omnius-audio-analyze", "respeaker", quality=0.67,
            metadata={
                "context_id": "utterance-1",
                "classifications": [{"label": "Speech", "confidence": 0.67}],
            },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "reply-context", "action", now + timedelta(seconds=1), "policy", "speech",
            quality=1.0,
            metadata={
                "candidate_response": "I'll remember your amber mug.",
                "spoken": True,
                "reason": "human label learned",
                "context_id": "utterance-1",
                "retrieval_influences": [{"owner_type": "entity", "owner_id": "object-1"}],
                "tool_calls": [
                    {"name": "fresh_vision", "success": True, "duration_ms": 220.0}
                ],
            },
        )
    )

    history = store.conversation_history()

    assert [item["role"] for item in history] == ["heard", "agent"]
    assert all(item["context_id"] == "utterance-1" for item in history)
    labels = {tag["label"] for tag in history[0]["tags"]}
    assert {
        "audio", "dual ASR", "label updated: amber mug", "audio comprehension ✓",
        "Speech 67%", "memory recall ×1", "fresh vision ✓", "person: Troy",
        "object: amber mug",
    } <= labels
    assert history[0]["tool_calls"][0]["name"] == "fresh_vision"
    store.close()
