from datetime import datetime, timedelta, timezone

from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.config import DefaultModeConfig, MemoryConfig
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def _store(tmp_path) -> MemoryStore:
    return MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))


def test_default_mode_replays_source_backed_gap_and_stops_after_answer(tmp_path) -> None:
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
    network = DefaultModeNetwork(
        store,
        DefaultModeConfig(
            reflection_min_evidence=2, curiosity_threshold=0.4, replay_limit=8
        ),
    )

    first = network.run_once()

    assert first["phase"] == "complete"
    assert first["replayed_entity_ids"] == ["object-1"]
    assert first["reflections_created"] == 1
    assert first["curiosity_candidates"][0]["predicate"] == "used_for"

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
