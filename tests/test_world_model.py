from datetime import datetime, timedelta, timezone

from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.cognition.world_model import WorldModelSynthesizer
from egg_companion.config import DefaultModeConfig, MemoryConfig
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def test_default_mode_projects_recurrent_evidence_into_revisable_meta_graph(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    now = datetime.now(timezone.utc)
    store.upsert_entity("person", "Troy", entity_id="person-1", now=now)
    store.upsert_entity("object", "amber mug", entity_id="object-1", now=now)
    for index in range(3):
        at = now + timedelta(hours=index)
        episode_id = f"episode-{index}"
        evidence_id = f"vision-{index}"
        store.append_evidence(
            EvidenceRef(
                evidence_id,
                "vision",
                at,
                "camera",
                "camera-0",
                quality=0.9,
                metadata={"label": "amber mug"},
            )
        )
        store.open_episode(at, episode_id=episode_id)
        store.append_episode_evidence(episode_id, evidence_id)
        for entity_id in ("person-1", "object-1"):
            store.link_entity_evidence(entity_id, evidence_id)
            store.link_episode_entity(episode_id, entity_id, confidence=0.9)
        store.close_episode(episode_id, at, "Troy and the amber mug were observed")
    store.append_evidence(
        EvidenceRef(
            "response-1",
            "action",
            now,
            "interaction-policy",
            "speech-output",
            quality=1.0,
            metadata={
                "input_transcript": "Is that my amber mug?",
                "candidate_response": "Yes, that is your amber mug.",
                "spoken": True,
                "reason": "fresh question-conditioned camera evidence",
            },
        )
    )
    network = DefaultModeNetwork(
        store,
        DefaultModeConfig(
            reflection_min_evidence=2,
            meta_graph_min_confirmations=2,
            meta_graph_limit=10,
        ),
    )

    result = network.run_once()

    assert result["meta_graph"]["abstractions_projected"] == 1
    assert len(result["meta_graph"]["documents"]) == 4
    graph = store.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}
    assert {
        "supports_pattern",
        "recurrently_associated_with",
        "informs_world_model",
        "maintains",
        "guides_communication",
    } <= relations
    abstractions = [
        node for node in graph["nodes"] if node.get("subtype") == "abstraction"
    ]
    assert len(abstractions) == 1
    detail = store.entity_detail(str(abstractions[0]["source_id"]))
    assert detail is not None
    assert detail["entity"]["metadata"]["epistemic_status"] == "inferred_noncausal"
    assert len(detail["entity"]["metadata"]["source_episode_ids"]) == 3

    context = ContextAssembler(store).build(
        "Is that my mug?", "Troy and one object are visible", ("person-1",)
    )
    assert "REFLECTIVE WORKING MODEL" in context
    assert "communication-strategy" in context
    assert "association, not causation" in context
    store.close()


def test_dream_replays_daily_chronology_and_revises_story_meta_graph(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    config = DefaultModeConfig(
        narrative_timezone="UTC",
        narrative_bucket_minutes=15,
        narrative_replay_max_days=30,
        meta_graph_min_confirmations=2,
    )
    synthesizer = WorldModelSynthesizer(store, config)
    morning = datetime(2026, 8, 14, 9, 5, tzinfo=timezone.utc)
    afternoon = datetime(2026, 8, 14, 15, 35, tzinfo=timezone.utc)
    store.upsert_entity("person", "Troy", entity_id="person-troy", now=morning)
    store.upsert_entity(
        "person", "person fragment", entity_id="person-fragment", now=morning
    )
    store.upsert_entity("object", "amber mug", entity_id="object-mug", now=morning)
    store.upsert_entity("content", "ORBITAL 42", entity_id="content-sign", now=morning)
    observations = [
        (
            morning,
            "morning-vision",
            "vision",
            {"summary": "Troy held the amber mug"},
            ("person-troy", "person-fragment", "object-mug"),
        ),
        (
            afternoon,
            "afternoon-ocr",
            "ocr",
            {"text": "ORBITAL 42"},
            ("person-troy", "content-sign"),
        ),
        (
            afternoon + timedelta(minutes=2),
            "afternoon-audio",
            "audio",
            {"transcript": "This is the orbital display", "admitted": True},
            ("person-troy", "content-sign"),
        ),
    ]
    for index, (captured_at, evidence_id, modality, metadata, entity_ids) in enumerate(
        observations
    ):
        episode_id = f"day-episode-{index}"
        store.append_evidence(
            EvidenceRef(
                evidence_id,
                modality,
                captured_at,
                "sensor",
                f"source-{index}",
                quality=0.9,
                metadata=metadata,
            )
        )
        store.open_episode(captured_at, episode_id=episode_id)
        store.append_episode_evidence(episode_id, evidence_id)
        for entity_id in entity_ids:
            store.link_entity_evidence(entity_id, evidence_id)
            store.link_episode_entity(episode_id, entity_id, confidence=0.9)
        store.close_episode(
            episode_id, captured_at + timedelta(seconds=5), str(metadata)
        )
    store.coalesce_identity_evidence(
        [
            {
                "canonical_id": "person-troy",
                "alias_id": "person-fragment",
                "similarity": 0.93,
                "reason": "test-dream",
            }
        ]
    )

    result = synthesizer.replay_dream(
        {
            "run_id": "dream-story-1",
            "requested_by": "scheduler",
            "profiles_examined": 2,
            "merges": 1,
            "aliases": [
                {
                    "canonical_id": "person-troy",
                    "alias_id": "person-fragment",
                    "similarity": 0.93,
                }
            ],
        },
        afternoon + timedelta(hours=1),
    )

    assert result["days_replayed"] == 1
    assert result["daily_narratives"][0]["local_date"] == "2026-08-14"
    detail = store.entity_detail("daily-narrative:2026-08-14")
    assert detail is not None
    metadata = detail["entity"]["metadata"]
    assert metadata["revision"] == 1
    assert len(metadata["timeline"]) == 2
    assert metadata["timeline"][0]["local_time"] == "09:05"
    assert metadata["timeline"][1]["local_time"] == "15:35–15:37"
    assert "Troy" in metadata["timeline"][0]["summary"]
    assert "amber mug" in metadata["timeline"][0]["summary"]
    assert "ORBITAL 42" in metadata["timeline"][1]["summary"]
    assert "This is the orbital display" in metadata["timeline"][1]["summary"]
    assert set(metadata["source_evidence_ids"]) == {
        "morning-vision",
        "afternoon-ocr",
        "afternoon-audio",
    }
    index = store.daily_narrative_index()
    assert index[0]["local_date"] == "2026-08-14"
    assert index[0]["timeline_entries"] == 2
    narrative = store.daily_narrative_detail("2026-08-14")
    assert narrative is not None
    assert len(narrative["timeline"][1]["episodes"]) == 2
    assert {item["modality"] for item in narrative["timeline"][1]["artifacts"]} == {
        "ocr",
        "audio",
    }
    assert any(
        item.get("text") == "This is the orbital display"
        for item in narrative["timeline"][1]["artifacts"]
    )
    documents = {
        item["metadata"]["document_kind"]: item["metadata"]
        for item in store.cognitive_documents()
    }
    assert "2026-08-14" in documents["my-story"]["content"]
    assert "People encountered: Troy" in documents["my-story"]["content"]
    graph = store.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}
    assert {
        "enters_dream_replay",
        "replays_day",
        "appears_in_day",
        "observed_in_day",
        "read_in_day",
        "experienced_day",
        "contributes_to_story",
    } <= relations

    second = synthesizer.replay_dream(
        {
            "run_id": "dream-story-2",
            "requested_by": "scheduler",
            "profiles_examined": 0,
            "merges": 0,
            "aliases": [],
        },
        afternoon + timedelta(hours=2),
    )
    assert second["days_replayed"] == 1
    assert second["daily_narratives"][0]["changed"] is False
    assert store.entity_detail("daily-narrative:2026-08-14")["entity"]["metadata"][
        "revision"
    ] == 1
    store.close()


def test_narrative_replay_backfills_every_unreviewed_day_oldest_first(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    synthesizer = WorldModelSynthesizer(
        store,
        DefaultModeConfig(
            narrative_timezone="UTC",
            narrative_replay_max_days=2,
        ),
    )
    first_day = datetime(2026, 8, 10, 10, 0, tzinfo=timezone.utc)
    store.upsert_entity("object", "daily object", entity_id="object-daily", now=first_day)
    for offset in range(4):
        captured_at = first_day + timedelta(days=offset)
        evidence_id = f"historical-{offset}"
        store.append_evidence(
            EvidenceRef(
                evidence_id,
                "vision",
                captured_at,
                "camera",
                "camera-0",
                quality=0.8,
                metadata={"summary": f"retained observation {offset}"},
            )
        )
        store.link_entity_evidence("object-daily", evidence_id)

    first = synthesizer.replay_dream(
        {"run_id": "catchup-1", "requested_by": "startup", "aliases": []},
        first_day + timedelta(days=4),
    )
    assert first["history_days_discovered"] == 4
    assert first["backlog_before"] == 4
    assert first["backfilled_days"] == ["2026-08-10", "2026-08-13"]
    assert first["backlog_remaining"] == 2

    second = synthesizer.replay_dream(
        {"run_id": "catchup-2", "requested_by": "startup", "aliases": []},
        first_day + timedelta(days=4, minutes=1),
    )
    assert second["backlog_before"] == 2
    assert second["backfilled_days"] == ["2026-08-11"]
    assert second["backlog_remaining"] == 1

    third = synthesizer.replay_dream(
        {"run_id": "catchup-3", "requested_by": "startup", "aliases": []},
        first_day + timedelta(days=4, minutes=2),
    )
    assert third["backlog_before"] == 1
    assert third["backfilled_days"] == ["2026-08-12"]
    assert third["backlog_remaining"] == 0
    assert [item["local_date"] for item in store.daily_narrative_index()] == [
        "2026-08-13",
        "2026-08-12",
        "2026-08-11",
        "2026-08-10",
    ]
    store.close()
