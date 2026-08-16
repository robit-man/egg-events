import hashlib
from datetime import datetime, timedelta, timezone

from egg_companion.cognition.default_mode import DefaultModeNetwork
from egg_companion.cognition.world_model import WorldModelSynthesizer
from egg_companion.config import DefaultModeConfig, MemoryConfig
from egg_companion.memory.context import ContextAssembler
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef


def test_default_mode_does_not_infer_semantics_from_recurrence(tmp_path) -> None:
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
    network = DefaultModeNetwork(store, DefaultModeConfig(meta_graph_limit=10))

    result = network.run_once()

    assert result["meta_graph"]["abstractions_projected"] == 0
    assert len(result["meta_graph"]["documents"]) == 4
    graph = store.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}
    assert {"maintains", "guides_communication"} <= relations
    assert "supports_pattern" not in relations
    assert "recurrently_associated_with" not in relations
    abstractions = [
        node for node in graph["nodes"] if node.get("subtype") == "abstraction"
    ]
    assert abstractions == []

    context = ContextAssembler(store).build(
        "Is that my mug?", "Troy and one object are visible", ("person-1",)
    )
    assert "REFLECTIVE WORKING MODEL" in context
    assert "communication-strategy" in context
    assert "No model-authored higher-order theme is active yet" in context
    store.close()


def test_dream_replays_daily_chronology_and_revises_story_meta_graph(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    config = DefaultModeConfig(
        narrative_timezone="UTC",
        narrative_bucket_minutes=15,
        narrative_replay_max_days=30,
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


def test_daily_narrative_ranks_interaction_and_collapses_repeated_camera_labels(
    tmp_path,
) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    synthesizer = WorldModelSynthesizer(
        store, DefaultModeConfig(narrative_timezone="UTC")
    )
    observed_at = datetime(2026, 8, 14, 4, 0, tzinfo=timezone.utc)
    for index in range(10):
        store.append_evidence(
            EvidenceRef(
                f"camera-{index}",
                "vision",
                observed_at + timedelta(seconds=index * 20),
                "camera",
                "camera-0",
                quality=0.8,
                metadata={
                    "detections": [
                        {"label": "computer monitor"},
                        {"label": "pillow"},
                    ]
                },
            )
        )
    store.append_evidence(
        EvidenceRef(
            "spoken-action",
            "action",
            observed_at + timedelta(minutes=5),
            "interaction-policy",
            "speech-output",
            quality=1.0,
            metadata={
                "candidate_response": "What should I call you?",
                "spoken": True,
            },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "screen-text",
            "ocr",
            observed_at + timedelta(minutes=6),
            "advanced-ocr",
            "camera-0",
            quality=0.9,
            metadata={"text": "SYSTEM READY"},
        )
    )
    store.append_evidence(
        EvidenceRef(
            "historical-rejected-transcript",
            "speech",
            observed_at + timedelta(minutes=7),
            "asr",
            "microphone",
                quality=0.1,
                metadata={"transcript": "Thanks for watching!", "admitted": False},
        )
    )

    replay = synthesizer.replay_dream(
        {"run_id": "rich-story", "requested_by": "startup", "aliases": []},
        observed_at + timedelta(hours=1),
    )
    period = replay["daily_narratives"][0]
    detail = store.daily_narrative_detail(str(period["local_date"]))
    assert detail is not None
    summary = detail["timeline"][0]["summary"]
    assert 'Egg replied: “What should I call you?”' in summary
    assert "Read: SYSTEM READY" in summary
    assert "Detector counts across 10 retained camera updates" in summary
    assert "computer monitor (10)" in summary
    assert "Observed:" not in summary
    assert "Thanks for watching" not in summary
    assert detail["timeline"][0]["recurring_detections"] == [
        {"label": "computer monitor", "frames": 10},
        {"label": "pillow", "frames": 10},
    ]
    store.close()


def test_daily_narrative_compounds_dialogue_into_policy_and_meta_graph(tmp_path) -> None:
    store = MemoryStore(MemoryConfig(storage_dir=str(tmp_path / "memory")))
    synthesizer = WorldModelSynthesizer(
        store, DefaultModeConfig(narrative_timezone="UTC")
    )
    observed_at = datetime(2026, 8, 14, 8, 0, tzinfo=timezone.utc)
    store.append_evidence(
        EvidenceRef(
            "heard-microscope",
            "speech",
            observed_at,
            "asr",
            "microphone",
            quality=0.95,
            metadata={
                "transcript": "The microscope is my microscope for inspecting circuit boards.",
                "context_id": "turn-microscope",
                "admitted": True,
            },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "reply-microscope",
            "action",
            observed_at + timedelta(seconds=2),
            "interaction-policy",
            "speech-output",
            quality=1.0,
            metadata={
                "candidate_response": "I will remember the microscope is used for inspecting circuit boards.",
                "context_id": "turn-microscope",
                "spoken": True,
                "memory_update": "microscope used_for inspecting circuit boards",
            },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "heard-open-question",
            "speech",
            observed_at + timedelta(minutes=1),
            "asr",
            "microphone",
            quality=0.95,
                metadata={
                    "transcript": "Where did I put the microscope?",
                    "context_id": "turn-open-question",
                    "admitted": True,
                    "directed": True,
                },
        )
    )
    store.append_evidence(
        EvidenceRef(
            "ambient-podcast-question",
            "audio",
            observed_at + timedelta(minutes=2),
            "respeaker",
            "respeaker-asr",
            quality=0.9,
            metadata={
                "transcript": "Where are you and what are you doing in the studio?",
                "context_id": "ambient-podcast",
                "vad_accepted": True,
            },
        )
    )

    replay = synthesizer.replay_dream(
        {"run_id": "semantic-story", "requested_by": "startup", "aliases": []},
        observed_at + timedelta(hours=1),
    )
    detail = store.daily_narrative_detail("2026-08-14")
    assert detail is not None
    assert detail["semantic_context"]["state"] == "pending_model_semantics"
    ledger = detail["conversation_ledger"]
    assert ledger["dialogue_turns"] == 3
    assert ledger["ambient_discourse_turns"] == 1
    assert ledger["heard_turns"] == 3
    assert ledger["focus_terms"] == []
    assert "memory update: microscope used_for inspecting circuit boards" in ledger["learned_context"]
    assert "Conversation and developing meaning" in detail["content"]
    assert replay["daily_narratives"][0]["semantic_links"] == 0

    pending = store.pending_narrative_semantics(1)[0]
    model_semantics = {
        "state": "model_complete",
        "narrative_summary": "The speaker established how the microscope fits into their electronics work.",
        "story_update": "I learned the microscope supports circuit-board inspection.",
        "themes": [
            {
                "label": "microscope-assisted electronics work",
                "summary": "A tool and its role were explained in conversation.",
                "confidence": 0.96,
                "entity_ids": [],
                "evidence_ids": ["heard-microscope", "reply-microscope"],
                "context_ids": ["turn-microscope"],
            }
        ],
        "topics": [],
        "focus_terms": ["microscope-assisted electronics work"],
        "episodes": [
            {
                "title": "Explaining the microscope",
                "summary": "The speaker explained the tool's role.",
                "significance": "The exchange connected a seen tool to its use.",
                "confidence": 0.95,
                "started_at": observed_at.isoformat(),
                "ended_at": (observed_at + timedelta(minutes=1)).isoformat(),
                "entity_ids": [],
                "evidence_ids": ["heard-microscope", "reply-microscope"],
                "context_ids": ["turn-microscope"],
            }
        ],
        "unresolved_questions": [
            {
                "summary": "Where the microscope was placed remains unresolved.",
                "confidence": 0.8,
                "entity_ids": [],
                "evidence_ids": ["heard-microscope"],
                "context_ids": ["turn-microscope"],
            }
        ],
        "learned_context": [
            {
                "summary": "The microscope is used to inspect circuit boards.",
                "confidence": 0.96,
                "entity_ids": [],
                "evidence_ids": ["heard-microscope", "reply-microscope"],
                "context_ids": ["turn-microscope"],
            }
        ],
    }
    theme_id = "narrative-theme:" + hashlib.sha256(
        b"microscope-assisted electronics work"
    ).hexdigest()[:20]
    policy = {
        "state": "model_complete",
        "focus_terms": ["locate the microscope when grounded evidence changes"],
        "focus_entity_ids": [],
        "open_questions": ["Where is the microscope now?"],
        "theme_ids": [theme_id],
        "directive": "Follow the unresolved location only when new grounded evidence bears on it.",
        "summary": "Use the learned tool relationship without treating its current location as known.",
        "attend_to": [],
        "deprioritize": [],
    }
    invented = {
        **model_semantics,
        "themes": [
            {
                **model_semantics["themes"][0],
                "evidence_ids": ["invented-artifact"],
            }
        ],
    }
    assert not store.apply_narrative_semantics(
        "2026-08-14",
        str(pending["semantic_input_fingerprint"]),
        invented,
        policy,
        None,
        "test-semantic-model",
        [],
        observed_at + timedelta(hours=2),
    )
    applied = store.apply_narrative_semantics(
        "2026-08-14",
        str(pending["semantic_input_fingerprint"]),
        model_semantics,
        policy,
        "Continue integrating dialogue and perception while preserving provenance.",
        "test-semantic-model",
        [{"tool": "memory_search", "success": True}],
        observed_at + timedelta(hours=2),
    )
    assert applied
    chapter = store.daily_narrative_detail("2026-08-14")
    assert chapter is not None
    assert chapter["abstract_summary"] == model_semantics["narrative_summary"]
    assert str(chapter["content"]).startswith("## Model-authored daily account")
    assert "## Chronological provenance ledger" in str(chapter["content"])
    assert "provenance_abstract_summary" in store.entity_detail(
        "daily-narrative:2026-08-14"
    )["entity"]["metadata"]
    store.upsert_entity(
        "daily_narrative",
        metadata={"content": "legacy pending display", "abstract_summary": "legacy"},
        entity_id="daily-narrative:2026-08-14",
        now=observed_at + timedelta(hours=2, minutes=1),
    )
    assert store.refresh_model_narrative_documents() == 1
    refreshed = store.daily_narrative_detail("2026-08-14")
    assert refreshed is not None
    assert refreshed["abstract_summary"] == model_semantics["narrative_summary"]
    assert str(refreshed["content"]).startswith("## Model-authored daily account")
    graph_update = synthesizer.apply_model_semantics(
        "2026-08-14", model_semantics, observed_at + timedelta(hours=2)
    )
    assert graph_update["projected"] == 1

    policy = store.observational_policy()
    assert policy["focus_terms"] == ["locate the microscope when grounded evidence changes"]
    assert policy["open_questions"] == ["Where is the microscope now?"]
    theme = store.entity_detail(theme_id)
    assert theme is not None
    assert theme["entity"]["metadata"]["abstraction_kind"] == "narrative_theme"
    graph = store.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}
    assert {
        "contains_narrative_episode",
        "leaves_open_question",
        "updates_world_model",
    } <= relations
    assert store.narrative_constitution()["revision"] == 1
    store.close()
