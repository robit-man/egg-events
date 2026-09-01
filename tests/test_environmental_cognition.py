import asyncio
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import EggConfig, EnvironmentalCognitionConfig
from egg_companion.core.environmental_cognition import (
    AdaptiveVisualNovelty,
    EnvironmentalNoveltyTracker,
)
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import (
    BoundingBox,
    Detection,
    EvidenceRef,
    Observation,
    PerceptualEvent,
)
from egg_companion.runtime import CompanionRuntime, _BackgroundVisionPreempted


def _observation(
    at: datetime,
    *,
    people: tuple[str, ...] = (),
    objects: tuple[str, ...] = (),
    semantic_labels: tuple[str, ...] = (),
) -> Observation:
    detections = [
        Detection(
            "person",
            0.9,
            BoundingBox(10 + index * 20, 10, 80 + index * 20, 180),
            {"identity_id": person_id, "frame_shape": [240, 320]},
        )
        for index, person_id in enumerate(people)
    ]
    detections.extend(
        Detection(
            "object",
            0.8,
            BoundingBox(20 + index * 20, 40, 60 + index * 20, 90),
            {"object_id": object_id, "frame_shape": [240, 320]},
        )
        for index, object_id in enumerate(objects)
    )
    return Observation(
        "camera-1", at, tuple(detections), semantic_labels=semantic_labels
    )


def _decision(**components):
    return [(object(), SimpleNamespace(components=components))]


def test_person_arrival_continuity_and_departure_are_structural_events() -> None:
    settings = EnvironmentalCognitionConfig(minimum_salience=0.1)
    tracker = EnvironmentalNoveltyTracker(settings)
    now = datetime.now(timezone.utc)

    assert tracker.observe(_observation(now), [], 0.0, 0.0) is None
    arrival = tracker.observe(
        _observation(now + timedelta(seconds=1), people=("person-1",)),
        _decision(prediction_error=0.8),
        1.0,
        1.0,
    )
    assert arrival is not None
    assert "person_presence_changed" in arrival.causes
    assert arrival.current_person_ids == ("person-1",)

    continuing = tracker.observe(
        _observation(now + timedelta(seconds=2), people=("person-1",)),
        [],
        0.0,
        2.0,
    )
    assert continuing is None

    departure = tracker.observe(
        _observation(now + timedelta(seconds=3)), [], 0.0, 3.0
    )
    assert departure is not None
    assert "person_presence_changed" in departure.causes
    assert departure.current_person_count == 0


def test_drastic_scene_constellation_change_is_admitted_without_people() -> None:
    tracker = EnvironmentalNoveltyTracker(
        EnvironmentalCognitionConfig(minimum_salience=0.1)
    )
    now = datetime.now(timezone.utc)
    tracker.observe(
        _observation(now, objects=("object-a",), semantic_labels=("desk",)),
        [],
        0.0,
        0.0,
    )

    changed = tracker.observe(
        _observation(
            now + timedelta(seconds=1),
            objects=("object-b",),
            semantic_labels=("open doorway",),
        ),
        _decision(prediction_error=0.9),
        0.9,
        1.0,
    )

    assert changed is not None
    assert "scene_constellation_changed" in changed.causes
    assert changed.current_person_count == 0


def test_repeated_person_transition_habituates_and_recovers() -> None:
    settings = EnvironmentalCognitionConfig(
        minimum_salience=0.01,
        habituation_half_life_seconds=100,
    )
    tracker = EnvironmentalNoveltyTracker(settings)
    now = datetime.now(timezone.utc)
    tracker.observe(_observation(now), [], 0.0, 0.0)
    first = tracker.observe(
        _observation(now, people=("person-1",)), [], 0.0, 1.0
    )
    tracker.observe(_observation(now), [], 0.0, 2.0)
    second = tracker.observe(
        _observation(now, people=("person-1",)), [], 0.0, 3.0
    )
    tracker.observe(_observation(now), [], 0.0, 1003.0)
    recovered = tracker.observe(
        _observation(now, people=("person-1",)), [], 0.0, 1004.0
    )

    assert first is not None and second is not None and recovered is not None
    assert second.salience < first.salience
    assert recovered.salience > second.salience


def test_stimulus_salience_falls_off_continuously() -> None:
    settings = EnvironmentalCognitionConfig(
        minimum_salience=0.01, salience_half_life_seconds=10
    )
    tracker = EnvironmentalNoveltyTracker(settings)
    now = datetime.now(timezone.utc)
    stimulus = tracker.observe(
        _observation(now, people=("person-1",)), [], 0.0, 10.0
    )

    assert stimulus is not None
    assert stimulus.decayed_salience(20.0, 10.0) == pytest.approx(0.5)
    assert stimulus.decayed_salience(40.0, 10.0) == pytest.approx(0.125)


def test_adaptive_raw_frame_novelty_wakes_sparse_perception_only_on_change() -> None:
    settings = EnvironmentalCognitionConfig(
        raw_novelty_minimum=0.02,
        raw_probe_min_interval_seconds=4,
        raw_reference_blend=0.05,
    )
    monitor = AdaptiveVisualNovelty(settings)
    base = np.zeros((120, 160, 3), dtype=np.uint8)
    changed = base.copy()
    changed[:, 50:] = 255

    _signal, initial_wake, _threshold = monitor.observe("camera-1", base, 0.0)
    _signal, static_wake, _threshold = monitor.observe("camera-1", base, 5.0)
    change_signal, change_wake, _threshold = monitor.observe(
        "camera-1", changed, 10.0
    )
    _signal, refractory_wake, _threshold = monitor.observe(
        "camera-1", base, 11.0
    )

    assert initial_wake
    assert not static_wake
    assert change_wake
    assert change_signal == 1.0
    assert not refractory_wake


def test_environmental_model_contracts_allow_silence_or_speech_without_fallback() -> None:
    assessment_json = """{"grounded":true,"confidence":0.8,"scene_summary":"A person is near the table.","people_visible":true,"person_continuity":"One visible person remains in view.","meaningful_change":"The person entered the camera view.","change_magnitude":0.8,"addressability":"They are visible but not necessarily addressing Egg.","prior_query_answer":null,"camera_observations":[{"camera_id":"camera-1","scene_summary":"One person is near a table.","scene_tags":["person","table"],"subjects":[{"local_id":"person-1","prior_local_id":null,"detector_support":[],"kind":"person","label":"person","visible_behavior":"standing near a table","behavior_confidence":0.7,"confidence":0.8,"tags":["standing"],"evidence":"A full person silhouette is visible beside the table."}],"relations":[],"uncertainties":["The person's intent is not visible."]}],"overall_uncertainties":["The person's intent is unknown."],"memory_query":"recent interactions connected with the visible person and table","next_visual_query":"Whether the person remains by the table"}"""
    assessment = OmniusClient.parse_environmental_assessment(assessment_json)
    contradictory_summary = OmniusClient.parse_environmental_assessment(
        assessment_json.replace('"people_visible":true', '"people_visible":false')
    )
    silent = OmniusClient.parse_environmental_deliberation(
        """{"action":"silence","utterance":null,"reflection":"A person entered, but there is no evidence that interruption would help.","confidence":0.8,"reason":"Presence alone does not warrant speech.","connections":[],"open_questions":["Whether the person will address Egg."]}"""
    )
    spoken = OmniusClient.parse_environmental_deliberation(
        """{"action":"speak","utterance":"I remember we left that setup unfinished.","reflection":"The visible setup may connect to a recent unfinished task, though the connection is revisable.","confidence":0.7,"reason":"A brief grounded reminder may be useful.","connections":["Recent unfinished setup."],"open_questions":[]}"""
    )

    assert assessment is not None and assessment["people_visible"] is True
    assert contradictory_summary is not None and contradictory_summary["people_visible"] is True
    assert silent is not None and silent["utterance"] is None
    assert spoken is not None and spoken["utterance"]
    assert OmniusClient.parse_environmental_deliberation(
        """{"action":"speak","utterance":null,"reflection":"Invalid.","confidence":0.7,"reason":"Missing speech.","connections":[],"open_questions":[]}"""
    ) is None


def test_environmental_assessment_tolerates_a_repeated_camera_tile() -> None:
    """A duplicated camera_id in camera_observations is a harmless model

    quirk (the same tile described twice), not evidence the assessment is
    broken. Regression test for a real production incident: this was
    causing a hard rejection of an otherwise-valid, complete assessment.
    """
    import json

    payload = {
        "grounded": True,
        "confidence": 0.8,
        "scene_summary": "A person is near the table.",
        "people_visible": True,
        "person_continuity": "One visible person remains in view.",
        "meaningful_change": "The person entered the camera view.",
        "change_magnitude": 0.8,
        "addressability": "They are visible but not necessarily addressing Egg.",
        "prior_query_answer": None,
        "camera_observations": [
            {
                "camera_id": "camera-1",
                "scene_summary": "One person is near a table.",
                "scene_tags": ["person", "table"],
                "subjects": [{
                    "local_id": "person-1", "prior_local_id": None, "detector_support": [],
                    "kind": "person", "label": "person",
                    "visible_behavior": "standing near a table", "behavior_confidence": 0.7,
                    "confidence": 0.8, "tags": ["standing"],
                    "evidence": "A full person silhouette is visible beside the table.",
                }],
                "relations": [],
                "uncertainties": ["The person's intent is not visible."],
            },
            {
                "camera_id": "camera-1",
                "scene_summary": "Repeated description of the same tile.",
                "scene_tags": ["person", "table"],
                "subjects": [],
                "relations": [],
                "uncertainties": ["Duplicate tile description."],
            },
        ],
        "overall_uncertainties": ["The person's intent is unknown."],
        "memory_query": "recent interactions connected with the visible person and table",
        "next_visual_query": "Whether the person remains by the table",
    }
    assessment = OmniusClient.parse_environmental_assessment(
        json.dumps(payload), camera_ids={"camera-1"}
    )
    assert assessment is not None
    assert len(assessment["camera_observations"]) == 1
    assert assessment["camera_observations"][0]["scene_summary"] == "One person is near a table."


def test_environmental_assessment_drops_relation_to_undeclared_subject() -> None:
    """A relation naming a background object that was never given its own

    subject entry (e.g. a person "near" a "monitor" that has no subject of
    its own) is a harmless omission, not evidence the assessment is broken.
    Regression test for a real production incident: this was causing a
    hard rejection of an otherwise-valid, complete assessment.
    """
    import json

    payload = {
        "grounded": True,
        "confidence": 0.8,
        "scene_summary": "A person is at a desk.",
        "people_visible": True,
        "person_continuity": "One visible person remains in view.",
        "meaningful_change": "The person entered the camera view.",
        "change_magnitude": 0.8,
        "addressability": "They are visible but not necessarily addressing Egg.",
        "prior_query_answer": None,
        "camera_observations": [
            {
                "camera_id": "camera-1",
                "scene_summary": "Person seated at a desk facing a monitor.",
                "scene_tags": ["person", "desk"],
                "subjects": [{
                    "local_id": "person-1", "prior_local_id": None, "detector_support": [],
                    "kind": "person", "label": "seated person",
                    "visible_behavior": "sitting", "behavior_confidence": 0.7,
                    "confidence": 0.8, "tags": ["seated"],
                    "evidence": "A person is visible seated at a desk.",
                }],
                "relations": [{
                    "source_local_id": "person-1",
                    "relation": "near",
                    "target_local_id": "monitor",
                    "confidence": 0.8,
                    "evidence": "Person faces the monitor on the desk.",
                }],
                "uncertainties": ["The person's intent is not visible."],
            },
        ],
        "overall_uncertainties": ["The person's intent is unknown."],
        "memory_query": "recent interactions connected with the visible person",
        "next_visual_query": "Whether the person remains at the desk",
    }
    assessment = OmniusClient.parse_environmental_assessment(
        json.dumps(payload), camera_ids={"camera-1"}
    )
    assert assessment is not None
    assert assessment["camera_observations"][0]["relations"] == []


def test_environmental_reflection_enters_retrieval_and_reflective_working_context(
    tmp_path,
) -> None:
    config = EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
            "environmental_cognition": {"reflection_characters": 1200},
        }
    )
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    reflection_id = "reflection:environmental:test"
    event = PerceptualEvent(
        "event:environmental:test",
        "environmental_reflection",
        now,
        "ornith-environmental-cognition",
        (
            EvidenceRef(
                "evidence:environmental:test",
                "vision",
                now,
                "ornith-environmental-reflection",
                "camera-1",
                quality=0.8,
                metadata={"model_id": "test"},
            ),
        ),
        (reflection_id, "person-1"),
        payload={
            "labels": ["environmental reflection"],
            "entities": [
                {
                    "id": "person-1",
                    "type": "person",
                    "label": "Known person",
                    "confidence": 1.0,
                },
                {
                    "id": reflection_id,
                    "type": "reflection",
                    "label": "The present arrangement may relate to an unfinished setup.",
                    "summary": "The present arrangement may relate to an unfinished setup.",
                    "confidence": 0.8,
                    "revisable": True,
                },
            ],
            "relations": [
                {
                    "source_id": reflection_id,
                    "relation": "reflects_on",
                    "target_id": "person-1",
                    "confidence": 0.8,
                }
            ],
            "skip_pairwise_co_observation": True,
            "environmental_reflection": {
                "action": "reflect",
                "reflection": "The present arrangement may relate to an unfinished setup.",
                "confidence": 0.8,
                "reason": "The connection is worth retaining without interrupting.",
                "connections": ["A recent unfinished setup."],
                "open_questions": ["Whether the arrangement is the same task."],
            },
        },
    )

    accepted, _closed = pipeline.ingest(event)
    document = store.entity_metadata(
        "cognitive-document:environmental-working-set"
    )
    context = pipeline.context_for(
        "unfinished setup arrangement", "one person and an arrangement are visible"
    )

    assert accepted
    assert document is not None
    assert "unfinished setup" in str(document["metadata"]["content"])
    assert "environmental-working-set" in context
    assert "unfinished setup" in context
    pipeline.close(now)


def _runtime_config() -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "environmental_cognition": {
                "minimum_salience": 0.01,
                "salience_half_life_seconds": 300,
            },
        }
    )


def _runtime_stimulus(runtime: CompanionRuntime):
    now = datetime.now(timezone.utc)
    observation = _observation(now, people=("person-1",))
    runtime._latest_observation = observation
    runtime._latest_observations = {"camera-1": observation}
    runtime._latest_frames = {
        "camera-1": (np.zeros((120, 160, 3), dtype=np.uint8), time.monotonic())
    }
    stimulus = runtime._environmental_novelty.observe(
        observation, _decision(prediction_error=0.9), 1.0, time.monotonic()
    )
    assert stimulus is not None
    return stimulus


def _grounded_assessment() -> dict[str, object]:
    return {
        "grounded": True,
        "confidence": 0.8,
        "scene_summary": "One person is visible near a table.",
        "people_visible": True,
        "person_continuity": "One visible person remains in view.",
        "meaningful_change": "A person entered the view.",
        "change_magnitude": 0.8,
        "addressability": "The person is visible; whether they welcome speech is uncertain.",
        "prior_query_answer": None,
        "camera_observations": [
            {
                "camera_id": "camera-1",
                "scene_summary": "One person is visible near a table.",
                "scene_tags": ["person", "table"],
                "subjects": [
                    {
                        "local_id": "person-1",
                        "prior_local_id": None,
                        "detector_support": [],
                        "kind": "person",
                        "label": "person",
                        "visible_behavior": "standing near a table",
                        "behavior_confidence": 0.7,
                        "confidence": 0.8,
                        "tags": ["standing"],
                        "evidence": "A person silhouette is visible beside the table.",
                    }
                ],
                "relations": [],
                "uncertainties": ["Their intent is not visible."],
            }
        ],
        "overall_uncertainties": ["Their intent is not visible."],
        "memory_query": "recent interactions connected with the visible person and table",
        "next_visual_query": "whether the person remains near the table",
    }


def test_runtime_model_can_choose_silence_while_reflection_is_retained() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_runtime_config())
        stimulus = _runtime_stimulus(runtime)
        spoken: list[str] = []
        retained: list[dict[str, object]] = []

        async def assess(*_args):
            return _grounded_assessment()

        async def deliberate(*_args):
            return {
                "action": "silence",
                "utterance": None,
                "reflection": "The arrival is visible, but interruption is not yet useful.",
                "confidence": 0.8,
                "reason": "Presence alone does not warrant speech.",
                "connections": [],
                "open_questions": ["Whether the person will address Egg."],
            }

        async def speak(text: str, expected_revision=None):
            spoken.append(text)
            return True

        async def retain(*args):
            retained.append(args[4])

        runtime._omnius.assess_environmental_change = assess  # type: ignore[method-assign]
        runtime._omnius.deliberate_environmental_response = deliberate  # type: ignore[method-assign]
        runtime._cognitive_context = lambda *_args: asyncio.sleep(0, result="memory")  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        runtime._queue_environmental_reflection_memory = retain  # type: ignore[method-assign]

        await runtime._ponder_environmental_stimulus(stimulus, stimulus.salience)
        await asyncio.sleep(0)

        state = runtime.telemetry.snapshot(runtime.config)["environmental_cognition"]
        assert spoken == []
        assert retained and retained[0]["action"] == "silence"
        assert state["state"] == "silent"

    asyncio.run(scenario())


def test_runtime_speaks_only_after_model_choice_and_current_person_recheck() -> None:
    async def scenario(person_still_present: bool) -> tuple[list[str], str]:
        runtime = CompanionRuntime(_runtime_config())
        stimulus = _runtime_stimulus(runtime)
        spoken: list[str] = []

        async def assess(*_args):
            return _grounded_assessment()

        async def deliberate(*_args):
            return {
                "action": "speak",
                "utterance": "I remember we left that setup unfinished.",
                "reflection": "The visible setup may connect to an unfinished task.",
                "confidence": 0.75,
                "reason": "A brief grounded reminder may be useful.",
                "connections": ["A recent unfinished setup."],
                "open_questions": [],
            }

        async def speak(text: str, expected_revision=None):
            spoken.append(text)
            return True

        async def retain(*_args):
            return None

        if not person_still_present:
            runtime._latest_observations = {
                "camera-1": _observation(datetime.now(timezone.utc))
            }
        runtime._omnius.assess_environmental_change = assess  # type: ignore[method-assign]
        runtime._omnius.deliberate_environmental_response = deliberate  # type: ignore[method-assign]
        runtime._cognitive_context = lambda *_args: asyncio.sleep(0, result="memory")  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        runtime._queue_environmental_reflection_memory = retain  # type: ignore[method-assign]

        await runtime._ponder_environmental_stimulus(stimulus, stimulus.salience)
        await asyncio.sleep(0)
        state = runtime.telemetry.snapshot(runtime.config)["environmental_cognition"]
        return spoken, str(state["state"])

    present_spoken, present_state = asyncio.run(scenario(True))
    absent_spoken, absent_state = asyncio.run(scenario(False))

    assert present_spoken == ["I remember we left that setup unfinished."]
    assert present_state == "spoken"
    assert absent_spoken == []
    assert absent_state == "suppressed"


def test_runtime_queue_coalesces_to_freshest_environmental_evidence() -> None:
    runtime = CompanionRuntime(_runtime_config())
    first = _runtime_stimulus(runtime)
    runtime._queue_environmental_stimulus(first)
    runtime._environmental_novelty.observe(
        _observation(datetime.now(timezone.utc)), [], 0.0, time.monotonic()
    )
    second = runtime._environmental_novelty.observe(
        _observation(datetime.now(timezone.utc), people=("person-2",)),
        _decision(prediction_error=0.9),
        1.0,
        time.monotonic(),
    )
    assert second is not None

    runtime._queue_environmental_stimulus(second)

    assert runtime._environmental_stimuli.qsize() == 1
    assert runtime._environmental_stimuli.get_nowait().stimulus_id == second.stimulus_id
    state = runtime.telemetry.snapshot(runtime.config)["environmental_cognition"]
    assert state["coalesced"] == 1


def test_environmental_subject_ids_require_explicit_detector_or_visual_continuity() -> None:
    runtime = CompanionRuntime(_runtime_config())
    stimulus = _runtime_stimulus(runtime)
    snapshot = runtime._capture_turn_visual_snapshot(
        stimulus.stimulus_id, time.monotonic()
    )
    ledger = runtime._environmental_detector_ledger(snapshot)
    assessment = _grounded_assessment()
    assessment["camera_observations"][0]["subjects"][0]["detector_support"] = [
        "detector:camera-1:0"
    ]

    first = runtime._materialize_environmental_assessment(
        assessment, stimulus, None, ledger
    )
    first_subject = first["camera_observations"][0]["subjects"][0]
    assert first_subject["entity_id"] == "person-1"
    assert first_subject["entity_source"] == "same_frame_detector_support"

    followup = _grounded_assessment()
    followup_subject = followup["camera_observations"][0]["subjects"][0]
    followup_subject["local_id"] = "person-current"
    followup_subject["prior_local_id"] = "person-1"
    continued = runtime._materialize_environmental_assessment(
        followup, stimulus, first, []
    )
    continued_subject = continued["camera_observations"][0]["subjects"][0]
    assert continued_subject["entity_id"] == "person-1"
    assert continued_subject["entity_source"] == "model_visual_continuity"


def test_camera_addressed_vlm_grounding_reaches_memory_and_world_model(tmp_path) -> None:
    async def scenario() -> None:
        payload = _runtime_config().model_dump()
        payload["memory"].update(
            {
                "enabled": True,
                "storage_dir": str(tmp_path / "memory"),
                "retain_raw_media": False,
            }
        )
        payload["identity"]["storage_dir"] = str(tmp_path / "identity")
        payload["object_learning"]["storage_dir"] = str(tmp_path / "objects")
        runtime = CompanionRuntime(EggConfig.model_validate(payload))
        assert runtime._memory is not None
        stimulus = _runtime_stimulus(runtime)
        snapshot = runtime._capture_turn_visual_snapshot(
            stimulus.stimulus_id, time.monotonic()
        )
        ledger = runtime._environmental_detector_ledger(snapshot)
        assessment = _grounded_assessment()
        assessment["camera_observations"][0]["subjects"][0]["detector_support"] = [
            "detector:camera-1:0"
        ]
        assessment = runtime._materialize_environmental_assessment(
            assessment, stimulus, None, ledger
        )
        encoded = [
            runtime._encode_visual_question_frame(item.frame)
            for item in snapshot.frames
        ]

        await runtime._queue_environmental_grounding_memory(
            stimulus, snapshot, encoded, assessment
        )
        event = runtime._memory_events.get_nowait()
        assert event.event_type == "vlm_observation"
        assert event.source_id == "ornith_vlm:camera-1"
        assert event.payload["complete_camera_frame"] is False
        runtime._memory._persist_event(event)

        assert runtime._memory.store.entity_detail("camera_view:camera-1") is not None
        assert runtime._memory.store.entity_detail("person-1") is not None
        assert runtime._memory._world_query is not None
        assert (
            runtime._memory._world_query.property_value("person-1", "behavior")
            == "standing near a table"
        )
        assert (
            runtime._memory._world_query.property_value(
                "camera_view:camera-1", "scene_summary"
            )
            == "One person is visible near a table."
        )
        runtime._memory.close(datetime.now(timezone.utc))

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_memory_recall_back_into_reply(tmp_path) -> None:
    async def scenario() -> None:
        payload = _runtime_config().model_dump()
        payload["memory"].update(
            {
                "enabled": True,
                "storage_dir": str(tmp_path / "memory"),
                "retain_raw_media": False,
            }
        )
        payload["identity"]["storage_dir"] = str(tmp_path / "identity")
        payload["object_learning"]["storage_dir"] = str(tmp_path / "objects")
        runtime = CompanionRuntime(EggConfig.model_validate(payload))
        assert runtime._memory is not None
        assert runtime._memory.world_query is not None
        contexts = []
        spoken = []

        async def conversation(
            utterance, context, history, *, allow_tool_requests=True, on_delta=None
        ):
            contexts.append((context, allow_tool_requests))
            if "OBJECT MEMORY TOOL RESULT" not in context:
                return "[[TOOL:MEMORY|my keys]]"
            assert "OBJECT MEMORY TOOL RESULT" in context
            return "You left your keys by camera-video1 a few minutes ago."

        def recall(
            query: str,
            limit: int = 5,
            history_per_entity: int = 3,
            since: str | None = None,
            until: str | None = None,
        ):
            assert query == "my keys"
            return [{
                "entity_id": "object-001",
                "label": "keys",
                "matched_property": "label",
                "sightings": [
                    {
                        "camera_id": "camera-video1",
                        "seen_at": "2026-08-30T00:00:00+00:00",
                        "confidence": 0.9,
                    }
                ],
            }]

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._memory.world_query.recall_object_sightings = recall  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Where did you last see my keys?",
            utterance_id="heard-semantic-memory",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["You left your keys by camera-video1 a few minutes ago."]
        runtime._memory.close(datetime.now(timezone.utc))

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_past_ocr_back_into_reply(tmp_path) -> None:
    async def scenario() -> None:
        payload = _runtime_config().model_dump()
        payload["memory"].update(
            {
                "enabled": True,
                "storage_dir": str(tmp_path / "memory"),
                "retain_raw_media": False,
            }
        )
        payload["identity"]["storage_dir"] = str(tmp_path / "identity")
        payload["object_learning"]["storage_dir"] = str(tmp_path / "objects")
        runtime = CompanionRuntime(EggConfig.model_validate(payload))
        assert runtime._memory is not None
        contexts = []
        spoken = []

        async def conversation(
            utterance, context, history, *, allow_tool_requests=True, on_delta=None
        ):
            contexts.append((context, allow_tool_requests))
            if "PAST CAMERA TEXT TOOL RESULT" not in context:
                return "[[TOOL:PAST_OCR|the sign by the door]]"
            assert "PAST CAMERA TEXT TOOL RESULT" in context
            return "It said ROOM 204."

        def recall(
            query: str,
            limit: int = 5,
            history_per_entity: int = 3,
            since: str | None = None,
            until: str | None = None,
        ):
            assert query == "the sign by the door"
            return [
                {
                    "entity_id": "object-002",
                    "label": "sign",
                    "matched_property": "label",
                    "sightings": [
                        {
                            "camera_id": "camera-video1",
                            "seen_at": "2026-08-30T00:00:00+00:00",
                            "confidence": 0.9,
                            "evidence_id": "ev:sign-1",
                        }
                    ],
                }
            ]

        def evidence_media(evidence_id: str):
            assert evidence_id == "ev:sign-1"
            return (b"fake-jpeg-bytes", "image/jpeg")

        async def run_advanced_ocr(image_png: bytes, **kwargs):
            assert image_png == b"fake-jpeg-bytes"
            return {"text": "ROOM 204", "confidence": 0.9, "engine": "local"}

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._memory.world_query.recall_object_sightings = recall  # type: ignore[method-assign]
        runtime.evidence_media = evidence_media  # type: ignore[method-assign]
        runtime._run_advanced_ocr = run_advanced_ocr  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "What did the sign by the door say?",
            utterance_id="heard-semantic-past-ocr",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["It said ROOM 204."]
        runtime._memory.close(datetime.now(timezone.utc))

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_past_ocr_no_evidence_status(tmp_path) -> None:
    async def scenario() -> None:
        payload = _runtime_config().model_dump()
        payload["memory"].update(
            {
                "enabled": True,
                "storage_dir": str(tmp_path / "memory"),
                "retain_raw_media": False,
            }
        )
        payload["identity"]["storage_dir"] = str(tmp_path / "identity")
        payload["object_learning"]["storage_dir"] = str(tmp_path / "objects")
        runtime = CompanionRuntime(EggConfig.model_validate(payload))
        assert runtime._memory is not None
        contexts = []
        spoken = []

        async def conversation(
            utterance, context, history, *, allow_tool_requests=True, on_delta=None
        ):
            contexts.append((context, allow_tool_requests))
            if "PAST CAMERA TEXT TOOL RESULT" not in context:
                return "[[TOOL:PAST_OCR|the sign by the door]]"
            assert "PAST CAMERA TEXT TOOL RESULT" in context
            return "I don't have a stored image of that to read."

        def recall(
            query: str,
            limit: int = 5,
            history_per_entity: int = 3,
            since: str | None = None,
            until: str | None = None,
        ):
            return [
                {
                    "entity_id": "object-002",
                    "label": "sign",
                    "matched_property": "label",
                    "sightings": [
                        {
                            "camera_id": "camera-video1",
                            "seen_at": "2026-08-30T00:00:00+00:00",
                            "confidence": 0.9,
                            "evidence_id": None,
                        }
                    ],
                }
            ]

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._memory.world_query.recall_object_sightings = recall  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "What did the sign by the door say?",
            utterance_id="heard-semantic-past-ocr-none",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["I don't have a stored image of that to read."]
        runtime._memory.close(datetime.now(timezone.utc))

    asyncio.run(scenario())


def test_associative_object_recall_uses_embedding_similarity_and_caches_labels() -> None:
    runtime = CompanionRuntime(_runtime_config())
    embed_calls: list[str] = []

    class FakeVision:
        def embed_text(self, text: str) -> np.ndarray:
            embed_calls.append(text)
            vectors = {
                "the drink": np.array([1.0, 0.0], dtype=np.float32),
                "red mug": np.array([0.9938, 0.1104], dtype=np.float32),
                "umbrella": np.array([0.0, 1.0], dtype=np.float32),
            }
            return vectors[text]

    class FakeWorldQuery:
        def __init__(self):
            self.bounds_seen: list[tuple[str | None, str | None]] = []

        def candidate_labels(self):
            return [
                {"entity_id": "object-1", "label": "red mug"},
                {"entity_id": "object-2", "label": "umbrella"},
            ]

        def sightings_for_entity(self, entity_id, history_per_entity, since, until):
            self.bounds_seen.append((since, until))
            return {
                "entity_id": entity_id,
                "sightings": [{"camera_id": "cam0", "seen_at": "x", "confidence": 0.5}],
            }

    runtime._vision = FakeVision()  # type: ignore[assignment]
    world_query = FakeWorldQuery()

    results = runtime._associative_object_recall(
        world_query, "the drink", 5, "2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00"
    )
    assert [r["entity_id"] for r in results] == ["object-1"]
    assert results[0]["matched_property"] == "embedding"
    assert embed_calls == ["the drink", "red mug", "umbrella"]
    assert world_query.bounds_seen == [
        ("2020-01-01T00:00:00+00:00", "2021-01-01T00:00:00+00:00")
    ]

    embed_calls.clear()
    runtime._associative_object_recall(world_query, "the drink", 5, None, None)
    assert embed_calls == ["the drink"]


def test_human_speech_preempts_environmental_model_work_without_polling() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_runtime_config())
        entered = asyncio.Event()
        never = asyncio.Event()

        async def background():
            entered.set()
            await never.wait()

        task = asyncio.create_task(runtime._run_background_visual(background()))
        await entered.wait()
        await runtime._on_utterance_started(time.monotonic())
        result = await asyncio.gather(task, return_exceptions=True)

        assert isinstance(result[0], _BackgroundVisionPreempted)
        assert not runtime._environmental_foreground_idle.is_set()
        assert not runtime._background_visual_tasks

    asyncio.run(scenario())


def test_newer_room_revision_discards_stale_grounding_before_deliberation() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_runtime_config())
        stimulus = _runtime_stimulus(runtime)
        deliberations = 0

        async def assess(*_args):
            return _grounded_assessment()

        async def deliberate(*_args):
            nonlocal deliberations
            deliberations += 1

        runtime._omnius.assess_environmental_change = assess  # type: ignore[method-assign]
        runtime._omnius.deliberate_environmental_response = deliberate  # type: ignore[method-assign]
        newer = runtime._environmental_novelty.observe(
            _observation(datetime.now(timezone.utc)),
            [],
            0.0,
            time.monotonic(),
        )
        assert newer is not None
        runtime._environmental_stimuli.put_nowait(newer)

        await runtime._ponder_environmental_stimulus(stimulus, stimulus.salience)

        state = runtime.telemetry.snapshot(runtime.config)["environmental_cognition"]
        assert deliberations == 0
        assert state["state"] == "stale"

    asyncio.run(scenario())


def test_preempted_candidate_yields_immediately_to_newer_room_event() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_runtime_config())
        older = _runtime_stimulus(runtime)
        newer = replace(
            older,
            stimulus_id="environment:newer",
            sequence=older.sequence + 1,
            observed_monotonic=time.monotonic(),
        )
        seen: list[str] = []
        completed = asyncio.Event()

        async def ponder(stimulus, _salience):
            seen.append(stimulus.stimulus_id)
            if stimulus.stimulus_id == older.stimulus_id:
                runtime._environmental_stimuli.put_nowait(newer)
                raise _BackgroundVisionPreempted
            completed.set()

        runtime._ponder_environmental_stimulus = ponder  # type: ignore[method-assign]
        runtime._environmental_stimuli.put_nowait(older)
        worker = asyncio.create_task(runtime._process_environmental_cognition())
        try:
            await asyncio.wait_for(completed.wait(), timeout=1)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        state = runtime.telemetry.snapshot(runtime.config)["environmental_cognition"]
        assert seen == [older.stimulus_id, newer.stimulus_id]
        assert state["preempted"] == 1
        assert state["stale"] == 1

    asyncio.run(scenario())


def test_background_timeout_releases_candidate_without_runtime_error() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_runtime_config())
        stimulus = _runtime_stimulus(runtime)
        attempted = asyncio.Event()

        async def ponder(_stimulus, _salience):
            attempted.set()
            raise asyncio.TimeoutError

        runtime._ponder_environmental_stimulus = ponder  # type: ignore[method-assign]
        runtime._environmental_stimuli.put_nowait(stimulus)
        worker = asyncio.create_task(runtime._process_environmental_cognition())
        try:
            await asyncio.wait_for(attempted.wait(), timeout=1)
            await asyncio.sleep(0)
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)

        snapshot = runtime.telemetry.snapshot(runtime.config)
        state = snapshot["environmental_cognition"]
        assert state["timed_out"] == 1
        assert state["errors"] == 0
        assert not snapshot["runtime_errors"]

    asyncio.run(scenario())
