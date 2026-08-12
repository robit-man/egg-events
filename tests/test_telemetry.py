from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.models import BoundingBox, Detection, Observation
from egg_companion.services.telemetry import RuntimeTelemetry


def make_config() -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default"},
            "omnius": {"model": "test", "voice_model": "test"},
        }
    )


def test_asr_telemetry_distinguishes_acceptance_rejection_and_errors() -> None:
    config = make_config()
    telemetry = RuntimeTelemetry(config)
    metadata = {"duration": 2.5, "accepted": True}

    telemetry.record_transcript("hello egg", metadata)
    telemetry.record_asr_rejection("high no-speech probability", {"accepted": False})
    telemetry.record_asr_error(TimeoutError())
    snapshot = telemetry.snapshot(config)

    assert snapshot["latest_transcript"] == "hello egg"
    assert snapshot["latest_transcript_at"]
    assert snapshot["transcript_count"] == 1
    assert snapshot["transcript_history"][-1]["metadata"] == metadata
    assert snapshot["asr"]["accepted"] == 1
    assert snapshot["asr"]["rejected"] == 1
    assert snapshot["asr"]["errors"] == 1
    assert snapshot["runtime_errors"][-1]["detail"] == "TimeoutError"


def test_object_learning_telemetry_exposes_each_pipeline_boundary() -> None:
    config = make_config()
    telemetry = RuntimeTelemetry(config)

    for stage in (
        "stable_candidate",
        "duplicate_candidate",
        "clip_query",
        "clip_recall",
        "vlm_request",
        "vlm_success",
        "vlm_rejection",
        "vlm_error",
        "speech_deferral",
    ):
        telemetry.record_object_learning(stage, "test")
    learning = telemetry.snapshot(config)["object_learning"]

    assert learning["stable_candidates"] == 1
    assert learning["duplicate_candidates"] == 1
    assert learning["clip_queries"] == 1
    assert learning["clip_recalls"] == 1
    assert learning["vlm_requests"] == 1
    assert learning["vlm_successes"] == 1
    assert learning["vlm_rejections"] == 1
    assert learning["vlm_errors"] == 1
    assert learning["speech_deferrals"] == 1
    assert learning["last_stage"] == "speech_deferral"


def test_tool_and_identity_dialogue_are_observable() -> None:
    config = make_config()
    telemetry = RuntimeTelemetry(config)

    telemetry.record_tool_call(
        "web_search", "latest news", True, "one result", 12.34,
        context_id="utterance-1",
    )
    telemetry.record_audio_comprehension(
        "completed",
        context_id="utterance-1",
        classifications=[{"label": "Speech", "confidence": 0.67}],
        duration_ms=812.4,
    )
    telemetry.record_identity_dialogue("awaiting_name", "person-1", "front")
    snapshot = telemetry.snapshot(config)

    assert snapshot["tool_calls"][-1]["name"] == "web_search"
    assert snapshot["tool_calls"][-1]["duration_ms"] == 12.3
    assert snapshot["tool_calls"][-1]["context_id"] == "utterance-1"
    assert snapshot["audio_comprehension"]["completed"] == 1
    assert snapshot["audio_comprehension"]["classifications"][0]["label"] == "Speech"
    assert snapshot["identity_dialogue"]["state"] == "awaiting_name"
    assert snapshot["identity_dialogue"]["profile_id"] == "person-1"

    telemetry.record_tool_call(
        "fresh_vision", "what is this", None, "tool invocation in progress", 0,
        context_id="utterance-2",
    )
    assert telemetry.snapshot(config)["tool_calls"][-1]["status"] == "running"
    telemetry.record_tool_call(
        "fresh_vision", "what is this", True, "a mug", 201.2,
        context_id="utterance-2",
    )
    updated = telemetry.snapshot(config)["tool_calls"][-1]
    assert updated["status"] == "completed"
    assert updated["success"] is True


def test_graph_activations_are_causal_bounded_and_use_real_graph_ids() -> None:
    config = make_config()
    telemetry = RuntimeTelemetry(config)

    telemetry.record_graph_activation(
        "voice",
        ["episode:turn-1", "evidence:audio-1", "bogus", "entity:"],
        origin_node_ids=["evidence:audio-1"],
        intensity=4.0,
        detail="what am I holding?",
    )
    snapshot = telemetry.graph_activation_snapshot()

    assert snapshot["sequence"] == 1
    assert snapshot["events"][-1]["node_ids"] == [
        "episode:turn-1",
        "evidence:audio-1",
    ]
    assert snapshot["events"][-1]["origin_node_ids"] == ["evidence:audio-1"]
    assert snapshot["events"][-1]["intensity"] == 1.0
    assert telemetry.snapshot(config)["graph_activations"] == snapshot


def test_inference_updates_never_rewrite_raw_camera_stream() -> None:
    config = EggConfig.model_validate(
        {
            "cameras": [{"id": "camera-0", "source": "/dev/video0", "rotation_degrees": 90}],
            "audio": {"input_device": "default"},
            "omnius": {"model": "test", "voice_model": "test"},
        }
    )
    telemetry = RuntimeTelemetry(config)
    telemetry.record_frame("camera-0", b"raw-frame", (1920, 1080, 3), 8.0)
    observation = Observation(
        "camera-0",
        datetime.now(timezone.utc),
        (
            Detection(
                "mug",
                0.8,
                BoundingBox(10, 20, 100, 200),
                {"mask_polygon": [[10, 20], [100, 20], [100, 200], [10, 200]]},
            ),
        ),
    )

    telemetry.record_observation(observation)
    camera = telemetry.snapshot(config)["cameras"][0]

    assert telemetry.frame("camera-0") == b"raw-frame"
    assert camera["frame_sequence"] == 1
    assert camera["detection_sequence"] == 1
    assert camera["detections"][0]["mask_polygon"]
