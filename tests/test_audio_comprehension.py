from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.runtime import (
    CompanionRuntime,
    _AudioComprehensionJob,
    _SpeechSegment,
)


def _config(tmp_path=None, *, memory: bool = False) -> EggConfig:
    payload = {
        "audio": {"input_device": "default", "doa_mode": "disabled"},
        "audio_comprehension": {"minimum_interval_seconds": 0, "queue_size": 1},
        "omnius": {"model": "test", "voice_model": "test"},
        "identity": {"enabled": False},
        "object_learning": {"enabled": False},
        "camera_discovery": {"enabled": False},
        "memory": {"enabled": memory},
    }
    if memory:
        payload["memory"]["storage_dir"] = str(tmp_path / "memory")
    return EggConfig.model_validate(payload)


def test_audio_comprehension_queue_coalesces_without_backpressuring_asr() -> None:
    runtime = CompanionRuntime(_config())
    first = _SpeechSegment("first", b"first", 1, 2, None, {}, {})
    second = _SpeechSegment("second", b"second", 2, 3, None, {}, {})

    runtime._queue_audio_comprehension("one", first)
    runtime._queue_audio_comprehension("two", second)

    assert runtime._audio_comprehension_jobs.qsize() == 1
    assert runtime._audio_comprehension_jobs.get_nowait().context_id == "second"
    state = runtime.telemetry.snapshot(runtime.config)["audio_comprehension"]
    assert state["queued"] == 2
    assert state["coalesced"] == 1


def test_grounded_sound_event_is_linked_to_visible_context(tmp_path) -> None:
    runtime = CompanionRuntime(_config(tmp_path, memory=True))
    media_key, media_checksum = runtime._memory.persist_media(
        "audio/test/utterance-1.wav", b"retained-audio-evidence"
    )
    job = _AudioComprehensionJob(
        context_id="utterance-1",
        audio=b"unused",
        transcript="what is that sound?",
        captured_at=datetime.now(timezone.utc),
        entities=(
            {
                "id": "person-1",
                "type": "person",
                "label": "Troy",
                "confidence": 0.95,
                "source": "visible-during-audio",
            },
        ),
        media_key=media_key,
        media_checksum=media_checksum,
    )

    runtime._queue_audio_comprehension_memory(
        job,
        {
            "classifications": [{"label": "Speech", "confidence": 0.67}],
            "model": "google/yamnet/1",
            "taxonomy": "AudioSet",
            "semantic_quality": "grounded classifier",
            "acoustic": {"duration": 6.0, "wav_rms": 0.2},
        },
    )
    event = runtime._memory_events.get_nowait()
    assert event.event_type == "audio_comprehension"
    assert event.evidence[0].modality == "audio_semantics"
    assert event.evidence[0].metadata["mock_evidence_discarded"] is True
    assert event.evidence[0].media_key == media_key
    assert event.payload["relations"][0]["relation"] == "heard_with"

    accepted, _ = runtime._memory.ingest(event)
    graph = runtime._memory.knowledge_graph_snapshot()
    sound_id = str(event.payload["entities"][0]["id"])
    assert accepted
    assert any(node["source_id"] == sound_id for node in graph["nodes"])
    assert any(link["relation"] == "heard_with" for link in graph["links"])
    runtime._memory.store.close()
