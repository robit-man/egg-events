"""Contract and fallback tests for the Qwen Omni adapter integration.

The adapter is an enhancement layered over Egg's existing Omnius perception
paths, so these tests care about two things above all: that speech and
non-speech acoustic evidence stay separated, and that every route degrades to
the Omnius path instead of losing a capability.
"""

from __future__ import annotations

import asyncio
import io
import wave

import numpy as np
import pytest

from egg_companion.adapters.omni import (
    ADAPTER_SCHEMA,
    OmniAdapterClient,
    OmniAdapterError,
    OmniAdapterUnavailable,
    normalize_input_wav,
)
from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import EggConfig, OmniAdapterConfig, OmniusConfig


def _wav(
    *, sample_rate: int = 16000, channels: int = 1, width: int = 2, seconds: float = 1.0
) -> bytes:
    frames = int(sample_rate * seconds)
    tone = np.sin(
        2 * np.pi * 220 * np.arange(frames * channels) / sample_rate
    ) * 0.4
    if width == 2:
        payload = (tone * 32767).astype("<i2").tobytes()
    else:
        payload = ((tone * 127) + 128).astype(np.uint8).tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(sample_rate)
        target.writeframes(payload)
    return buffer.getvalue()


def _client(**overrides) -> OmniAdapterClient:
    config = OmniAdapterConfig(mode="omni", **overrides)
    client = OmniAdapterClient(config)
    # Skip the health probe; each test drives _post directly.
    client._healthy_until = float("inf")
    return client


def _adapter_response(
    *,
    content: str = "",
    observation: str | None = None,
    transcript: str | None = None,
    audio_observation: str | None = None,
    audio: dict[str, str] | None = None,
) -> dict[str, object]:
    metadata: dict[str, object] = {"schema": ADAPTER_SCHEMA, "route": ["comprehension"]}
    if observation is not None:
        metadata["observation"] = observation
    if transcript is not None:
        metadata["input_transcript"] = transcript
    if audio_observation is not None:
        metadata["audio_observation"] = audio_observation
    message: dict[str, object] = {"role": "assistant", "content": content}
    if audio is not None:
        message["audio"] = audio
    return {"message": message, "adapter": metadata}


# -- wire contract -------------------------------------------------------


def test_perception_requests_are_comprehension_only_and_never_speak() -> None:
    """A perception pass must not run the language model or synthesize audio."""

    client = _client()
    sent: list[dict[str, object]] = []

    async def capture(payload, *, timeout_seconds):
        sent.append(payload)
        return _adapter_response(
            observation="<speech_transcript>turn the light on</speech_transcript>"
            "<audio_observation>a fan hums steadily</audio_observation>"
        )

    client._post = capture
    asyncio.run(client.perceive_audio(_wav()))

    payload = sent[0]
    assert payload["omni"] == {
        "schema": ADAPTER_SCHEMA,
        "task": "describe",
        "include_audio_from_video": True,
    }
    assert payload["stream"] is False
    assert payload["speech_mode"] == "never"
    assert payload["response_modalities"] == ["text"]
    assert payload["think"] is False
    assert "speech" not in payload
    assert payload["messages"][0]["audios"][0]["mime_type"] == "audio/wav"


def test_perception_separates_speech_from_non_speech_evidence() -> None:
    client = _client()

    async def respond(payload, *, timeout_seconds):
        return _adapter_response(
            observation=(
                "<speech_transcript>put the kettle on</speech_transcript>"
                "<audio_observation>a door closes, then a dog barks twice"
                "</audio_observation>"
            )
        )

    client._post = respond
    result = asyncio.run(client.perceive_audio(_wav()))

    assert result["transcript"] == "put the kettle on"
    assert result["audio_observation"] == "a door closes, then a dog barks twice"


def test_server_parsed_tags_take_priority_over_local_parsing() -> None:
    client = _client()

    async def respond(payload, *, timeout_seconds):
        return _adapter_response(
            observation="<speech_transcript>ignored</speech_transcript>",
            transcript="authoritative",
            audio_observation="rain on a window",
        )

    client._post = respond
    result = asyncio.run(client.perceive_audio(_wav()))

    assert result["transcript"] == "authoritative"
    assert result["audio_observation"] == "rain on a window"


def test_untagged_output_is_acoustic_evidence_not_a_transcript() -> None:
    """Never attribute an unlabeled description to a speaker."""

    client = _client()

    async def respond(payload, *, timeout_seconds):
        return _adapter_response(observation="Traffic passes outside the window.")

    client._post = respond
    result = asyncio.run(client.perceive_audio(_wav()))

    assert result["transcript"] is None
    assert result["audio_observation"] == "Traffic passes outside the window."


def test_sound_only_audio_yields_no_transcript() -> None:
    client = _client()

    async def respond(payload, *, timeout_seconds):
        return _adapter_response(
            observation=(
                "<speech_transcript></speech_transcript>"
                "<audio_observation>a siren passes</audio_observation>"
            )
        )

    client._post = respond
    result = asyncio.run(client.perceive_audio(_wav()))

    assert result["transcript"] is None
    assert result["audio_observation"] == "a siren passes"


def test_synthesis_requests_speech_and_returns_a_wav() -> None:
    import base64

    client = _client(
        speech_enabled=True,
        voice="F4",
        voice_language="en",
        voice_profile_path="/nonexistent/voice-profile.json",
    )
    sent: list[dict[str, object]] = []
    wav = _wav(sample_rate=24000)

    async def respond(payload, *, timeout_seconds):
        sent.append(payload)
        return _adapter_response(
            content="hello",
            audio={"data": base64.b64encode(wav).decode("ascii")},
        )

    client._post = respond
    assert asyncio.run(client.synthesize("hello")) == wav

    payload = sent[0]
    assert payload["omni"]["task"] == "synthesize"
    assert payload["speech_mode"] == "always"
    assert payload["response_modalities"] == ["text", "audio"]
    assert payload["speech"] == {"voice": "F4", "language": "en"}


def test_synthesis_rejects_a_response_without_audio() -> None:
    client = _client(speech_enabled=True)

    async def respond(payload, *, timeout_seconds):
        return _adapter_response(content="hello")

    client._post = respond
    with pytest.raises(OmniAdapterError):
        asyncio.run(client.synthesize("hello"))


def test_video_requests_carry_bounded_sampling() -> None:
    client = _client(video_fps=1.5, video_max_frames=12, video_include_audio=False)
    sent: list[dict[str, object]] = []

    async def respond(payload, *, timeout_seconds):
        sent.append(payload)
        return _adapter_response(
            observation="<visual_observation>A person waves.</visual_observation>"
        )

    client._post = respond
    # A minimal MP4 signature is enough: encoding is local, validation is remote.
    result = asyncio.run(client.describe_video(b"\x00\x00\x00\x18ftypmp42"))

    sampling = sent[0]["messages"][0]["videos"][0]["sampling"]
    assert sampling == {"fps": 1.5, "max_frames": 12, "include_audio": False}
    assert sent[0]["omni"]["include_audio_from_video"] is False
    assert result["visual_observation"] == "A person waves."


# -- audio normalization -------------------------------------------------


def test_contract_audio_passes_through_untouched() -> None:
    audio = _wav()
    assert normalize_input_wav(audio) is audio


def test_off_contract_audio_is_converted_to_16k_mono_pcm16() -> None:
    converted = normalize_input_wav(_wav(sample_rate=48000, channels=2, seconds=0.5))

    with wave.open(io.BytesIO(converted), "rb") as source:
        assert source.getframerate() == 16000
        assert source.getnchannels() == 1
        assert source.getsampwidth() == 2
        assert source.getnframes() == pytest.approx(8000, abs=2)


def test_a_non_wav_payload_is_refused_before_leaving_the_device() -> None:
    with pytest.raises(OmniAdapterError):
        normalize_input_wav(b"not a wav at all")


# -- health gating -------------------------------------------------------


def test_traditional_mode_is_unavailable_without_any_request() -> None:
    client = OmniAdapterClient(OmniAdapterConfig(mode="traditional"))

    with pytest.raises(OmniAdapterUnavailable):
        asyncio.run(client._ensure_available())


def test_a_failure_parks_the_adapter_for_its_cooldown() -> None:
    """A stopped adapter must cost one timeout, not one per spoken turn."""

    client = OmniAdapterClient(OmniAdapterConfig(mode="omni"))
    probes = 0

    async def failing_health():
        nonlocal probes
        probes += 1
        raise OSError("connection refused")

    client.health = failing_health
    for _ in range(3):
        with pytest.raises(OmniAdapterUnavailable):
            asyncio.run(client._ensure_available())

    assert probes == 1
    assert client.status()["cooling_down"] is True
    assert client.status()["healthy"] is False


# -- OmniusClient routing and fallback -----------------------------------


def _omnius(omni: OmniAdapterClient | None) -> OmniusClient:
    return OmniusClient(OmniusConfig(model="test", voice_model="test"), omni)


def _speech_evidence() -> dict[str, object]:
    return {"speech_detected": True, "source_rms": 0.09, "minimum_rms": 0.01}


def test_transcription_prefers_the_adapter_and_records_its_backend() -> None:
    adapter = _client()

    async def perceive(wav_audio):
        return {
            "transcript": "open the blinds please",
            "audio_observation": "a clock ticks",
            "model": "robit/ornith-1.5-omni:q4km",
            "backend": "qwen3-omni",
        }

    adapter.perceive_audio = perceive
    client = _omnius(adapter)

    transcript = asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert transcript == "open the blinds please"
    assert client.last_transcription_metadata["backend"] == "qwen3-omni"
    assert client.last_transcription_metadata["accepted"] is True
    # Room sound is kept apart from what the user said.
    assert client.last_audio_observation["observation"] == "a clock ticks"


def test_an_unavailable_adapter_falls_back_to_the_omnius_asr_path() -> None:
    adapter = _client()

    async def perceive(wav_audio):
        raise OmniAdapterUnavailable("adapter is down")

    adapter.perceive_audio = perceive
    client = _omnius(adapter)
    calls: list[bytes] = []

    async def omnius_path(wav_audio, evidence, language):
        calls.append(wav_audio)
        return "fallback transcript"

    client._transcribe_via_omnius = omnius_path

    transcript = asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert transcript == "fallback transcript"
    assert len(calls) == 1


def test_the_adapter_transcript_still_faces_the_grounding_gate() -> None:
    """The quality bar for what Egg acts on must not depend on the backend."""

    adapter = _client()

    async def perceive(wav_audio):
        return {
            "transcript": "Allah Allah Allah Allah Allah Allah Allah Allah",
            "audio_observation": None,
            "model": "robit/ornith-1.5-omni:q4km",
            "backend": "qwen3-omni",
        }

    adapter.perceive_audio = perceive
    client = _omnius(adapter)

    transcript = asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert transcript is None
    assert client.last_transcription_metadata["accepted"] is False
    assert client.last_transcription_metadata["rejection_reason"] is not None


def test_a_stale_audio_observation_never_survives_the_next_turn() -> None:
    adapter = _client()
    client = _omnius(adapter)
    client.last_audio_observation = {"observation": "a previous siren"}

    async def perceive(wav_audio):
        raise OmniAdapterUnavailable("adapter is down")

    adapter.perceive_audio = perceive

    async def omnius_path(wav_audio, evidence, language):
        return "fresh transcript"

    client._transcribe_via_omnius = omnius_path
    asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert client.last_audio_observation == {}


def test_speech_routes_to_the_adapter_only_when_it_is_enabled() -> None:
    adapter = _client(speech_enabled=False)
    calls: list[str] = []

    async def synthesize(text):
        calls.append(text)
        return b"RIFFomni"

    adapter.synthesize = synthesize
    client = _omnius(adapter)
    client._synthesize_via_omnius = lambda text: None

    adapter.config = adapter.config.model_copy(update={"speech_enabled": True})
    assert asyncio.run(client.synthesize("hello")) == b"RIFFomni"
    assert calls == ["hello"]


# -- single Ollama slot --------------------------------------------------


def _config(**adapter_overrides) -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default"},
            "omnius": {
                "model": "robit/ornith-1.5:9b",
                "vision_model": "robit/ornith-1.5:9b",
                "voice_model": "supertonic",
            },
            "omni_adapter": adapter_overrides,
        }
    )


def test_omni_is_the_default_mode() -> None:
    config = _config()

    assert config.omni_adapter.mode == "omni"
    assert config.omni_adapter.enabled is True
    assert config.omni_adapter.exclusive is True


def test_omni_mode_points_every_call_at_one_ollama_tag() -> None:
    """Two names for identical weights would evict each other on this device."""

    config = _config(mode="omni")

    assert config.omni_adapter.language_model == "robit/ornith-1.5-omni:q4km"
    assert config.omnius.model == "robit/ornith-1.5-omni:q4km"
    assert config.omnius.vision_model == "robit/ornith-1.5-omni:q4km"


def test_traditional_mode_leaves_the_existing_tags_alone() -> None:
    config = _config(mode="traditional")

    assert config.omnius.model == "robit/ornith-1.5:9b"
    assert config.omnius.vision_model == "robit/ornith-1.5:9b"


def test_slot_sharing_can_be_declined_for_a_multi_model_host() -> None:
    config = _config(mode="omni", share_ollama_slot=False)

    assert config.omnius.model == "robit/ornith-1.5:9b"
    assert config.omni_adapter.language_model == "robit/ornith-1.5-omni:q4km"


# -- exclusive omni: nothing else stays loaded ---------------------------


def test_exclusive_omni_silences_every_other_discrete_model() -> None:
    """One weights package means YOLOE, SAM, CLIP, pose, and face ONNX do not load."""

    config = _config(mode="omni")

    assert config.vision.enabled is False
    assert config.identity.enabled is False
    assert config.object_learning.enabled is False
    assert config.occupancy.enabled is False
    assert config.dreams.enabled is False
    assert config.ocr.enabled is False


def test_traditional_mode_keeps_the_whole_discrete_stack() -> None:
    config = _config(mode="traditional")

    assert config.vision.enabled is True
    assert config.identity.enabled is True
    assert config.ocr.enabled is True


def test_omni_can_coexist_with_the_discrete_stack_when_asked() -> None:
    config = _config(mode="omni", exclusive=False)

    assert config.omni_adapter.enabled is True
    assert config.vision.enabled is True
    assert config.identity.enabled is True


def test_a_legacy_enabled_flag_still_selects_the_mode() -> None:
    """A stale `enabled: false` must not silently turn omni on."""

    assert OmniAdapterConfig.model_validate({"enabled": False}).mode == "traditional"
    assert OmniAdapterConfig.model_validate({"enabled": True}).mode == "omni"
    # An explicit mode always wins over the legacy key.
    assert (
        OmniAdapterConfig.model_validate({"enabled": False, "mode": "omni"}).mode == "omni"
    )


def test_capability_pins_narrow_omni_mode_but_never_widen_traditional() -> None:
    omni = OmniAdapterConfig(mode="omni", speech_enabled=False)
    assert omni.uses_speech is False
    assert omni.uses_transcription is True

    traditional = OmniAdapterConfig(mode="traditional", speech_enabled=True)
    assert traditional.uses_speech is False


def test_exclusive_mode_reports_open_vocabulary_audio_without_yamnet() -> None:
    """YAMNet is one of the discrete classifiers the single package replaces."""

    adapter = _client()

    async def describe(wav_audio):
        return "a kettle whistles, then a cupboard closes"

    adapter.describe_audio = describe
    client = _omnius(adapter)

    async def refuse(*args, **kwargs):
        raise AssertionError("exclusive omni mode must not call the YAMNet classifier")

    client._call_audio_classifier = refuse
    result = asyncio.run(client.analyze_audio_scene(_wav(seconds=2.0)))

    assert result["classifications"] == []
    assert result["taxonomy"] == "open-vocabulary"
    assert result["backend"] == "qwen3-omni"
    assert result["observation"] == "a kettle whistles, then a cupboard closes"


# -- video comprehension -------------------------------------------------


def _runtime(**adapter_overrides):
    from egg_companion.runtime import CompanionRuntime

    return CompanionRuntime(
        EggConfig.model_validate(
            {
                "audio": {"input_device": "default", "doa_mode": "disabled"},
                "omnius": {"model": "t", "voice_model": "t"},
                "camera_discovery": {"enabled": False},
                "memory": {"enabled": False},
                "omni_adapter": adapter_overrides,
            }
        )
    )


def _fill_buffer(runtime, camera_id="cam0", frames=14, height=1080, width=1920):
    rng = np.random.default_rng(0)
    for index in range(frames):
        runtime._buffer_video_frame(
            camera_id,
            rng.integers(0, 255, (height, width, 3), dtype=np.uint8),
            index * 0.6,
        )


def test_clip_buffer_is_bounded_decimated_and_downscaled() -> None:
    """A few seconds of full-resolution frames per camera would not be cheap."""

    runtime = _runtime()
    _fill_buffer(runtime)

    buffer = runtime._video_buffers["cam0"]
    # video_buffer_seconds 6.0 * video_fps 2.0
    assert len(buffer) == 12
    assert buffer[0][0].shape[:2] == (360, 640)


def test_no_clip_is_buffered_in_traditional_mode() -> None:
    runtime = _runtime(mode="traditional")
    _fill_buffer(runtime)

    assert runtime._video_buffers == {}


def test_a_buffered_clip_encodes_to_an_mp4_with_its_own_evidence() -> None:
    runtime = _runtime()
    _fill_buffer(runtime, height=720, width=1280)

    payload, evidence = runtime._encode_recent_clip("cam0")

    assert payload[4:8] == b"ftyp"
    assert evidence["camera_id"] == "cam0"
    assert evidence["frames"] == 12
    assert evidence["width"] == 640 and evidence["height"] % 2 == 0


def test_an_empty_buffer_yields_no_clip_rather_than_an_error() -> None:
    runtime = _runtime()

    assert runtime._encode_recent_clip("cam0") is None
    assert asyncio.run(runtime.describe_recent_video("cam0")) is None


def test_video_comprehension_is_optional_evidence_when_the_adapter_is_down() -> None:
    runtime = _runtime()
    _fill_buffer(runtime)

    async def unavailable(*args, **kwargs):
        raise OmniAdapterUnavailable("adapter is down")

    runtime._omni_adapter.describe_video = unavailable

    assert asyncio.run(runtime.describe_recent_video("cam0")) is None


def test_video_comprehension_returns_the_description_with_clip_metadata() -> None:
    runtime = _runtime()
    _fill_buffer(runtime)
    seen: list[dict[str, object]] = []

    async def describe(payload, **kwargs):
        seen.append({"bytes": len(payload), **kwargs})
        return {"visual_observation": "A person raises one hand.", "observation": "..."}

    runtime._omni_adapter.describe_video = describe
    result = asyncio.run(runtime.describe_recent_video("cam0", "what just happened"))

    assert result["visual_observation"] == "A person raises one hand."
    assert result["clip"]["camera_id"] == "cam0"
    assert seen[0]["prompt"] == "what just happened"


def test_the_video_tool_is_advertised_only_when_it_can_be_executed() -> None:
    def names(mode: str) -> list[str]:
        client = OmniusClient(
            OmniusConfig(model="t", voice_model="t"),
            OmniAdapterClient(OmniAdapterConfig(mode=mode)),
        )
        return [item["function"]["name"] for item in client._realtime_tool_definitions()]

    assert "watch_recent_camera_motion" in names("omni")
    assert "watch_recent_camera_motion" not in names("traditional")


def test_the_video_tool_call_normalizes_into_a_video_marker() -> None:
    client = OmniusClient(
        OmniusConfig(model="t", voice_model="t"),
        OmniAdapterClient(OmniAdapterConfig(mode="omni")),
    )

    marker = client._finalize_realtime_message(
        {
            "tool_calls": [
                {
                    "function": {
                        "name": "watch_recent_camera_motion",
                        "arguments": '{"question":"what did they just do","camera_id":"cam0"}',
                    }
                }
            ]
        },
        allow_tool_requests=True,
    )

    call = OmniusClient.parse_realtime_tool_call(marker)
    assert call is not None
    assert call["tool"] == "video"
    assert call["arguments"]["camera_id"] == "cam0"
    assert call["arguments"]["question"] == "what did they just do"


# -- voice presets carried over from the adapter's own profile -----------


def _profile(tmp_path) -> str:
    import json as _json

    voices = tmp_path / "voices"
    voices.mkdir()
    # Distinguishable references, so selecting a preset is actually observable.
    (voices / "female_voice.wav").write_bytes(_wav(seconds=0.25))
    (voices / "default_voice.wav").write_bytes(_wav(seconds=0.5))
    profile = tmp_path / "voice-profile.json"
    profile.write_text(
        _json.dumps(
            {
                "schema": "robit.omni.voice-profile.v1",
                "language": "en",
                "presets": [
                    {"id": "female", "speaker_file": "voices/female_voice.wav", "default": True},
                    {"id": "male", "speaker_file": "voices/default_voice.wav"},
                ],
                "temperature": 0.7,
                "top_k": 40,
                "top_p": 0.9,
                "seed": 42,
                "max_frames": 512,
            }
        )
    )
    return str(profile)


def test_the_profile_default_preset_is_the_voice_egg_speaks_with(tmp_path) -> None:
    client = _client(voice_profile_path=_profile(tmp_path))

    speech = client._speech_settings()

    assert speech["voice"] == "female"
    assert speech["language"] == "en"
    assert speech["temperature"] == 0.7
    assert speech["seed"] == 42
    assert speech["max_frames"] == 512
    assert speech["speaker_audio"]


def test_a_named_preset_selects_its_own_reference(tmp_path) -> None:
    path = _profile(tmp_path)
    female = _client(voice_profile_path=path, voice_preset="female")._speech_settings()
    male = _client(voice_profile_path=path, voice_preset="male")._speech_settings()

    assert male["voice"] == "male"
    assert male["speaker_audio"] != female["speaker_audio"]


def test_an_unknown_preset_falls_back_to_the_profile_default(tmp_path) -> None:
    client = _client(voice_profile_path=_profile(tmp_path), voice_preset="nonexistent")

    assert client._speech_settings()["voice"] == "female"


def test_config_overrides_win_over_the_profile(tmp_path) -> None:
    client = _client(
        voice_profile_path=_profile(tmp_path), voice_temperature=0.1, voice_seed=7
    )

    speech = client._speech_settings()
    assert speech["temperature"] == 0.1
    assert speech["seed"] == 7
    # Untouched values still come from the profile.
    assert speech["top_k"] == 40


def test_a_missing_profile_degrades_to_backend_defaults() -> None:
    client = _client(voice_profile_path="/nonexistent/voice-profile.json")

    assert client._speech_settings() == {}


# -- streamed speech -----------------------------------------------------


def _ndjson_chunks(events: list[dict[str, object]]) -> list[bytes]:
    import json as _json

    return [(_json.dumps(event) + "\n").encode() for event in events]


def _audio_delta(sequence: int, pcm: bytes) -> dict[str, object]:
    import base64 as _b64

    return {
        "type": "audio_delta",
        "audio": {
            "sequence": sequence,
            "block": 0,
            "blocks": 1,
            "encoding": "base64",
            "data": _b64.b64encode(pcm).decode(),
        },
    }


class _FakeStream:
    """Minimal stand-in for aiohttp's streamed response body."""

    def __init__(self, lines: list[bytes], status: int = 200) -> None:
        self.status = status
        self.content = self
        self._lines = lines

    def __aiter__(self):
        async def iterator():
            for line in self._lines:
                yield line

        return iterator()

    async def text(self) -> str:
        return ""

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args) -> None:
        return None


def test_streamed_speech_yields_pcm_windows_in_order(monkeypatch) -> None:
    import egg_companion.adapters.omni as module

    client = _client()
    lines = _ndjson_chunks(
        [
            {"type": "stage", "stage": "tts", "blocks": 1},
            _audio_delta(0, b"\x01\x00" * 40),
            _audio_delta(1, b"\x02\x00" * 40),
            {"type": "audio_end", "samples": 80},
            {"type": "final", "response": {}},
        ]
    )

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return _FakeStream(lines)

    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda *a, **k: _Session())

    async def collect() -> list[bytes]:
        return [chunk async for chunk in client.stream_speech("hello")]

    chunks = asyncio.run(collect())

    assert chunks == [b"\x01\x00" * 40, b"\x02\x00" * 40]
    wav = OmniAdapterClient.pcm_to_wav(b"".join(chunks))
    with wave.open(io.BytesIO(wav), "rb") as source:
        assert source.getframerate() == 24000
        assert source.getnchannels() == 1


def test_a_lost_pcm_window_fails_loudly_rather_than_concatenating(monkeypatch) -> None:
    """A gap means audio was lost; joining across it yields a subtly wrong utterance."""

    import egg_companion.adapters.omni as module

    client = _client()
    lines = _ndjson_chunks([_audio_delta(0, b"\x01\x00" * 8), _audio_delta(2, b"\x02\x00" * 8)])

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return _FakeStream(lines)

    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda *a, **k: _Session())

    async def collect() -> list[bytes]:
        return [chunk async for chunk in client.stream_speech("hello")]

    with pytest.raises(OmniAdapterError, match="lost a window"):
        asyncio.run(collect())


def test_a_stream_error_event_surfaces_as_an_adapter_error(monkeypatch) -> None:
    import egg_companion.adapters.omni as module

    client = _client()
    lines = _ndjson_chunks([{"type": "error", "error": "tts worker died"}])

    class _Session:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        def post(self, *args, **kwargs):
            return _FakeStream(lines)

    monkeypatch.setattr(module.aiohttp, "ClientSession", lambda *a, **k: _Session())

    async def collect() -> list[bytes]:
        return [chunk async for chunk in client.stream_speech("hello")]

    with pytest.raises(OmniAdapterError, match="tts worker died"):
        asyncio.run(collect())


# -- sound-only captures are context, never a request --------------------


def _perceiving(adapter, transcript, observation):
    async def perceive(wav_audio):
        return {
            "transcript": transcript,
            "audio_observation": observation,
            "model": "robit/ornith-1.5-omni:q4km",
            "backend": "qwen3-omni",
        }

    adapter.perceive_audio = perceive


def test_a_sound_only_capture_is_retained_and_never_answered() -> None:
    adapter = _client()
    client = _omnius(adapter)
    _perceiving(adapter, None, "a door closes down the hall")

    transcript = asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert transcript is None
    assert client.pending_audio_context[0]["observation"] == "a door closes down the hall"


def test_retained_sound_observations_are_bounded() -> None:
    adapter = _client()
    client = _omnius(adapter)
    for index in range(9):
        _perceiving(adapter, None, f"sound {index}")
        asyncio.run(
            client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
        )

    retained = client.pending_audio_context
    assert len(retained) == 6
    # The oldest fall off rather than accumulating a stale account of the room.
    assert retained[0]["observation"] == "sound 3"
    assert retained[-1]["observation"] == "sound 8"


def test_consuming_the_audio_context_clears_it() -> None:
    """Replaying them later would present stale room sound as currently audible."""

    adapter = _client()
    client = _omnius(adapter)
    _perceiving(adapter, None, "a kettle whistles")
    asyncio.run(client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence()))

    assert len(client.consume_audio_context()) == 1
    assert client.consume_audio_context() == []
    assert client.pending_audio_context == []


def test_a_spoken_turn_does_not_retain_its_own_sound_as_pending_context() -> None:
    adapter = _client()
    client = _omnius(adapter)
    _perceiving(adapter, "put the kettle on", "a kettle whistles")

    transcript = asyncio.run(
        client.transcribe(_wav(seconds=2.0), acoustic_evidence=_speech_evidence())
    )

    assert transcript == "put the kettle on"
    # It belongs to this turn's evidence, not to the queue awaiting a later one.
    assert client.pending_audio_context == []
    assert client.last_audio_observation["observation"] == "a kettle whistles"


# -- residency integration -----------------------------------------------


def _refusing_manager():
    """A manager that cannot admit anything."""

    from egg_companion.services.residency import WeightResidencyManager

    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0)
    manager.register(
        _residency_component("omni_comprehension", 16.8, loaded=False, fits=False)
    )
    manager.register(_residency_component("omni_speech", 4.0, loaded=False, fits=False))
    return manager


def _residency_component(name: str, cost: float, *, loaded: bool, fits: bool):
    from egg_companion.services.residency import Component

    state = {"loaded": loaded}

    async def load() -> None:
        if not fits:
            raise AssertionError("must not attempt a load that cannot fit")
        state["loaded"] = True

    async def unload() -> None:
        state["loaded"] = False

    async def is_loaded() -> bool:
        return state["loaded"]

    return Component(
        name=name, cost_gib=cost, load=load, unload=unload, is_loaded=is_loaded
    )


def test_a_refused_component_degrades_like_an_unhealthy_adapter(monkeypatch) -> None:
    """Losing a capability this turn is recoverable; an over-commit is not."""

    from egg_companion.services import residency as residency_module

    monkeypatch.setattr(residency_module, "available_memory_gib", lambda: 1.0)
    client = OmniAdapterClient(
        OmniAdapterConfig(mode="omni"), _refusing_manager()
    )
    client._healthy_until = float("inf")

    async def unreachable(*args, **kwargs):
        raise AssertionError("the adapter must not be called without residency")

    client._post = unreachable

    with pytest.raises(OmniAdapterUnavailable, match="cannot be made resident"):
        asyncio.run(client.perceive_audio(_wav()))


def test_speech_is_refused_the_same_way(monkeypatch) -> None:
    from egg_companion.services import residency as residency_module

    monkeypatch.setattr(residency_module, "available_memory_gib", lambda: 1.0)
    client = OmniAdapterClient(
        OmniAdapterConfig(mode="omni", speech_enabled=True), _refusing_manager()
    )
    client._healthy_until = float("inf")

    async def unreachable(*args, **kwargs):
        raise AssertionError("the adapter must not be called without residency")

    client._post = unreachable

    with pytest.raises(OmniAdapterUnavailable):
        asyncio.run(client.synthesize("hello"))


def test_a_client_without_a_manager_behaves_as_before() -> None:
    """Residency is optional; an unmanaged deployment is unchanged."""

    client = _client()
    calls: list[str] = []

    async def respond(payload, *, timeout_seconds):
        calls.append("posted")
        return _adapter_response(observation="<speech_transcript>hi</speech_transcript>")

    client._post = respond
    result = asyncio.run(client.perceive_audio(_wav()))

    assert result["transcript"] == "hi"
    assert calls == ["posted"]


def test_a_resident_component_is_pinned_for_the_request(monkeypatch) -> None:
    """Weights must not be evicted mid-request."""

    from egg_companion.services import residency as residency_module
    from egg_companion.services.residency import WeightResidencyManager

    monkeypatch.setattr(residency_module, "available_memory_gib", lambda: 25.0)
    manager = WeightResidencyManager(total_gib=30.0, reserve_gib=3.0)
    manager.register(
        _residency_component("omni_comprehension", 16.8, loaded=True, fits=True)
    )
    client = OmniAdapterClient(OmniAdapterConfig(mode="omni"), manager)
    client._healthy_until = float("inf")
    seen: list[bool] = []

    async def respond(payload, *, timeout_seconds):
        seen.append(manager._components["omni_comprehension"].pinned)
        return _adapter_response(observation="<speech_transcript>hi</speech_transcript>")

    client._post = respond
    asyncio.run(client.perceive_audio(_wav()))

    assert seen == [True]
    # And released afterwards.
    assert manager._components["omni_comprehension"].pinned is False
