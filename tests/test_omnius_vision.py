import asyncio
import io
import wave

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import OmniusConfig


def test_vlm_object_response_requires_bounded_json_label() -> None:
    assert OmniusClient.parse_object_classification('{"label":"ceramic mug","confidence":0.91}') == ("ceramic mug", 0.91)
    assert OmniusClient.parse_object_classification('{"label":null,"confidence":0.91}') is None
    assert OmniusClient.parse_object_classification('a mug') is None
    assert OmniusClient.parse_object_classification('{"label":"mug","confidence":1.2}') is None


def test_person_name_parser_requires_explicit_bounded_json_name() -> None:
    assert OmniusClient.parse_person_name('{"name":"Ada Lovelace"}') == "Ada Lovelace"
    assert OmniusClient.parse_person_name('{"name":null}') is None
    assert OmniusClient.parse_person_name('My name is Ada') is None


def test_asr_grounding_rejects_no_speech_low_probability_and_repetition() -> None:
    silence = {"segments": [{"no_speech_prob": 0.8, "avg_logprob": -0.2, "compression_ratio": 1.0}]}
    unlikely = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -1.2, "compression_ratio": 1.0}]}
    repetitive = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 2.5}]}
    grounded = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.1}]}

    assert OmniusClient.transcription_rejection_reason(silence) == "high no-speech probability"
    assert OmniusClient.transcription_rejection_reason(unlikely) == "low average token probability"
    assert OmniusClient.transcription_rejection_reason(repetitive) == "repetitive transcript compression"
    assert OmniusClient.transcription_rejection_reason(grounded) is None
    assert not OmniusClient.transcription_is_grounded(silence)
    assert OmniusClient.transcription_is_grounded(grounded)


def test_asr_grounding_falls_back_to_text_repetition_when_engine_omits_quality_metadata() -> None:
    # Engines such as transcribe-cli never populate no_speech_prob/avg_logprob/
    # compression_ratio on segments, which would otherwise make the checks
    # above silently unreachable and accept any non-empty hallucinated text.
    repetition_loop = {
        "text": "Allah Allah Allah Allah Allah Allah",
        "segments": [{"id": 0, "start": 0, "end": 3, "text": "Allah Allah Allah Allah Allah Allah"}],
    }
    real_sentence = {
        "text": "Can you turn on the kitchen light",
        "segments": [{"id": 0, "start": 0, "end": 2, "text": "Can you turn on the kitchen light"}],
    }

    assert OmniusClient.transcription_rejection_reason(repetition_loop) == "repetitive transcript compression"
    assert OmniusClient.transcription_rejection_reason(real_sentence) is None


def test_asr_grounding_rejects_backend_silence_artifact_without_quality_metadata() -> None:
    artifact = {
        "text": "Thank you for watching!",
        "segments": [{"id": 0, "start": 0, "end": 2, "text": "Thank you for watching!"}],
    }

    assert OmniusClient.transcription_rejection_reason(artifact) == "known silence hallucination"

    no_segments = {"text": "Thanks for watching!", "segments": []}
    assert OmniusClient.transcription_rejection_reason(no_segments) == (
        "known silence hallucination"
    )

    repeated = {"text": "Thanks for watching. Thanks for watching."}
    assert OmniusClient.transcription_rejection_reason(repeated) == (
        "known silence hallucination"
    )

    translated_artifact = {
        "text": "ご視聴ありがとうございました",
        "segments": [{"id": 0, "start": 0, "end": 12, "text": "ご視聴ありがとうございました"}],
    }
    assert OmniusClient.transcription_rejection_reason(translated_artifact) == (
        "known silence hallucination"
    )

    embellished_artifact = {"text": "Oh, thanks for watching!"}
    assert OmniusClient.transcription_rejection_reason(embellished_artifact) == (
        "known silence hallucination"
    )


def test_asr_grounding_rejects_sparse_text_from_fragmented_max_window() -> None:
    sparse = {
        "text": "Ah! Ah! Shit!",
        "duration": 12,
        "segments": [
            {"id": 0, "start": 0, "end": 1, "text": "Ah!"},
            {"id": 1, "start": 2, "end": 3, "text": "Ah!"},
            {"id": 2, "start": 4, "end": 5, "text": "Shit!"},
        ],
    }
    substantive = {
        "text": "Please stop and tell me why the camera moved toward the window just now",
        "duration": 12,
        "segments": [],
    }
    evidence = {"duration": 12, "boundary_reason": "max_utterance"}

    assert OmniusClient.transcription_rejection_reason(sparse, evidence) == (
        "sparse transcript over max-length acoustic window"
    )
    assert OmniusClient.transcription_rejection_reason(substantive, evidence) is None


def test_asr_grounding_rejects_live_repetition_sparse_and_language_mismatch() -> None:
    repeated = {
        "text": "I can't believe it. I can't believe it. I can't believe it.",
        "segments": [
            {"text": "I can't believe it."},
            {"text": "I can't believe it."},
            {"text": "I can't believe it."},
        ],
    }
    max_window = {"duration": 6, "boundary_reason": "max_utterance"}
    assert OmniusClient.transcription_rejection_reason(repeated, max_window) == (
        "repetitive transcript loop"
    )
    assert OmniusClient.transcription_rejection_reason(
        {"text": "Thank you.", "duration": 6}, max_window
    ) == "sparse transcript over max-length acoustic window"
    assert OmniusClient.transcription_rejection_reason(
        {"text": "JR東日本E233系電車", "duration": 6},
        {**max_window, "requested_language": "en"},
    ) == "transcript script conflicts with requested language"


def test_large_whisper_is_blocked_from_live_jetson_runtime() -> None:
    model = {
        "id": "large-v3",
        "readiness": {"weightsReady": True, "device": "cuda:0"},
    }
    assert "memory budget" in str(OmniusClient._live_asr_unavailable_reason(model))

    base = {
        "id": "base",
        "readiness": {"weightsReady": True, "device": "cuda:0"},
    }
    assert OmniusClient._live_asr_unavailable_reason(base) is None


def test_asr_rejects_digital_silence_before_calling_backend() -> None:
    wav_audio = io.BytesIO()
    with wave.open(wav_audio, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 48000)
    client = OmniusClient(OmniusConfig(model="test-model", voice_model="test-voice"))

    transcript = asyncio.run(client.transcribe(wav_audio.getvalue()))

    assert transcript is None
    assert client.last_transcription_metadata["rejection_reason"] == "digital silence input"
    assert client.last_transcription_metadata["accepted"] is False


def test_asr_rejects_failed_source_acoustic_evidence_after_normalization() -> None:
    evidence = {
        "source_rms": 0.0008,
        "minimum_rms": 0.001,
        "speech_detected": True,
        "wav_rms": 0.08,
        "wav_peak": 0.2,
    }

    assert OmniusClient.acoustic_rejection_reason(evidence) == (
        "source RMS below admission threshold"
    )


def test_asr_rejects_near_floor_ambient_window_before_backend() -> None:
    evidence = {
        "source_rms": 0.061,
        "minimum_rms": 0.05,
        "speech_detected": True,
        "speech_ratio": 0.38,
        "boundary_reason": "max_utterance",
        "wav_rms": 0.12,
        "wav_peak": 0.98,
    }

    assert OmniusClient.acoustic_rejection_reason(evidence) == (
        "near-threshold max-window ambience"
    )


def test_rejected_asr_segment_metadata_does_not_retain_hallucinated_text() -> None:
    segments = [{"id": 0, "start": 0, "end": 12, "text": "Thank you for watching!"}]
    redacted = OmniusClient._segment_metadata(segments, redact_text=True)

    assert redacted == [{"id": 0, "start": 0, "end": 12}]
    assert "watching" not in str(redacted).casefold()


def test_dedicated_asr_backend_rejection_is_authoritative() -> None:
    assert OmniusClient.transcription_rejection_reason(
        {
            "text": "plausible but uncorroborated words",
            "rejection_reason": "dual Whisper disagreement on weak base decode",
        }
    ) == "dual Whisper disagreement on weak base decode"
