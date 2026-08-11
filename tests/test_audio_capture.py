import io
import wave

import numpy as np
import pytest

from egg_companion.adapters.audio import (
    ReSpeakerCapture,
    ReSpeakerWaveformCapture,
    condition_speech_band,
)
from egg_companion.config import AudioConfig, TranscriptionConfig


class FinishedProcess:
    def __init__(self, payload: bytes) -> None:
        self.stdout = io.BytesIO(payload)

    @staticmethod
    def poll() -> int:
        return 0


def test_waveform_short_read_resets_process_and_recovers_next_chunk() -> None:
    audio = AudioConfig(input_device="default", channels=6, waveform_fps=30)
    capture = ReSpeakerWaveformCapture(audio)
    frame_count = audio.sample_rate // audio.waveform_fps
    expected_bytes = frame_count * audio.channels * 2
    capture._ensure_process = lambda: None
    capture._process = FinishedProcess(bytes(expected_bytes - 2))

    with pytest.raises(RuntimeError, match="waveform capture was short"):
        capture.read_chunk()
    assert capture._process is None

    capture._process = FinishedProcess(bytes(expected_bytes))
    samples = capture.read_chunk()

    assert samples.shape == (frame_count,)
    assert not samples.any()


def test_single_stream_samples_are_vad_gated_and_encoded_for_asr() -> None:
    audio = AudioConfig(input_device="default", channels=6)
    processor = ReSpeakerCapture(audio, TranscriptionConfig())

    wav_audio, rms = processor.process_samples(np.zeros(audio.sample_rate * 3, dtype=np.float32))

    assert wav_audio.startswith(b"RIFF")
    assert rms == 0
    assert not processor.last_speech_detected
    assert processor.last_speech_ms == 0


def test_speech_band_conditioning_removes_respeaker_rumble_but_preserves_speech() -> None:
    sample_rate = 16000
    seconds = 3
    timeline = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    rumble = 0.08 * np.sin(2 * np.pi * 117 * timeline)
    speech_tone = 0.08 * np.sin(2 * np.pi * 1000 * timeline)

    filtered_rumble = condition_speech_band(rumble, sample_rate)
    filtered_speech = condition_speech_band(speech_tone, sample_rate)

    assert np.sqrt(np.mean(np.square(filtered_rumble))) < 0.01
    assert np.sqrt(np.mean(np.square(filtered_speech))) > 0.05


def test_asr_gain_uses_conditioned_voiced_energy_instead_of_raw_rumble() -> None:
    sample_rate = 16000
    audio = AudioConfig(input_device="default", channels=6, sample_rate=sample_rate)
    processor = ReSpeakerCapture(audio, TranscriptionConfig(vad_min_voiced_rms=0.02))
    timeline = np.arange(sample_rate * 3, dtype=np.float32) / sample_rate
    rumble = 0.08 * np.sin(2 * np.pi * 117 * timeline)

    wav_audio, raw_rms = processor.process_samples(rumble.astype(np.float32))
    with wave.open(io.BytesIO(wav_audio), "rb") as source:
        output = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2").astype(
            np.float32
        ) / 32768

    assert raw_rms > 0.05
    assert processor.last_conditioned_rms < 0.01
    assert not processor.last_speech_detected
    assert np.sqrt(np.mean(np.square(output))) < 0.1
