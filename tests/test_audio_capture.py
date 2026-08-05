import io

import numpy as np
import pytest

from egg_companion.adapters.audio import ReSpeakerCapture, ReSpeakerWaveformCapture
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
