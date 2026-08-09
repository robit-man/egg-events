import numpy as np

from egg_companion.adapters.audio import UtteranceSegmenter
from egg_companion.config import AudioConfig, TranscriptionConfig


def _segmenter(voiced_frames: set[int]) -> tuple[UtteranceSegmenter, list[int]]:
    audio = AudioConfig(input_device="default")
    transcription = TranscriptionConfig(
        segment_seconds=2.0,
        vad_min_contiguous_ms=60,  # 2 frames at 30ms onset requirement
        vad_hangover_ms=100,  # rounds to 3 frames at 30ms (100/30 -> 3.33 -> 3)
        vad_pre_roll_ms=120,  # 4 frames at 30ms of pre-roll lookback (> onset, so
        # it retains genuine silent lead-in, not just the onset frames themselves)
    )
    segmenter = UtteranceSegmenter(audio, transcription)
    frame_calls: list[int] = []

    def fake_is_voiced(frame: np.ndarray) -> bool:
        index = len(frame_calls)
        frame_calls.append(index)
        return index in voiced_frames

    segmenter._is_voiced = fake_is_voiced  # type: ignore[method-assign]
    return segmenter, frame_calls


def _frames(count: int, sample_rate: int = 16000, frame_ms: int = 30) -> np.ndarray:
    frame_samples = sample_rate * frame_ms // 1000
    # Distinct per-frame values so pre-roll/utterance contents are verifiable.
    return np.concatenate(
        [np.full(frame_samples, fill_value=float(index), dtype=np.float32) for index in range(count)]
    )


def test_utterance_starts_only_after_contiguous_onset_and_includes_pre_roll() -> None:
    # Frames 0-1 silent (genuine pre-roll lead-in), frames 2-3 voiced (onset
    # confirmed at frame 3), frame 4 voiced, frames 5-7 silent (hangover of 3
    # frames triggers finalize).
    segmenter, _ = _segmenter(voiced_frames={2, 3, 4})
    samples = _frames(8)

    utterances = segmenter.feed(samples)

    assert len(utterances) == 1
    frame_samples = 16000 * 30 // 1000
    utterance_frame_count = utterances[0].size // frame_samples
    # Pre-roll (frames 0-1) + onset/speech (2-4) + hangover (5-7) = 8 frames.
    assert utterance_frame_count == 8
    assert utterances[0][0] == 0.0  # silent pre-roll lead-in retained, not clipped


def test_no_utterance_without_sustained_onset() -> None:
    # A single voiced frame surrounded by silence never reaches the onset
    # requirement (2 contiguous frames), so nothing should ever be recorded.
    segmenter, _ = _segmenter(voiced_frames={3})
    samples = _frames(10)

    utterances = segmenter.feed(samples)

    assert utterances == []


def test_utterance_finalizes_at_max_length_cap_without_hangover() -> None:
    # Continuous speech with no silence must still finalize at the
    # segment_seconds cap rather than buffering forever; enough input is fed to
    # cross that cap twice, confirming it keeps chunking rather than growing
    # unbounded.
    segmenter, _ = _segmenter(voiced_frames=set(range(200)))
    samples = _frames(200)

    utterances = segmenter.feed(samples)

    assert len(utterances) >= 2
    frame_samples = 16000 * 30 // 1000
    max_frames = round(2.0 * 1000 / 30)
    assert utterances[0].size // frame_samples == max_frames
    assert utterances[1].size // frame_samples == max_frames


def test_reset_discards_in_progress_utterance() -> None:
    segmenter, _ = _segmenter(voiced_frames={0, 1, 2})
    segmenter.feed(_frames(3))  # onset confirmed, recording in progress

    segmenter.reset()
    utterances = segmenter.feed(_frames(3))

    # After reset, onset must be re-confirmed from scratch; three fresh voiced
    # frames using the same fake_is_voiced indices as before will not appear
    # voiced again (indices keep incrementing), so nothing should finalize yet.
    assert utterances == []
