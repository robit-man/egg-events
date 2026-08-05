from datetime import datetime, timedelta, timezone

from egg_companion.config import MemoryConfig
from egg_companion.memory.buffer import BufferedMediaRef, PerceptualBuffer


def test_perceptual_buffer_enforces_count_ttl_and_byte_bounds(tmp_path) -> None:
    config = MemoryConfig(
        storage_dir=str(tmp_path),
        buffer_frames_per_camera=2,
        buffer_audio_segments=2,
        buffer_ttl_seconds=10,
        buffer_max_bytes=1_048_576,
    )
    buffer = PerceptualBuffer(config)
    now = datetime.now(timezone.utc)
    for index in range(3):
        buffer.append_frame(
            BufferedMediaRef(
                "camera-0", now + timedelta(seconds=index), f"frame-{index}.jpg", 100, {}
            )
        )
    buffer.append_audio(BufferedMediaRef("mic", now, "old.wav", 100, {"vad": True}))

    snapshot = buffer.snapshot(now + timedelta(seconds=11))

    assert [item.media_key for item in snapshot["frames"]["camera-0"]] == ["frame-1.jpg", "frame-2.jpg"]
    assert snapshot["audio"] == []
    assert snapshot["bytes"] == 200
