from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock

from egg_companion.config import MemoryConfig


@dataclass(frozen=True)
class BufferedMediaRef:
    source_id: str
    captured_at: datetime
    media_key: str
    size_bytes: int
    metadata: dict[str, object]


class PerceptualBuffer:
    """Bounded references to recent media; decoded frames and audio are never retained."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._frames: dict[str, deque[BufferedMediaRef]] = {}
        self._audio: deque[BufferedMediaRef] = deque(maxlen=config.buffer_audio_segments)
        self._lock = RLock()

    def append_frame(self, ref: BufferedMediaRef) -> None:
        with self._lock:
            frames = self._frames.setdefault(
                ref.source_id, deque(maxlen=self.config.buffer_frames_per_camera)
            )
            frames.append(ref)
            self._prune(ref.captured_at)

    def append_audio(self, ref: BufferedMediaRef) -> None:
        with self._lock:
            self._audio.append(ref)
            self._prune(ref.captured_at)

    def snapshot(self, now: datetime) -> dict[str, object]:
        with self._lock:
            self._prune(now)
            frames = {
                source_id: list(items) for source_id, items in self._frames.items() if items
            }
            audio = list(self._audio)
            return {
                "frames": frames,
                "audio": audio,
                "bytes": sum(
                    item.size_bytes
                    for items in (*frames.values(), audio)
                    for item in items
                ),
            }

    def _prune(self, now: datetime) -> None:
        cutoff = now - timedelta(seconds=self.config.buffer_ttl_seconds)
        for source_id, items in list(self._frames.items()):
            while items and items[0].captured_at < cutoff:
                items.popleft()
            if not items:
                self._frames.pop(source_id, None)
        while self._audio and self._audio[0].captured_at < cutoff:
            self._audio.popleft()
        while self._total_bytes() > self.config.buffer_max_bytes:
            oldest_frame = min(
                ((items[0].captured_at, items) for items in self._frames.values() if items),
                default=None,
                key=lambda item: item[0],
            )
            oldest_audio = self._audio[0].captured_at if self._audio else None
            if oldest_frame is not None and (
                oldest_audio is None or oldest_frame[0] <= oldest_audio
            ):
                oldest_frame[1].popleft()
            elif self._audio:
                self._audio.popleft()
            else:
                break

    def _total_bytes(self) -> int:
        return sum(
            item.size_bytes
            for items in (*self._frames.values(), self._audio)
            for item in items
        )
