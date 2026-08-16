from __future__ import annotations

import math
from typing import Protocol


class ActivitySettings(Protocol):
    enabled: bool
    idle_floor: float
    decay_seconds: float
    novelty_threshold: float


class ActivityGovernor:
    """Whole-system alertness, driven by visual novelty/presence and audio
    speech activity, that perception loops scale their inference rate against.

    Mirrors habituation: as long as the room holds something worth looking at
    or listening to, alertness sits at 1.0 and inference runs at its full
    configured rate. Once the cameras and microphone go genuinely quiet,
    alertness decays exponentially toward `idle_floor` over `decay_seconds`.
    Any new novelty, presence, or speech snaps alertness back to 1.0
    immediately, so recovery is instant while the wind-down is gradual.
    """

    def __init__(self, settings: ActivitySettings) -> None:
        self._settings = settings
        self._last_activity_at: float | None = None

    def note_visual(self, novelty: float, has_presence: bool, now: float) -> None:
        if has_presence or novelty >= self._settings.novelty_threshold:
            self._last_activity_at = now

    def note_audio(self, speech_detected: bool, now: float) -> None:
        if speech_detected:
            self._last_activity_at = now

    def scale(self, now: float) -> float:
        """Current alertness in [idle_floor, 1.0]. 1.0 while active or unstarted."""
        if not self._settings.enabled or self._last_activity_at is None:
            return 1.0
        elapsed = max(0.0, now - self._last_activity_at)
        floor = self._settings.idle_floor
        decayed = floor + (1.0 - floor) * math.exp(-elapsed / self._settings.decay_seconds)
        return max(floor, min(1.0, decayed))

    def scaled_fps(self, base_fps: float, now: float) -> float:
        """Lower effective fps while quiet; unchanged while active."""
        return base_fps * self.scale(now)

    def scaled_interval(self, base_interval: float, now: float) -> float:
        """Wider effective interval (i.e. lower frequency) while quiet."""
        return base_interval / self.scale(now)
