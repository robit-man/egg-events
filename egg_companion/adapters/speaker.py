from __future__ import annotations

import asyncio
import io
import time
import wave
from dataclasses import dataclass
from typing import Literal

from egg_companion.config import AudioConfig


PlaybackOutcome = Literal["completed", "interrupted"]


@dataclass(frozen=True)
class PlaybackResult:
    playback_id: str
    outcome: PlaybackOutcome
    played_seconds: float
    resume_seconds: float
    duration_seconds: float


@dataclass
class _ActivePlayback:
    playback_id: str
    source_audio: bytes
    process: asyncio.subprocess.Process
    started_at: float
    start_seconds: float
    duration_seconds: float
    interrupted: bool = False
    resume_seconds: float = 0.0
    settled: asyncio.Event | None = None


@dataclass(frozen=True)
class _PausedPlayback:
    playback_id: str
    source_audio: bytes
    resume_seconds: float
    duration_seconds: float


@dataclass
class _StartingPlayback:
    playback_id: str
    source_audio: bytes
    start_seconds: float
    duration_seconds: float
    interrupted: bool = False
    resume_seconds: float = 0.0
    settled: asyncio.Event | None = None


class Speaker:
    """Identity-bound, cancellable WAV playback with tail-only resume."""

    def __init__(self, config: AudioConfig) -> None:
        self.config = config
        self._starting: _StartingPlayback | None = None
        self._active: _ActivePlayback | None = None
        self._paused: _PausedPlayback | None = None

    @property
    def active_playback_id(self) -> str | None:
        if self._active:
            return self._active.playback_id
        return self._starting.playback_id if self._starting else None

    @property
    def is_playing(self) -> bool:
        return bool(
            (self._starting is not None and not self._starting.interrupted)
            or (self._active is not None and not self._active.interrupted)
        )

    @property
    def paused_playback_id(self) -> str | None:
        return self._paused.playback_id if self._paused else None

    async def play_wav(
        self, audio: bytes, *, playback_id: str, start_seconds: float = 0.0
    ) -> PlaybackResult:
        if self._starting is not None or self._active is not None:
            raise RuntimeError("speaker already owns an active playback")
        duration = self.wav_duration(audio)
        start = min(duration, max(0.0, float(start_seconds)))
        if start >= duration:
            self._paused = None
            return PlaybackResult(playback_id, "completed", 0.0, duration, duration)
        payload = audio if start == 0 else self.slice_wav(audio, start)
        starting = _StartingPlayback(
            playback_id=playback_id,
            source_audio=audio,
            start_seconds=start,
            duration_seconds=duration,
            resume_seconds=start,
            settled=asyncio.Event(),
        )
        self._starting = starting
        try:
            process = await asyncio.create_subprocess_exec(
                "aplay",
                "-q",
                "-D",
                self.config.output_device,
                "-",
                stdin=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except BaseException:
            if self._starting is starting:
                self._starting = None
            if starting.settled is not None:
                starting.settled.set()
            raise
        active = _ActivePlayback(
            playback_id=playback_id,
            source_audio=audio,
            process=process,
            started_at=time.monotonic(),
            start_seconds=start,
            duration_seconds=duration,
            interrupted=starting.interrupted,
            resume_seconds=starting.resume_seconds,
            settled=starting.settled,
        )
        self._active = active
        if self._starting is starting:
            self._starting = None
        self._paused = None
        try:
            if active.interrupted and process.returncode is None:
                process.terminate()
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(payload), timeout=self.config.playback_timeout_seconds
                )
            except asyncio.TimeoutError as error:
                process.kill()
                await process.wait()
                raise RuntimeError("audio playback timed out") from error
            if active.interrupted:
                paused = _PausedPlayback(
                    playback_id=playback_id,
                    source_audio=audio,
                    resume_seconds=active.resume_seconds,
                    duration_seconds=duration,
                )
                self._paused = paused
                return PlaybackResult(
                    playback_id,
                    "interrupted",
                    max(0.0, active.resume_seconds - start),
                    active.resume_seconds,
                    duration,
                )
            exit_code = process.returncode
            if exit_code:
                detail = stderr.decode("utf-8", errors="replace").strip() if stderr else ""
                raise RuntimeError(f"audio playback exited with {exit_code}: {detail}")
            self._paused = None
            return PlaybackResult(
                playback_id,
                "completed",
                max(0.0, duration - start),
                duration,
                duration,
            )
        except asyncio.CancelledError:
            if process.returncode is None:
                process.terminate()
                try:
                    await asyncio.wait_for(process.wait(), timeout=1)
                except asyncio.TimeoutError:
                    process.kill()
                    await process.wait()
            self._paused = None
            raise
        finally:
            if self._active is active:
                self._active = None
            if active.settled is not None:
                active.settled.set()

    async def interrupt(self, playback_id: str) -> PlaybackResult | None:
        active = self._active
        starting = self._starting
        if active is None and starting is not None and starting.playback_id == playback_id:
            starting.interrupted = True
            starting.resume_seconds = starting.start_seconds
            return PlaybackResult(
                playback_id,
                "interrupted",
                0.0,
                starting.resume_seconds,
                starting.duration_seconds,
            )
        if active is None or active.playback_id != playback_id:
            return None
        elapsed = max(0.0, time.monotonic() - active.started_at)
        rewind = self.config.playback_resume_rewind_ms / 1000
        active.resume_seconds = min(
            active.duration_seconds,
            max(active.start_seconds, active.start_seconds + elapsed - rewind),
        )
        active.interrupted = True
        if active.process.returncode is None:
            active.process.terminate()
        asyncio.create_task(
            self._enforce_interruption(active),
            name=f"speaker-interrupt-watchdog:{playback_id}",
        )
        return PlaybackResult(
            playback_id,
            "interrupted",
            max(0.0, active.resume_seconds - active.start_seconds),
            active.resume_seconds,
            active.duration_seconds,
        )

    async def resume(self, playback_id: str) -> PlaybackResult | None:
        settling = None
        if self._starting is not None and self._starting.playback_id == playback_id:
            settling = self._starting.settled
        elif self._active is not None and self._active.playback_id == playback_id:
            settling = self._active.settled
        if settling is not None:
            try:
                await asyncio.wait_for(settling.wait(), timeout=2)
            except asyncio.TimeoutError:
                return None
        paused = self._paused
        if paused is None or paused.playback_id != playback_id:
            return None
        return await self.play_wav(
            paused.source_audio,
            playback_id=playback_id,
            start_seconds=paused.resume_seconds,
        )

    def discard(self, playback_id: str) -> bool:
        if self._paused is None or self._paused.playback_id != playback_id:
            return False
        self._paused = None
        return True

    @staticmethod
    async def _enforce_interruption(active: _ActivePlayback) -> None:
        if active.settled is None:
            return
        try:
            await asyncio.wait_for(active.settled.wait(), timeout=1)
            return
        except asyncio.TimeoutError:
            pass
        if active.process.returncode is None:
            active.process.kill()
            try:
                await asyncio.wait_for(active.process.wait(), timeout=1)
            except asyncio.TimeoutError:
                return

    @staticmethod
    def wav_duration(audio: bytes) -> float:
        try:
            with wave.open(io.BytesIO(audio), "rb") as wav:
                return wav.getnframes() / max(1, wav.getframerate())
        except (EOFError, wave.Error) as error:
            raise RuntimeError("speaker received an invalid WAV payload") from error

    @staticmethod
    def slice_wav(audio: bytes, start_seconds: float) -> bytes:
        try:
            with wave.open(io.BytesIO(audio), "rb") as source:
                params = source.getparams()
                start_frame = min(
                    params.nframes,
                    max(0, round(float(start_seconds) * params.framerate)),
                )
                source.setpos(start_frame)
                frames = source.readframes(params.nframes - start_frame)
        except (EOFError, wave.Error) as error:
            raise RuntimeError("speaker received an invalid WAV payload") from error
        payload = io.BytesIO()
        with wave.open(payload, "wb") as target:
            target.setparams(params)
            target.writeframes(frames)
        return payload.getvalue()
