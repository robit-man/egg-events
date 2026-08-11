from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, replace
from typing import Literal
from uuid import uuid4


VoiceFloor = Literal[
    "listening",
    "audio_detected",
    "transcribing",
    "processing",
    "response_playing",
    "barge_pending",
]
PlaybackStatus = Literal[
    "playing", "barge_pending", "completed", "interrupted", "superseded", "failed"
]
BargeOutcome = Literal["resume", "interrupted", "audio_first"]


@dataclass(frozen=True)
class TranscriptTurn:
    turn_id: str
    revision: int
    role: Literal["heard", "agent"]
    text: str
    status: str
    started_at: float
    ended_at: float


@dataclass(frozen=True)
class PlaybackLease:
    playback_id: str
    response_revision: int
    text: str
    status: PlaybackStatus
    started_at: float
    resume_seconds: float = 0.0


@dataclass(frozen=True)
class BargeLease:
    barge_id: str
    playback_id: str
    response_revision: int
    started_at: float
    resume_seconds: float = 0.0
    heard_revision: int | None = None


@dataclass(frozen=True)
class AudioTurn:
    utterance_id: str
    revision: int
    text: str
    started_at: float
    ended_at: float
    barge_id: str | None = None


class ConversationTurnController:
    """Single in-memory authority for Egg's conversational floor.

    Model and TTS work may happen speculatively, but a response can start
    playback only while its heard-audio revision is still current. Playback and
    barge callbacks are also identity-bound, making stale completions harmless.
    """

    def __init__(self, history_limit: int = 24) -> None:
        self.revision = 0
        self.floor: VoiceFloor = "listening"
        self.active_playback: PlaybackLease | None = None
        self.active_barge: BargeLease | None = None
        self.pending_ingress = 0
        self._history: deque[TranscriptTurn] = deque(maxlen=max(4, history_limit))

    def is_current(self, revision: int) -> bool:
        return revision == self.revision

    def can_publish(self, revision: int) -> bool:
        return revision == self.revision and self.pending_ingress == 0

    def barge_decision_current(self, barge_id: str, heard_revision: int) -> bool:
        return bool(
            self.can_publish(heard_revision)
            and self.active_barge is not None
            and self.active_barge.barge_id == barge_id
            and self.active_barge.heard_revision == heard_revision
        )

    def speech_started(self, started_at: float | None = None) -> BargeLease | None:
        now = time.monotonic() if started_at is None else started_at
        self.pending_ingress += 1
        playback = self.active_playback
        if (
            playback is not None
            and playback.status == "barge_pending"
            and self.active_barge is not None
        ):
            self.floor = "barge_pending"
            return self.active_barge
        if playback is None or playback.status != "playing":
            self.floor = "audio_detected"
            return None
        barge = BargeLease(
            barge_id=str(uuid4()),
            playback_id=playback.playback_id,
            response_revision=playback.response_revision,
            started_at=now,
        )
        self.active_playback = replace(playback, status="barge_pending")
        self.active_barge = barge
        self.floor = "barge_pending"
        return barge

    def bind_barge_cursor(self, barge_id: str, resume_seconds: float) -> BargeLease | None:
        barge = self.active_barge
        if barge is None or barge.barge_id != barge_id:
            return None
        cursor = max(0.0, float(resume_seconds))
        self.active_barge = replace(barge, resume_seconds=cursor)
        if self.active_playback is not None:
            self.active_playback = replace(self.active_playback, resume_seconds=cursor)
        return self.active_barge

    def speech_ended(self) -> None:
        self.floor = "transcribing"

    def reject_audio_input(self) -> None:
        self.pending_ingress = max(0, self.pending_ingress - 1)
        if self.active_barge is None and self.pending_ingress == 0:
            self.floor = "listening"

    def reject_reasoning(self) -> None:
        if self.active_playback is None and self.pending_ingress == 0:
            self.floor = "listening"

    def finish_processing(self, revision: int) -> None:
        if (
            revision == self.revision
            and self.active_playback is None
            and self.pending_ingress == 0
        ):
            self.floor = "listening"

    def finalize_audio_turn(
        self,
        text: str,
        *,
        utterance_id: str,
        started_at: float,
        ended_at: float | None = None,
        barge_id: str | None = None,
    ) -> AudioTurn:
        normalized = " ".join(text.strip().split())
        if not normalized:
            raise ValueError("heard audio text is required")
        self.pending_ingress = max(0, self.pending_ingress - 1)
        self.revision += 1
        ended = time.monotonic() if ended_at is None else ended_at
        turn = AudioTurn(
            utterance_id=utterance_id,
            revision=self.revision,
            text=normalized,
            started_at=started_at,
            ended_at=ended,
            barge_id=barge_id,
        )
        self._history.append(
            TranscriptTurn(
                turn_id=utterance_id,
                revision=turn.revision,
                role="heard",
                text=normalized,
                status="final",
                started_at=started_at,
                ended_at=ended,
            )
        )
        if barge_id and self.active_barge and self.active_barge.barge_id == barge_id:
            self.active_barge = replace(self.active_barge, heard_revision=turn.revision)
        self.floor = "processing"
        return turn

    def begin_playback(
        self,
        text: str,
        *,
        expected_revision: int,
        playback_id: str | None = None,
        started_at: float | None = None,
    ) -> PlaybackLease | None:
        normalized = " ".join(text.strip().split())
        if (
            not normalized
            or expected_revision != self.revision
            or self.pending_ingress > 0
            or self.active_playback is not None
        ):
            return None
        playback = PlaybackLease(
            playback_id=playback_id or str(uuid4()),
            response_revision=expected_revision,
            text=normalized,
            status="playing",
            started_at=time.monotonic() if started_at is None else started_at,
        )
        self.active_playback = playback
        self.floor = "response_playing"
        return playback

    def resolve_barge(
        self,
        barge_id: str,
        outcome: BargeOutcome,
        *,
        ended_at: float | None = None,
    ) -> PlaybackLease | None:
        barge = self.active_barge
        playback = self.active_playback
        if (
            barge is None
            or playback is None
            or barge.barge_id != barge_id
            or playback.playback_id != barge.playback_id
            or playback.status != "barge_pending"
        ):
            return None
        if outcome == "resume":
            self.active_barge = None
            self.active_playback = replace(
                playback, status="playing", resume_seconds=barge.resume_seconds
            )
            self.floor = "response_playing"
            return self.active_playback
        status: PlaybackStatus = "interrupted"
        terminal = replace(playback, status=status, resume_seconds=barge.resume_seconds)
        self._record_agent_terminal(terminal, ended_at)
        self.active_barge = None
        self.active_playback = None
        self.floor = "processing" if barge.heard_revision is not None else "listening"
        return terminal

    def complete_playback(
        self, playback_id: str, *, ended_at: float | None = None
    ) -> PlaybackLease | None:
        playback = self.active_playback
        if (
            playback is None
            or playback.playback_id != playback_id
            or playback.status != "playing"
        ):
            return None
        terminal = replace(playback, status="completed")
        self._record_agent_terminal(terminal, ended_at)
        self.active_playback = None
        self.floor = "listening"
        return terminal

    def terminate_playback(
        self,
        playback_id: str,
        status: Literal["superseded", "failed"],
        *,
        ended_at: float | None = None,
    ) -> PlaybackLease | None:
        playback = self.active_playback
        if playback is None or playback.playback_id != playback_id:
            return None
        terminal = replace(playback, status=status)
        self._record_agent_terminal(terminal, ended_at)
        self.active_playback = None
        self.active_barge = None
        self.floor = "listening"
        return terminal

    def history(self) -> tuple[TranscriptTurn, ...]:
        return tuple(sorted(self._history, key=lambda turn: (turn.started_at, turn.ended_at)))

    def prompt_history(self) -> list[dict[str, object]]:
        return [
            {
                "role": turn.role,
                "text": turn.text,
                "revision": turn.revision,
                "status": turn.status,
            }
            for turn in self.history()
        ]

    def snapshot(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "floor": self.floor,
            "active_playback_id": (
                self.active_playback.playback_id if self.active_playback else None
            ),
            "playback_status": (
                self.active_playback.status if self.active_playback else None
            ),
            "active_barge_id": self.active_barge.barge_id if self.active_barge else None,
            "pending_ingress": self.pending_ingress,
            "history_turns": len(self._history),
        }

    def _record_agent_terminal(
        self, playback: PlaybackLease, ended_at: float | None
    ) -> None:
        ended = time.monotonic() if ended_at is None else ended_at
        self._history.append(
            TranscriptTurn(
                turn_id=playback.playback_id,
                revision=playback.response_revision,
                role="agent",
                text=playback.text,
                status=playback.status,
                started_at=playback.started_at,
                ended_at=ended,
            )
        )
