from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta

from egg_companion.config import EventSegmentationConfig, MemoryConfig
from egg_companion.models import EpisodeDraft, PerceptualEvent


@dataclass
class _ActiveEpisode:
    episode_id: str
    context_key: str
    started_at: object
    last_event_at: object
    evidence: list = field(default_factory=list)
    entity_ids: set[str] = field(default_factory=set)
    signature: tuple = ()
    pending_signature: tuple = ()
    pending_count: int = 0
    surprise: dict[str, float] = field(default_factory=dict)


class EventSegmenter:
    """Converts repeated sensory observations into bounded, explainable episodes."""

    def __init__(self, memory: MemoryConfig, config: EventSegmentationConfig) -> None:
        self.memory = memory
        self.config = config
        self._active: dict[str, _ActiveEpisode] = {}
        self._last_boundary: dict[str, object] = {}
        self._accepted_by_context: dict[str, int] = {}
        self._closed_by_context: dict[str, int] = {}

    @staticmethod
    def _context(event: PerceptualEvent) -> str:
        return (
            "conversation"
            if event.event_type in {"speech", "audio_comprehension", "user_correction"}
            else event.source_id
        )

    @staticmethod
    def _signature(event: PerceptualEvent) -> tuple:
        # Open-vocabulary detector class names are intentionally excluded from vision
        # boundaries: they oscillate on static masks. Stable scene cues and resolved
        # entities remain eligible, while the raw labels stay attached to evidence.
        label_key = "scene_labels" if event.event_type == "vision" else "labels"
        labels = tuple(sorted(str(label) for label in event.payload.get(label_key, ())))
        boundary_entities = event.payload.get("boundary_entity_ids", event.entity_ids)
        entity_ids = tuple(sorted(str(value) for value in boundary_entities))
        behavior = (
            tuple(
                sorted(
                    str(value)
                    for value in event.payload.get(
                        "boundary_behaviors", event.payload.get("behaviors", ())
                    )
                )
            )
            if entity_ids
            else ()
        )
        return event.event_type, entity_ids, labels, behavior

    def ingest(self, event: PerceptualEvent) -> tuple[bool, tuple[EpisodeDraft, ...]]:
        """Return whether event evidence was accepted plus any closed episode drafts."""
        context = self._context(event)
        active = self._active.get(context)
        signature = self._signature(event)
        if active is None:
            self._active[context] = self._start(context, event, signature)
            self._record_boundary(context, event, "new context", {"new_context": 1.0})
            self._count(self._accepted_by_context, context)
            return True, ()
        elapsed = event.occurred_at - active.last_event_at
        duration = event.occurred_at - active.started_at
        explicit_boundary = event.event_type in {
            "speech", "audio_comprehension", "user_correction"
        }
        changed = signature != active.signature
        maxed = duration >= timedelta(seconds=self.memory.episode_max_seconds)
        entity_changed = signature[1] != active.signature[1] and bool(
            signature[1] or active.signature[1]
        )
        confirmed_change = False
        # Stable entity IDs still blink as masks briefly disappear or recall
        # resolves asynchronously. Require the same changed signature across
        # three observations for entity changes too; otherwise one missed frame
        # turns a static room into dozens of false episode boundaries.
        if changed:
            if signature == active.pending_signature:
                active.pending_count += 1
            else:
                active.pending_signature = signature
                active.pending_count = 1
            confirmed_change = active.pending_count >= 3
        else:
            active.pending_signature = ()
            active.pending_count = 0
        if explicit_boundary or confirmed_change or maxed:
            reason = (
                "explicit interaction" if explicit_boundary
                else "entity changed" if entity_changed and confirmed_change
                else "confirmed scene change" if confirmed_change
                else "maximum duration"
            )
            closed = self._close(active, event.occurred_at)
            self._active[context] = self._start(context, event, signature)
            self._record_boundary(
                context,
                event,
                reason,
                {
                    "explicit": float(explicit_boundary),
                    "entity_change": float(entity_changed),
                    "scene_change": float(confirmed_change),
                    "max_duration": float(maxed),
                },
            )
            self._count(self._accepted_by_context, context)
            self._count(self._closed_by_context, context)
            return True, (closed,)
        active.last_event_at = event.occurred_at
        if elapsed >= timedelta(seconds=self.config.inactivity_seconds):
            closed = self._close(active, event.occurred_at)
            self._active[context] = self._start(context, event, signature)
            self._record_boundary(context, event, "inactivity", {"inactivity": 1.0})
            self._count(self._accepted_by_context, context)
            self._count(self._closed_by_context, context)
            return True, (closed,)
        # Repeated or transiently relabelled frames maintain continuity without durable inflation.
        return False, ()

    def flush(self, at) -> tuple[EpisodeDraft, ...]:
        drafts = tuple(self._close(active, at) for active in self._active.values())
        self._active.clear()
        return drafts

    def snapshot(self) -> dict[str, object]:
        return {
            "active": [
                {
                    "episode_id": episode.episode_id,
                    "context": episode.context_key,
                    "started_at": episode.started_at.isoformat(),
                    "last_event_at": episode.last_event_at.isoformat(),
                }
                for episode in self._active.values()
            ],
            "last_boundary": dict(self._last_boundary),
            "accepted_by_context": dict(self._accepted_by_context),
            "closed_by_context": dict(self._closed_by_context),
        }

    @staticmethod
    def _count(counters: dict[str, int], context: str) -> None:
        counters[context] = counters.get(context, 0) + 1

    def _record_boundary(
        self,
        context: str,
        event: PerceptualEvent,
        reason: str,
        components: dict[str, float],
    ) -> None:
        self._last_boundary = {
            "context": context,
            "event_id": event.event_id,
            "at": event.occurred_at.isoformat(),
            "reason": reason,
            "components": components,
        }

    def _start(self, context: str, event: PerceptualEvent, signature: tuple) -> _ActiveEpisode:
        active = _ActiveEpisode(event.event_id, context, event.occurred_at, event.occurred_at, signature=signature)
        active.evidence.extend(event.evidence)
        active.entity_ids.update(event.entity_ids)
        return active

    @staticmethod
    def _close(active: _ActiveEpisode, ended_at) -> EpisodeDraft:
        return EpisodeDraft(
            episode_id=active.episode_id,
            context_key=active.context_key,
            started_at=active.started_at,
            ended_at=ended_at,
            evidence=tuple(active.evidence),
            entity_ids=tuple(sorted(active.entity_ids)),
            surprise=dict(active.surprise),
        )
