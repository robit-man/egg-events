"""Bitemporal time semantics for the world model.

Every semantic state transition carries two notions of time:

    VALID TIME     — When this was true in the world.
    SYSTEM TIME    — When Egg learned/recorded it.

This enables:
    "What did Egg believe at 10:03?"
    vs
    "Where does Egg now believe the object was at 10:03?"
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class BitemporalInterval:
    """A closed-open interval in both valid and system time."""

    valid_from: datetime
    valid_to: datetime | None = None
    recorded_at: datetime = field(default_factory=utcnow)
    superseded_at: datetime | None = None

    def is_valid_at(self, when: datetime) -> bool:
        if when < self.valid_from:
            return False
        if self.valid_to is not None and when >= self.valid_to:
            return False
        return True

    def is_current(self, at_system_time: datetime | None = None) -> bool:
        t = at_system_time or utcnow()
        if self.superseded_at is not None and t >= self.superseded_at:
            return False
        return self.is_valid_at(t)

    def overlaps(self, other: BitemporalInterval) -> bool:
        if not self.is_valid_at(other.valid_from):
            return False
        if other.valid_to is not None and not self.is_valid_at(other.valid_to - timedelta(microseconds=1)):
            if other.valid_to <= self.valid_from:
                return False
        return True

    def close_valid(self, at: datetime) -> BitemporalInterval:
        return BitemporalInterval(
            valid_from=self.valid_from,
            valid_to=at,
            recorded_at=self.recorded_at,
            superseded_at=self.superseded_at,
        )


def make_current_interval(valid_from: datetime | None = None) -> BitemporalInterval:
    return BitemporalInterval(valid_from=valid_from or utcnow())


def make_bounded_interval(
    valid_from: datetime,
    valid_to: datetime,
) -> BitemporalInterval:
    return BitemporalInterval(valid_from=valid_from, valid_to=valid_to)


def freshness_seconds(valid_from: datetime, now: datetime | None = None) -> float:
    t = now or utcnow()
    return max(0.0, (t - valid_from).total_seconds())
