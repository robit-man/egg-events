from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from egg_companion.config import PrivacyConfig
from egg_companion.memory.store import MemoryStore


@dataclass(frozen=True)
class RetentionPlan:
    media_cutoff: datetime
    evidence_cutoff: datetime
    batch_size: int


class RetentionPlanner:
    """Deterministic retention plan shared by verification and consolidation."""

    def __init__(self, store: MemoryStore, privacy: PrivacyConfig) -> None:
        self.store = store
        self.privacy = privacy

    def plan(self, now: datetime | None = None) -> RetentionPlan:
        now = now or datetime.now(timezone.utc)
        return RetentionPlan(
            now - timedelta(hours=self.store.config.raw_media_retention_hours),
            now - timedelta(days=self.privacy.evidence_retention_days),
            self.store.config.consolidation_batch_size,
        )

    def execute(self, now: datetime | None = None) -> dict[str, int]:
        plan = self.plan(now)
        return {
            "expired_media": self.store.expire_media_before(
                plan.media_cutoff, plan.batch_size
            ),
            "expired_evidence": self.store.delete_evidence_before(
                plan.evidence_cutoff, plan.batch_size
            ),
        }
