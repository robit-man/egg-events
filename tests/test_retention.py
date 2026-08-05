from datetime import datetime, timezone

from egg_companion.config import MemoryConfig, PrivacyConfig
from egg_companion.memory.retention import RetentionPlanner
from egg_companion.memory.store import MemoryStore


def test_retention_plan_is_deterministic_and_non_mutating(tmp_path) -> None:
    store = MemoryStore(
        MemoryConfig(
            storage_dir=str(tmp_path), raw_media_retention_hours=12,
            consolidation_batch_size=7,
        )
    )
    now = datetime(2026, 8, 5, tzinfo=timezone.utc)

    plan = RetentionPlanner(store, PrivacyConfig(evidence_retention_days=30)).plan(now)

    assert plan.media_cutoff.isoformat() == "2026-08-04T12:00:00+00:00"
    assert plan.evidence_cutoff.isoformat() == "2026-07-06T00:00:00+00:00"
    assert plan.batch_size == 7
    assert store.memory_stats()["evidence"] == 0
