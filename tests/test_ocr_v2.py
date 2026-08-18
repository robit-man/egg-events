"""OCR V2 regression tests — independent proposers, resolution, readiness, dedup."""

from __future__ import annotations

import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from egg_companion.ocr.jobs import (
    OcrBackfillScheduler,
    OcrJobLedger,
    OcrReadiness,
    OcrReadinessTracker,
    OcrRefinementPolicy,
    image_phash,
    should_skip_dedup,
)
from egg_companion.ocr.resolve import OcrResolution, resolve_text_observations
from egg_companion.world.normalize import ObservationNormalizer
from egg_companion.world.types import EpistemicKind, TypedValue, ValueType, WorldDelta


# ---------------------------------------------------------------------------
# Resolution layer
# ---------------------------------------------------------------------------


class TestResolveTextObservations:
    """resolve_text_observations merges local and Omnius results."""

    def test_local_only(self) -> None:
        local = {"text": "HELLO WORLD", "confidence": 0.85, "engine": "local-tesseract-multipass"}
        result = resolve_text_observations(local, None)
        assert result is not None
        assert result.text == "HELLO WORLD"
        assert result.engine == "local-tesseract-multipass"
        assert result.local_text == "HELLO WORLD"
        assert result.omnius_text is None
        assert result.source_count == 1

    def test_omnius_only(self) -> None:
        omnius = {"text": "OPENAI 4.0", "confidence": 0.92, "engine": "omnius-ocr-image-advanced"}
        result = resolve_text_observations(None, omnius)
        assert result is not None
        assert result.text == "OPENAI 4.0"
        assert result.engine == "omnius-ocr"
        assert result.omnius_text == "OPENAI 4.0"
        assert result.local_text is None
        assert result.source_count == 1

    def test_neither_has_text(self) -> None:
        local = {"text": "", "confidence": 0.3, "engine": "local-tesseract-multipass"}
        omnius = {"text": "", "confidence": 0.2, "engine": "omnius-ocr"}
        result = resolve_text_observations(local, omnius)
        assert result is None

    def test_both_agree_high_similarity(self) -> None:
        """When both agree (jaccard > 0.6), use higher confidence."""
        local = {"text": "WELCOME GATE 3", "confidence": 0.68, "engine": "local-tesseract-multipass"}
        omnius = {"text": "WELCOME GATE 3", "confidence": 0.91, "engine": "omnius-ocr-image-advanced"}
        result = resolve_text_observations(local, omnius)
        assert result is not None
        assert result.text == "WELCOME GATE 3"
        assert result.source_count == 2
        # Omnius confidence is higher → chosen
        assert result.engine == "omnius-ocr"
        assert result.confidence == pytest.approx(0.91)

    def test_both_disagree_omnius_confident(self) -> None:
        """When they disagree but Omnius is confident, prefer Omnius."""
        local = {"text": "GATE THREE", "confidence": 0.65, "engine": "local-tesseract-multipass"}
        omnius = {"text": "GATE 3 NORTH", "confidence": 0.88, "engine": "omnius-ocr-image-advanced"}
        result = resolve_text_observations(local, omnius, confidence_threshold=0.72)
        assert result is not None
        assert result.text == "GATE 3 NORTH"
        assert result.engine == "omnius-ocr"

    def test_both_disagree_local_preferred(self) -> None:
        """When they disagree and Omnius is not confident enough, keep local."""
        local = {"text": "NOTICE BOARD", "confidence": 0.80, "engine": "local-tesseract-multipass"}
        omnius = {"text": "NOTI EBOA RD", "confidence": 0.55, "engine": "omnius-ocr-image-advanced"}
        result = resolve_text_observations(local, omnius, confidence_threshold=0.72)
        assert result is not None
        assert result.text == "NOTICE BOARD"
        assert result.engine == "local-tesseract-multipass"


# ---------------------------------------------------------------------------
# Readiness state machine
# ---------------------------------------------------------------------------


class TestOcrReadinessTracker:
    """OcrReadinessTracker transitions through states based on outcomes."""

    def test_initial_state_is_probing(self) -> None:
        tracker = OcrReadinessTracker()
        assert tracker.state == OcrReadiness.PROBING

    def test_local_success_moves_to_degraded(self) -> None:
        tracker = OcrReadinessTracker()
        tracker.note_local_success()
        assert tracker.state == OcrReadiness.DEGRADED_LOCAL_ONLY
        assert tracker.can_use_local is True
        assert tracker.can_use_omnius is False

    def test_both_success_moves_to_ready(self) -> None:
        tracker = OcrReadinessTracker()
        tracker.note_local_success()
        tracker.note_omnius_success()
        assert tracker.state == OcrReadiness.READY
        assert tracker.can_use_local is True
        assert tracker.can_use_omnius is True

    def test_consecutive_failures_trigger_cooldown(self) -> None:
        tracker = OcrReadinessTracker()
        for _ in range(6):
            tracker.note_local_failure()
            tracker.note_omnius_failure()
        assert tracker.state == OcrReadiness.COOLDOWN

    def test_success_resets_failure_count(self) -> None:
        tracker = OcrReadinessTracker()
        for _ in range(3):
            tracker.note_local_failure()
        assert tracker._consecutive_failures == 3
        tracker.note_local_success()
        assert tracker._consecutive_failures == 0
        assert tracker.state != OcrReadiness.UNAVAILABLE

    def test_snapshot_includes_all_fields(self) -> None:
        tracker = OcrReadinessTracker()
        snap = tracker.snapshot()
        assert "state" in snap
        assert "local_healthy" in snap
        assert "omnius_healthy" in snap
        assert "consecutive_failures" in snap


# ---------------------------------------------------------------------------
# Refinement policy
# ---------------------------------------------------------------------------


class TestOcrRefinementPolicy:
    """OcrRefinementPolicy decides when Omnius should refine local results."""

    def test_high_confidence_no_refinement(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(0.90, "HELLO WORLD") is False

    def test_low_confidence_triggers_refinement(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(0.50, "HELLO WORLD") is True

    def test_short_text_no_refinement(self) -> None:
        policy = OcrRefinementPolicy(
            local_confidence_threshold=0.72,
            min_text_length_for_refinement=6,
        )
        assert policy.needs_refinement(0.40, "HI") is False

    def test_budget_exhausted(self) -> None:
        policy = OcrRefinementPolicy(
            local_confidence_threshold=0.72,
            max_refinements_per_minute=2,
        )
        policy.record_refinement()
        policy.record_refinement()
        assert policy.needs_refinement(0.30, "ENOUGH TEXT HERE") is False


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    """Perceptual hash dedup skips identical images."""

    def test_same_phash_skipped_within_window(self) -> None:
        seen: dict[str, float] = {}
        assert should_skip_dedup("abc123", seen, 300.0, now=100.0) is False
        assert should_skip_dedup("abc123", seen, 300.0, now=200.0) is True

    def test_different_phash_not_skipped(self) -> None:
        seen: dict[str, float] = {}
        assert should_skip_dedup("abc123", seen, 300.0, now=100.0) is False
        assert should_skip_dedup("def456", seen, 300.0, now=200.0) is False

    def test_phash_expires_after_window(self) -> None:
        seen: dict[str, float] = {}
        assert should_skip_dedup("abc123", seen, 300.0, now=100.0) is False
        assert should_skip_dedup("abc123", seen, 300.0, now=500.0) is False

    def test_none_phash_never_skipped(self) -> None:
        seen: dict[str, float] = {}
        assert should_skip_dedup(None, seen, 300.0) is False


# ---------------------------------------------------------------------------
# Job ledger
# ---------------------------------------------------------------------------


class TestOcrJobLedger:
    """OcrJobLedger persists OCR jobs with idempotency."""

    def test_enqueue_and_dequeue(self, tmp_path: Path) -> None:
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        job = ledger.enqueue(
            camera_id="cam-0",
            image_phash="abc123",
            observed_at=datetime.now(timezone.utc),
            source_scope="frame",
            parent_id="scene:cam-0",
        )
        assert job is not None
        pending = ledger.dequeue_pending()
        assert len(pending) == 1
        assert pending[0].job_id == job.job_id
        ledger.close()

    def test_idempotent_enqueue(self, tmp_path: Path) -> None:
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        now = datetime.now(timezone.utc)
        job1 = ledger.enqueue("cam-0", "abc123", now, "frame", "scene:cam-0")
        assert job1 is not None
        # Second enqueue with same phash within window → returns None
        job2 = ledger.enqueue("cam-0", "abc123", now, "frame", "scene:cam-0")
        assert job2 is None
        ledger.close()

    def test_complete_and_pending_count(self, tmp_path: Path) -> None:
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        job = ledger.enqueue(
            "cam-0", None, datetime.now(timezone.utc), "frame", "scene:cam-0",
        )
        assert ledger.pending_count() == 1
        ledger.complete(job.job_id, "HELLO", "local-tesseract-multipass")
        assert ledger.pending_count() == 0
        ledger.close()

    def test_fail_records_error(self, tmp_path: Path) -> None:
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        job = ledger.enqueue(
            "cam-0", None, datetime.now(timezone.utc), "frame", "scene:cam-0",
        )
        ledger.fail(job.job_id, "tesseract not found")
        assert ledger.pending_count() == 0
        ledger.close()


# ---------------------------------------------------------------------------
# Backfill scheduler
# ---------------------------------------------------------------------------


class TestOcrBackfillScheduler:
    """OcrBackfillScheduler scans for unprocessed evidence."""

    def test_should_scan_respects_interval(self) -> None:
        sched = OcrBackfillScheduler(scan_interval_seconds=60.0)
        assert sched.should_scan(now=0.0) is True
        sched.record_scan(now=0.0)
        assert sched.should_scan(now=30.0) is False
        assert sched.should_scan(now=61.0) is True

    def test_disabled_never_scans(self) -> None:
        sched = OcrBackfillScheduler(enabled=False)
        assert sched.should_scan(now=0.0) is False


# ---------------------------------------------------------------------------
# OCR normalize displays_text
# ---------------------------------------------------------------------------


class TestOcrDisplaysTextNormalization:
    """OCR events emit displays_text for dynamic text with validity window."""

    def test_static_text_emits_visible_text(self) -> None:
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera-0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.payload = {"text": "STOP SIGN", "text_type": "static"}
        event.entity_ids = ("object:stop-sign",)
        delta = normalizer.normalize_event(event)
        assert len(delta.assertions) == 1
        assertion = delta.assertions[0]
        assert assertion["property_id"] == "visible_text"
        assert assertion["value"].raw == "STOP SIGN"
        assert "valid_for_seconds" not in assertion

    def test_dynamic_text_emits_displays_text(self) -> None:
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera-0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.payload = {"text": "12:45 PM", "text_type": "dynamic"}
        event.entity_ids = ("object:clock",)
        delta = normalizer.normalize_event(event)
        assert len(delta.assertions) == 1
        assertion = delta.assertions[0]
        assert assertion["property_id"] == "displays_text"
        assert assertion["value"].raw == "12:45 PM"
        assert assertion["valid_for_seconds"] == 30.0

    def test_dynamic_text_custom_validity(self) -> None:
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera-0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.payload = {"text": "Loading...", "text_type": "dynamic", "valid_for_seconds": 5.0}
        event.entity_ids = ("object:screen",)
        delta = normalizer.normalize_event(event)
        assertion = delta.assertions[0]
        assert assertion["valid_for_seconds"] == 5.0

    def test_empty_text_returns_empty_delta(self) -> None:
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera-0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.payload = {"text": ""}
        event.entity_ids = ()
        delta = normalizer.normalize_event(event)
        assert len(delta.assertions) == 0
        assert len(delta.events) == 0


# ---------------------------------------------------------------------------
# Ontology displays_text property
# ---------------------------------------------------------------------------


class TestOntologyDisplaysText:
    """Ontology registry includes displays_text property type."""

    def test_displays_text_in_property_types(self) -> None:
        from egg_companion.world.ontology import OntologyRegistry
        registry = OntologyRegistry()
        prop = registry._property_types.get("displays_text")
        assert prop is not None
        assert prop.stale_after == 30.0
        assert prop.decay_model == "linear"
        assert prop.volatility == "dynamic"
