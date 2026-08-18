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
    parse_utc_datetime,
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

    def test_short_text_still_refines_when_vlm_confirms_text_present(self) -> None:
        """Length is a cost heuristic, not an epistemic gate — a short,
        low-confidence read like "STOP" must still be refinable when the
        VLM independently confirms text is present."""
        policy = OcrRefinementPolicy(
            local_confidence_threshold=0.72,
            min_text_length_for_refinement=6,
        )
        assert policy.needs_refinement(0.40, "STOP") is False
        assert policy.needs_refinement(
            0.40, "STOP", vlm_text_positive=True,
        ) is True

    def test_long_confident_text_not_forced_to_refine_by_vlm_alone(self) -> None:
        """The vlm_text_positive escalation only applies to short reads —
        a long, confident local read shouldn't get re-refined just because
        the VLM also noticed text somewhere in frame."""
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(
            0.95, "this is a long and confident local ocr read", vlm_text_positive=True,
        ) is False

    def test_explicit_read_request_always_refines(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(
            0.99, "high confidence long text here", explicit_read_request=True,
        ) is True

    def test_local_disagreement_triggers_refinement(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(
            0.95, "some fairly long text", local_disagreement=0.5,
        ) is True

    def test_dynamic_display_below_threshold_refines_even_if_short(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(
            0.50, "42", dynamic_display=True,
        ) is True

    def test_dynamic_display_high_confidence_still_skips(self) -> None:
        policy = OcrRefinementPolicy(local_confidence_threshold=0.72)
        assert policy.needs_refinement(
            0.95, "42", dynamic_display=True,
        ) is False


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

    def test_dedup_is_scoped_by_target_not_just_phash(self, tmp_path: Path) -> None:
        """A phash collision between two different targets must not cause
        the second target's job to be silently skipped — idempotency needs
        to be scoped by which target/region this is, not just image hash."""
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        now = datetime.now(timezone.utc)
        job_a = ledger.enqueue("cam-0", "sameHash", now, "frame", "object:sign-a")
        job_b = ledger.enqueue("cam-0", "sameHash", now, "frame", "object:sign-b")
        assert job_a is not None
        assert job_b is not None
        assert job_a.job_id != job_b.job_id
        # But re-enqueueing the *same* target with the same hash is still deduped.
        job_a_again = ledger.enqueue("cam-0", "sameHash", now, "frame", "object:sign-a")
        assert job_a_again is None
        ledger.close()

    def test_source_evidence_id_round_trips(self, tmp_path: Path) -> None:
        ledger = OcrJobLedger(tmp_path / "test.sqlite3")
        job = ledger.enqueue(
            "cam-0", None, datetime.now(timezone.utc), "backfill", "camera_view:cam-0",
            source_evidence_id="evidence:abc123",
        )
        assert job.source_evidence_id == "evidence:abc123"
        pending = ledger.dequeue_pending()
        assert pending[0].source_evidence_id == "evidence:abc123"
        ledger.close()


class TestParseUtcDatetime:
    """parse_utc_datetime distinguishes observation time from processing time."""

    def test_passthrough_aware_datetime(self) -> None:
        dt = datetime(2026, 1, 1, tzinfo=timezone.utc)
        assert parse_utc_datetime(dt) == dt

    def test_naive_datetime_assumed_utc(self) -> None:
        dt = datetime(2026, 1, 1)
        result = parse_utc_datetime(dt)
        assert result.tzinfo is not None
        assert result.hour == 0

    def test_iso_string(self) -> None:
        result = parse_utc_datetime("2026-08-16T14:03:22+00:00")
        assert result == datetime(2026, 8, 16, 14, 3, 22, tzinfo=timezone.utc)

    def test_sqlite_style_string(self) -> None:
        # SQLite round-trips datetimes as space-separated text, not ISO 'T'.
        result = parse_utc_datetime("2026-08-16 14:03:22.123456+00:00")
        assert result.year == 2026 and result.hour == 14

    def test_none_falls_back_to_now(self) -> None:
        before = datetime.now(timezone.utc)
        result = parse_utc_datetime(None)
        after = datetime.now(timezone.utc)
        assert before <= result <= after


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

    @staticmethod
    def _memory_store(tmp_path: Path):
        from egg_companion.config import EggConfig
        from egg_companion.memory.store import MemoryStore
        config = EggConfig.model_validate(
            {
                "audio": {"input_device": "default", "doa_mode": "disabled"},
                "omnius": {"model": "test", "voice_model": "test"},
                "identity": {"enabled": False},
                "object_learning": {"enabled": False},
                "camera_discovery": {"enabled": False},
                "memory": {"storage_dir": str(tmp_path / "memory")},
            }
        )
        return MemoryStore(config.memory)

    def test_finds_vision_evidence_with_no_ocr_row(self, tmp_path: Path) -> None:
        from egg_companion.models import EvidenceRef
        store = self._memory_store(tmp_path)
        now = datetime.now(timezone.utc)
        store.append_evidence(EvidenceRef(
            "vis-1", "vision", now, "camera", "cam0",
            media_key="vision/1.jpg", metadata={},
        ))
        sched = OcrBackfillScheduler()
        found = sched.find_unprocessed_evidence(store, job_ledger=None)
        assert {item["evidence_id"] for item in found} == {"vis-1"}

    def test_excludes_evidence_with_matching_ocr_source_evidence_id(self, tmp_path: Path) -> None:
        """The JOIN condition must match the real (flat) evidence.payload_json
        shape — evidence.metadata is written directly as payload_json, not
        nested under a "metadata" key — or this exclusion silently never
        matches anything."""
        from egg_companion.models import EvidenceRef
        store = self._memory_store(tmp_path)
        now = datetime.now(timezone.utc)
        store.append_evidence(EvidenceRef(
            "vis-1", "vision", now, "camera", "cam0",
            media_key="vision/1.jpg", metadata={},
        ))
        store.append_evidence(EvidenceRef(
            "ocr-1", "ocr", now, "camera-advanced-ocr", "cam0",
            metadata={"source_evidence_id": "vis-1", "text": "hi"},
        ))
        sched = OcrBackfillScheduler()
        found = sched.find_unprocessed_evidence(store, job_ledger=None)
        assert found == []

    def test_one_processed_evidence_row_does_not_hide_other_unprocessed_rows(
        self, tmp_path: Path
    ) -> None:
        """Regression test: `x NOT IN (subquery)` is NULL (excludes
        everything) if the subquery yields even one NULL — e.g. an OCR
        evidence row whose source_evidence_id extraction fails.  A single
        such row must not make every other vision evidence row invisible
        to backfill forever."""
        from egg_companion.models import EvidenceRef
        store = self._memory_store(tmp_path)
        now = datetime.now(timezone.utc)
        store.append_evidence(EvidenceRef(
            "vis-1", "vision", now, "camera", "cam0",
            media_key="vision/1.jpg", metadata={},
        ))
        store.append_evidence(EvidenceRef(
            "vis-2", "vision", now, "camera", "cam0",
            media_key="vision/2.jpg", metadata={},
        ))
        # OCR row that resolved e.g. from a live (non-backfill) capture and
        # therefore carries no source_evidence_id at all.
        store.append_evidence(EvidenceRef(
            "ocr-1", "ocr", now, "camera-advanced-ocr", "cam0",
            metadata={"text": "hi"},
        ))
        store.append_evidence(EvidenceRef(
            "ocr-2", "ocr", now, "camera-advanced-ocr", "cam0",
            metadata={"source_evidence_id": "vis-1", "text": "bye"},
        ))
        sched = OcrBackfillScheduler()
        found = sched.find_unprocessed_evidence(store, job_ledger=None)
        assert {item["evidence_id"] for item in found} == {"vis-2"}


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

    def test_payload_target_id_takes_priority_over_entity_ids(self) -> None:
        """OCR events carry both a physical target (target_id) and content
        entities (entity_ids may include a content: id for the recognized
        text) — the semantic subject of visible_text must be the explicit
        target_id, not whichever entity happened to be listed first."""
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera-0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.payload = {"text": "ROOM 312", "target_id": "object:door-sign"}
        event.entity_ids = ("content:abcdef123456",)
        delta = normalizer.normalize_event(event)
        assert delta.assertions[0]["subject_id"] == "object:door-sign"

    def test_full_queue_ocr_memory_style_payload_reaches_visible_text(self) -> None:
        """Regression test for the OCR V2 integration bug: _queue_ocr_memory
        must publish a payload shape the normalizer can actually turn into
        world state (text/target_id/confidence/engine/regions/text_type),
        not just labels/entities/relations for the associative memory graph.
        """
        normalizer = ObservationNormalizer()
        event = MagicMock()
        event.event_type = "ocr"
        event.source_id = "camera:cam0"
        event.occurred_at = datetime(2025, 1, 1, tzinfo=timezone.utc)
        event.entity_ids = ("object:door-sign", "content:abc123")
        event.payload = {
            "text": "ROOM 312",
            "target_id": "object:door-sign",
            "text_type": "static",
            "ocr_confidence": 0.87,
            "ocr_engine": "omnius-advanced-ocr",
            "regions": [{"text": "ROOM 312", "bbox": [0, 0, 10, 10]}],
            "scope": "frame",
            "trigger": "scheduled",
            "labels": ["ocr", "door sign"],
            "entities": [{"id": "object:door-sign", "type": "physical_object"}],
            "relations": [{"source_id": "object:door-sign", "relation": "contains_text",
                            "target_id": "content:abc123"}],
            "skip_pairwise_co_observation": True,
        }
        delta = normalizer.normalize_event(event, evidence_ids=("ev:1",))
        assert len(delta.assertions) == 1
        assertion = delta.assertions[0]
        assert assertion["subject_id"] == "object:door-sign"
        assert assertion["property_id"] == "visible_text"
        assert assertion["value"].raw == "ROOM 312"
        assert assertion["confidence"] == pytest.approx(0.87)
        assert delta.events[0]["confidence"] == pytest.approx(0.87)


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
