"""OCR V2 pipeline — independent proposers, resolution layer, readiness tracking."""

from egg_companion.ocr.jobs import (
    OcrJob,
    OcrJobLedger,
    OcrBackfillScheduler,
    OcrReadiness,
    OcrReadinessTracker,
    OcrRefinementPolicy,
    image_phash,
    should_skip_dedup,
)
from egg_companion.ocr.resolve import (
    OcrResolution,
    resolve_text_observations,
)

__all__ = [
    "OcrJob",
    "OcrJobLedger",
    "OcrBackfillScheduler",
    "OcrReadiness",
    "OcrReadinessTracker",
    "OcrRefinementPolicy",
    "OcrResolution",
    "image_phash",
    "resolve_text_observations",
    "should_skip_dedup",
]
