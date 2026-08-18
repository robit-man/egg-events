"""OCR V2 job ledger, readiness state machine, refinement policy, dedup, backfill."""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Perceptual hash (simple DCT-based pHash for dedup)
# ---------------------------------------------------------------------------


def image_phash(image_png: bytes, hash_size: int = 8) -> str | None:
    """Compute a perceptual hash of a PNG image.

    Returns a hex string of hash_size*hash_size bits, or None if decoding fails.
    Uses a simplified approach: downsample to hash_size x hash_size grayscale,
    compute mean, return binary hash.
    """
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None

    buf = np.frombuffer(image_png, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None or img.size == 0:
        return None

    resized = cv2.resize(img, (hash_size + 1, hash_size), interpolation=cv2.INTER_AREA)
    # Compute horizontal differences
    diff = resized[:, 1:] > resized[:, :-1]
    bits = 0
    for row in range(hash_size):
        for col in range(hash_size - 1):
            if diff[row, col]:
                bits |= 1 << (row * (hash_size - 1) + col)
    total_bits = hash_size * (hash_size - 1)
    return format(bits, f"0{total_bits // 4}x")


def should_skip_dedup(
    phash: str | None,
    seen: dict[str, float],
    window_seconds: float,
    now: float | None = None,
) -> bool:
    """Return True if this phash was seen within the dedup window."""
    if phash is None:
        return False
    now = now if now is not None else time.monotonic()
    last_seen = seen.get(phash)
    if last_seen is not None and (now - last_seen) < window_seconds:
        return True
    seen[phash] = now
    return False


# ---------------------------------------------------------------------------
# Readiness state machine
# ---------------------------------------------------------------------------


class OcrReadiness(str, Enum):
    DISABLED = "disabled"
    PROBING = "probing"
    READY = "ready"
    DEGRADED_LOCAL_ONLY = "degraded_local_only"
    UNAVAILABLE = "unavailable"
    COOLDOWN = "cooldown"


@dataclass
class OcrReadinessTracker:
    """Tracks OCR engine readiness via health probes and live outcomes."""

    state: OcrReadiness = OcrReadiness.PROBING
    local_healthy: bool = False
    omnius_healthy: bool = False
    _last_probe_at: float = 0.0
    _consecutive_failures: int = 0
    _cooldown_until: float = 0.0
    _probe_interval: float = 30.0
    _cooldown_base: float = 10.0
    _cooldown_max: float = 300.0

    def note_local_success(self) -> None:
        self.local_healthy = True
        self._consecutive_failures = 0
        self._update_state()

    def note_local_failure(self) -> None:
        self.local_healthy = False
        self._consecutive_failures += 1
        self._update_state()

    def note_omnius_success(self) -> None:
        self.omnius_healthy = True
        self._consecutive_failures = 0
        self._update_state()

    def note_omnius_failure(self) -> None:
        self.omnius_healthy = False
        self._consecutive_failures += 1
        self._update_state()

    def should_probe(self, now: float | None = None) -> bool:
        now = now if now is not None else time.monotonic()
        if self.state == OcrReadiness.DISABLED:
            return False
        if self.state == OcrReadiness.COOLDOWN and now < self._cooldown_until:
            return False
        return (now - self._last_probe_at) >= self._probe_interval

    def record_probe(self, now: float | None = None) -> None:
        self._last_probe_at = now if now is not None else time.monotonic()

    def _update_state(self) -> None:
        if self.local_healthy and self.omnius_healthy:
            self.state = OcrReadiness.READY
        elif self.local_healthy:
            self.state = OcrReadiness.DEGRADED_LOCAL_ONLY
        elif self.omnius_healthy:
            self.state = OcrReadiness.DEGRADED_LOCAL_ONLY
        else:
            if self._consecutive_failures >= 5:
                self.state = OcrReadiness.COOLDOWN
                backoff = min(
                    self._cooldown_max,
                    self._cooldown_base * (2 ** min(self._consecutive_failures - 5, 6)),
                )
                self._cooldown_until = time.monotonic() + backoff
            elif self._consecutive_failures >= 2:
                self.state = OcrReadiness.UNAVAILABLE
            else:
                self.state = OcrReadiness.PROBING

    @property
    def can_use_local(self) -> bool:
        return self.state in (OcrReadiness.READY, OcrReadiness.DEGRADED_LOCAL_ONLY, OcrReadiness.PROBING)

    @property
    def can_use_omnius(self) -> bool:
        return self.state in (OcrReadiness.READY,)

    def snapshot(self) -> dict[str, object]:
        return {
            "state": self.state.value,
            "local_healthy": self.local_healthy,
            "omnius_healthy": self.omnius_healthy,
            "consecutive_failures": self._consecutive_failures,
        }


# ---------------------------------------------------------------------------
# Refinement policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OcrRefinementPolicy:
    """Decides when Omnius should refine a local OCR result."""

    local_confidence_threshold: float = 0.72
    min_text_length_for_refinement: int = 6
    max_refinements_per_minute: int = 20
    _refinement_count: int = field(default=0, repr=False)
    _window_start: float = field(default=-1.0, repr=False)

    def needs_refinement(
        self,
        local_confidence: float,
        local_text: str,
        now: float | None = None,
    ) -> bool:
        """Return True if local result is below quality threshold and budget allows."""
        now = now if now is not None else time.monotonic()
        if self._window_start < 0:
            object.__setattr__(self, "_window_start", now)
        elif (now - self._window_start) >= 60.0:
            object.__setattr__(self, "_refinement_count", 0)
            object.__setattr__(self, "_window_start", now)
        if self._refinement_count >= self.max_refinements_per_minute:
            return False
        if len(local_text) < self.min_text_length_for_refinement:
            return False
        return local_confidence < self.local_confidence_threshold

    def record_refinement(self) -> None:
        object.__setattr__(self, "_refinement_count", self._refinement_count + 1)


# ---------------------------------------------------------------------------
# OCR Job and Ledger (SQLite)
# ---------------------------------------------------------------------------


@dataclass
class OcrJob:
    job_id: str
    camera_id: str
    image_phash: str | None
    observed_at: datetime
    source_scope: str
    parent_id: str
    status: str = "pending"
    local_text: str | None = None
    local_confidence: float | None = None
    local_engine: str | None = None
    omnius_text: str | None = None
    omnius_confidence: float | None = None
    resolved_text: str | None = None
    resolved_engine: str | None = None
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None


class OcrJobLedger:
    """Durable SQLite-backed OCR job queue with idempotency."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), timeout=10)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._migrate()

    def _migrate(self) -> None:
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ocr_jobs (
                job_id TEXT PRIMARY KEY,
                camera_id TEXT NOT NULL,
                image_phash TEXT,
                observed_at TEXT NOT NULL,
                source_scope TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                local_text TEXT,
                local_confidence REAL,
                local_engine TEXT,
                omnius_text TEXT,
                omnius_confidence REAL,
                resolved_text TEXT,
                resolved_engine TEXT,
                error TEXT,
                created_at TEXT NOT NULL,
                completed_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_ocr_jobs_status ON ocr_jobs(status);
            CREATE INDEX IF NOT EXISTS idx_ocr_jobs_phash ON ocr_jobs(image_phash, observed_at);
            CREATE INDEX IF NOT EXISTS idx_ocr_jobs_camera ON ocr_jobs(camera_id, observed_at);
            """
        )
        self._conn.commit()

    def enqueue(
        self,
        camera_id: str,
        image_phash: str | None,
        observed_at: datetime,
        source_scope: str,
        parent_id: str,
    ) -> OcrJob | None:
        """Create a job if no identical phash was processed recently (idempotent)."""
        if image_phash is not None:
            row = self._conn.execute(
                "SELECT job_id FROM ocr_jobs "
                "WHERE image_phash = ? AND observed_at > ? AND status != 'failed' "
                "ORDER BY observed_at DESC LIMIT 1",
                (image_phash, (observed_at - __import__("datetime").timedelta(seconds=300)).isoformat()),
            ).fetchone()
            if row is not None:
                return None

        job_id = f"ocr:{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "INSERT INTO ocr_jobs "
            "(job_id, camera_id, image_phash, observed_at, source_scope, parent_id, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
            (job_id, camera_id, image_phash, observed_at.isoformat(), source_scope, parent_id, now),
        )
        self._conn.commit()
        return OcrJob(
            job_id=job_id,
            camera_id=camera_id,
            image_phash=image_phash,
            observed_at=observed_at,
            source_scope=source_scope,
            parent_id=parent_id,
        )

    def dequeue_pending(self, limit: int = 4) -> list[OcrJob]:
        rows = self._conn.execute(
            "SELECT job_id, camera_id, image_phash, observed_at, source_scope, parent_id "
            "FROM ocr_jobs WHERE status = 'pending' ORDER BY observed_at ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            OcrJob(
                job_id=r[0], camera_id=r[1], image_phash=r[2],
                observed_at=datetime.fromisoformat(r[3]),
                source_scope=r[4], parent_id=r[5],
            )
            for r in rows
        ]

    def update_local_result(
        self, job_id: str, text: str, confidence: float, engine: str
    ) -> None:
        self._conn.execute(
            "UPDATE ocr_jobs SET local_text = ?, local_confidence = ?, local_engine = ?, "
            "status = 'local_done' WHERE job_id = ?",
            (text, confidence, engine, job_id),
        )
        self._conn.commit()

    def update_omnius_result(
        self, job_id: str, text: str, confidence: float
    ) -> None:
        self._conn.execute(
            "UPDATE ocr_jobs SET omnius_text = ?, omnius_confidence = ? WHERE job_id = ?",
            (text, confidence, job_id),
        )
        self._conn.commit()

    def complete(
        self, job_id: str, resolved_text: str, resolved_engine: str
    ) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE ocr_jobs SET resolved_text = ?, resolved_engine = ?, "
            "status = 'done', completed_at = ? WHERE job_id = ?",
            (resolved_text, resolved_engine, now, job_id),
        )
        self._conn.commit()

    def fail(self, job_id: str, error: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self._conn.execute(
            "UPDATE ocr_jobs SET error = ?, status = 'failed', completed_at = ? WHERE job_id = ?",
            (error, now, job_id),
        )
        self._conn.commit()

    def pending_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM ocr_jobs WHERE status = 'pending'"
        ).fetchone()
        return row[0] if row else 0

    def recent_phashes(self, window_seconds: float = 300.0) -> dict[str, float]:
        """Return {phash: observed_at_timestamp} for recent completed jobs."""
        cutoff = (datetime.now(timezone.utc).isoformat())
        rows = self._conn.execute(
            "SELECT image_phash, observed_at FROM ocr_jobs "
            "WHERE image_phash IS NOT NULL AND status = 'done' "
            "ORDER BY observed_at DESC LIMIT 500",
        ).fetchall()
        result: dict[str, float] = {}
        for phash, observed_at in rows:
            try:
                ts = datetime.fromisoformat(observed_at).timestamp()
                result[phash] = ts
            except (ValueError, TypeError):
                continue
        return result

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Backfill scheduler
# ---------------------------------------------------------------------------


@dataclass
class OcrBackfillScheduler:
    """Scans retained visual evidence for unprocessed text and queues OCR jobs."""

    enabled: bool = True
    scan_interval_seconds: float = 60.0
    batch_size: int = 4
    _last_scan_at: float = -1.0

    def should_scan(self, now: float | None = None) -> bool:
        if not self.enabled:
            return False
        now = now if now is not None else time.monotonic()
        if self._last_scan_at < 0:
            return True
        return (now - self._last_scan_at) >= self.scan_interval_seconds

    def record_scan(self, now: float | None = None) -> None:
        self._last_scan_at = now if now is not None else time.monotonic()

    def find_unprocessed_evidence(
        self,
        memory_store: Any,
        job_ledger: OcrJobLedger,
        limit: int = 4,
    ) -> list[dict[str, object]]:
        """Query memory store for recent visual evidence without OCR results."""
        if memory_store is None:
            return []
        try:
            rows = memory_store.db.execute(
                "SELECT evidence_id, media_key, recorded_at, camera_id "
                "FROM evidence "
                "WHERE modality = 'video' "
                "AND evidence_id NOT IN ("
                "  SELECT DISTINCT json_extract(metadata, '$.source_evidence_id') "
                "  FROM evidence WHERE modality = 'ocr'"
                ") "
                "ORDER BY recorded_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        except Exception as error:
            logger.debug("backfill scan failed: %s", error)
            return []
        return [
            {"evidence_id": r[0], "media_key": r[1], "recorded_at": r[2], "camera_id": r[3]}
            for r in rows
        ]
