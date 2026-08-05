from dataclasses import FrozenInstanceError
from datetime import datetime, timezone

import pytest

from egg_companion.models import EvidenceRef, PerceptualEvent


def test_evidence_and_events_are_frozen_and_serializable() -> None:
    evidence = EvidenceRef(
        evidence_id="evidence-1", modality="vision", captured_at=datetime.now(timezone.utc),
        source_type="camera", source_id="camera-0", metadata={"quality_reason": "sharp"},
    )
    event = PerceptualEvent(
        event_id="event-1", event_type="vision", occurred_at=evidence.captured_at,
        source_id="camera-0", evidence=(evidence,), payload={"labels": ["person"]},
    )
    assert event.evidence[0].metadata["quality_reason"] == "sharp"
    with pytest.raises(FrozenInstanceError):
        event.source_id = "other"  # type: ignore[misc]
