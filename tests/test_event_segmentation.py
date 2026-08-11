from datetime import datetime, timedelta, timezone

from egg_companion.config import EventSegmentationConfig, MemoryConfig
from egg_companion.memory.segmentation import EventSegmenter
from egg_companion.models import EvidenceRef, PerceptualEvent


def event(
    event_id: str,
    at: datetime,
    event_type: str = "vision",
    labels: tuple[str, ...] = ("person",),
    scene_labels: tuple[str, ...] = (),
    entity_ids: tuple[str, ...] = (),
) -> PerceptualEvent:
    evidence = EvidenceRef(event_id, event_type, at, "camera", "camera-0")
    return PerceptualEvent(
        event_id,
        event_type,
        at,
        "camera-0",
        (evidence,),
        entity_ids,
        payload={"labels": labels, "scene_labels": scene_labels},
    )


def test_identical_frames_do_not_create_episodes() -> None:
    start = datetime.now(timezone.utc)
    segmenter = EventSegmenter(MemoryConfig(), EventSegmentationConfig(inactivity_seconds=8))
    accepted, closed = segmenter.ingest(event("first", start))
    assert accepted and not closed
    for index in range(1, 100):
        accepted, closed = segmenter.ingest(event(str(index), start + timedelta(milliseconds=100 * index)))
        assert not accepted
        assert not closed
    assert len(segmenter.flush(start + timedelta(seconds=11))) == 1
    assert segmenter.snapshot()["accepted_by_context"] == {"camera-0": 1}


def test_speech_is_accepted_and_scene_change_closes_boundary() -> None:
    start = datetime.now(timezone.utc)
    segmenter = EventSegmenter(MemoryConfig(), EventSegmentationConfig())
    segmenter.ingest(event("vision", start))
    accepted, closed = segmenter.ingest(event("speech", start + timedelta(seconds=1), "speech", ()))
    assert accepted and not closed
    for offset in (2, 3):
        accepted, closed = segmenter.ingest(
            event(
                f"object-{offset}", start + timedelta(seconds=offset),
                labels=("mug",), scene_labels=("object introduced",),
            )
        )
        assert not accepted and not closed
    accepted, closed = segmenter.ingest(
        event(
            "object-4", start + timedelta(seconds=4), labels=("mug",),
            scene_labels=("object introduced",),
        )
    )
    assert accepted and len(closed) == 1
    assert segmenter.snapshot()["accepted_by_context"]["camera-0"] == 2


def test_transient_detector_labels_do_not_split_episode() -> None:
    start = datetime.now(timezone.utc)
    segmenter = EventSegmenter(MemoryConfig(), EventSegmentationConfig())
    segmenter.ingest(event("initial", start, labels=("table",)))
    for index, label in enumerate(("artist", "window", "fabric", "screen"), start=1):
        accepted, closed = segmenter.ingest(event(str(index), start + timedelta(seconds=index), labels=(label,)))
        assert not accepted and not closed
    assert len(segmenter.flush(start + timedelta(seconds=5))) == 1


def test_single_frame_entity_blink_does_not_split_static_episode() -> None:
    start = datetime.now(timezone.utc)
    segmenter = EventSegmenter(MemoryConfig(), EventSegmentationConfig())
    segmenter.ingest(event("initial", start, entity_ids=("object-1",)))

    accepted, closed = segmenter.ingest(
        event("missed-mask", start + timedelta(seconds=1), entity_ids=())
    )
    assert not accepted and not closed
    accepted, closed = segmenter.ingest(
        event("recovered", start + timedelta(seconds=2), entity_ids=("object-1",))
    )
    assert not accepted and not closed
    assert segmenter.snapshot()["accepted_by_context"] == {"camera-0": 1}


def test_object_recall_ids_remain_evidence_but_do_not_drive_visual_boundaries() -> None:
    start = datetime.now(timezone.utc)
    segmenter = EventSegmenter(MemoryConfig(), EventSegmentationConfig())
    first = event("first", start, labels=("keyboard",), entity_ids=("object-1",))
    first.payload["boundary_entity_ids"] = []
    segmenter.ingest(first)
    for offset, object_id in enumerate(("object-2", "object-1", "object-3"), start=1):
        changed = event(
            f"object-{offset}", start + timedelta(seconds=offset),
            labels=("monitor",), entity_ids=(object_id,),
        )
        changed.payload["boundary_entity_ids"] = []
        accepted, closed = segmenter.ingest(changed)
        assert not accepted and not closed
    assert segmenter.snapshot()["accepted_by_context"] == {"camera-0": 1}
