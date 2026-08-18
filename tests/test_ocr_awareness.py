from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import BoundingBox, Detection, EvidenceRef, PerceptualEvent
from egg_companion.runtime import CompanionRuntime, _OcrCandidate, _OcrTarget


def _config(tmp_path) -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
        }
    )


def test_ocr_fragments_preserve_nested_content(tmp_path) -> None:
    runtime = object.__new__(CompanionRuntime)
    runtime.config = _config(tmp_path)

    assert runtime._ocr_fragments("First heading\nSecond nested line. Final line.", 8) == [
        "First heading",
        "Second nested line.",
        "Final line.",
    ]


def test_ocr_mask_parent_tracks_one_text_object_without_grouping_all_screens(tmp_path) -> None:
    runtime = object.__new__(CompanionRuntime)
    runtime.config = _config(tmp_path)
    runtime._ocr_mask_tracks = {}
    first = Detection("monitor", 0.9, BoundingBox(10, 10, 210, 110))
    same = Detection("monitor", 0.9, BoundingBox(15, 12, 215, 112))
    other = Detection("monitor", 0.9, BoundingBox(300, 10, 500, 110))

    first_id = runtime._ocr_parent_for_detection("camera-0", first, 10.0)
    same_id = runtime._ocr_parent_for_detection("camera-0", same, 11.0)
    other_id = runtime._ocr_parent_for_detection("camera-0", other, 11.0)

    assert same_id == first_id
    assert other_id != first_id
    assert first_id.startswith("visual-mask:camera-0:")


def test_tesseract_tsv_parser_keeps_confident_nested_lines_and_rejects_noise() -> None:
    header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"
    rows = [
        "5\t1\t1\t1\t1\t1\t10\t10\t90\t20\t92\tWELCOME",
        "5\t1\t1\t1\t2\t1\t10\t40\t40\t20\t88\tGate",
        "5\t1\t1\t1\t2\t2\t55\t40\t15\t20\t91\t3",
        "5\t1\t1\t1\t3\t1\t10\t70\t10\t20\t12\tx",
    ]

    parsed = CompanionRuntime._parse_tesseract_tsv("\n".join([header, *rows]))

    assert parsed is not None
    assert parsed["text"] == "WELCOME\nGate 3"
    assert len(parsed["regions"]) == 2
    noisy = "\n".join(
        [header, "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t70\tMG"]
    )
    assert CompanionRuntime._parse_tesseract_tsv(noisy) is None

    weak_short_fragments = "\n".join(
        [
            header,
            "5\t1\t1\t1\t1\t1\t0\t0\t30\t12\t62\tdad",
            "5\t1\t1\t1\t2\t1\t0\t20\t40\t12\t66\tvier",
        ]
    )
    assert CompanionRuntime._parse_tesseract_tsv(weak_short_fragments) is None

    confident_single_label = "\n".join(
        [header, "5\t1\t1\t1\t1\t1\t0\t0\t100\t20\t93\tOCULUS"]
    )
    assert CompanionRuntime._parse_tesseract_tsv(confident_single_label) is not None


def test_pixel_text_regions_attach_to_smallest_nested_mask_without_label_filter() -> None:
    person = _OcrTarget(
        "person-1", "person", "Sam", 0.9, (100, 50, 900, 950)
    )
    unknown_item = _OcrTarget(
        "visual-mask-1",
        "object_category",
        "unseen thing",
        0.8,
        (300, 300, 700, 650),
        ((300, 300), (700, 300), (700, 650), (300, 650)),
    )
    candidate = _OcrCandidate(
        "camera-0",
        b"frame",
        datetime.now(timezone.utc),
        "frame",
        "scene:camera-0",
        "object_category",
        "camera-0 scene",
        0.55,
        source_size=(1000, 1000),
        targets=(person, unknown_item),
    )

    associations = CompanionRuntime._associate_ocr_regions(
        candidate,
        {
            "image_size": [500, 500],
            "regions": [
                {"text": "ORBITAL", "confidence": 0.94, "bbox": [175, 190, 275, 220]}
            ],
        },
    )

    assert len(associations) == 1
    target, regions = associations[0]
    assert target.parent_id == "visual-mask-1"
    assert regions[0]["bbox"] == [350.0, 380.0, 550.0, 440.0]


def test_ocr_event_persists_parent_content_fragments_and_explicit_relations(tmp_path) -> None:
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    evidence = EvidenceRef(
        "ocr-evidence",
        "ocr",
        now,
        "camera-advanced-ocr",
        "camera-0",
        quality=0.88,
        metadata={"text": "WELCOME\nGate 3"},
    )
    event = PerceptualEvent(
        "ocr-event",
        "ocr",
        now,
        "camera-0",
        (evidence,),
        ("object-screen", "content-block", "content-fragment"),
        payload={
            "skip_pairwise_co_observation": True,
            "entities": [
                {"id": "object-screen", "type": "object", "label": "screen", "confidence": 0.88},
                {"id": "content-block", "type": "content", "label": "WELCOME Gate 3", "confidence": 0.88},
                {"id": "content-fragment", "type": "content", "label": "Gate 3", "confidence": 0.88},
            ],
            "relations": [
                {"source_id": "object-screen", "relation": "contains_text", "target_id": "content-block", "confidence": 0.88},
                {"source_id": "content-block", "relation": "contains_fragment", "target_id": "content-fragment", "confidence": 0.88},
            ],
        },
    )

    pipeline._persist_event(event)
    graph = pipeline.knowledge_graph_snapshot()
    relations = {link["relation"] for link in graph["links"]}

    assert {"contains_text", "contains_fragment"} <= relations
    assert "co_observed_with" not in relations
    assert any(node["subtype"] == "content" for node in graph["nodes"])
    assert any(node["subtype"] == "ocr" and node["label"] == "WELCOME\nGate 3" for node in graph["nodes"])


def test_ocr_event_with_full_payload_reaches_world_model_visible_text(tmp_path) -> None:
    """End-to-end regression test for the OCR V2 integration bug: an OCR
    PerceptualEvent shaped like what _queue_ocr_memory actually publishes
    (text/target_id/confidence/engine/regions, plus the entities/relations
    used by the associative memory graph) must result in a queryable
    visible_text property in the operational world model — not just an
    associative-graph edge."""
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    evidence = EvidenceRef(
        "ocr-evidence-2",
        "ocr",
        now,
        "camera-advanced-ocr",
        "camera-0",
        quality=0.9,
        metadata={"text": "ROOM 312", "parent_id": "object-door-sign"},
    )
    event = PerceptualEvent(
        "ocr-event-2",
        "ocr",
        now,
        "camera-0",
        (evidence,),
        ("object-door-sign", "content-room312"),
        payload={
            "text": "ROOM 312",
            "target_id": "object-door-sign",
            "text_type": "static",
            "ocr_confidence": 0.9,
            "ocr_engine": "omnius-advanced-ocr",
            "regions": [],
            "scope": "frame",
            "trigger": "scheduled",
            "labels": ["ocr", "door sign"],
            "entities": [
                {"id": "object-door-sign", "type": "object", "label": "door sign", "confidence": 0.9},
                {"id": "content-room312", "type": "content", "label": "ROOM 312", "confidence": 0.9},
            ],
            "relations": [
                {"source_id": "object-door-sign", "relation": "contains_text", "target_id": "content-room312", "confidence": 0.9},
            ],
            "skip_pairwise_co_observation": True,
        },
    )

    pipeline._persist_event(event)

    assert pipeline._world_query is not None
    value = pipeline._world_query.property_value("object-door-sign", "visible_text")
    assert value == "ROOM 312"


def test_vision_event_with_detections_reaches_world_model(tmp_path) -> None:
    """End-to-end regression test: a live-camera vision PerceptualEvent
    shaped like what _queue_vision_memory actually publishes (a top-level
    "detections" list, not just "entities") must populate the operational
    world model — this is what the ObservationNormalizer's _normalize_
    visual_event has always required, but the live pipeline never sent it
    before this fix, so det:* labels never reached current_property_state
    outside of the one-time startup backfill."""
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    now = datetime.now(timezone.utc)
    evidence = EvidenceRef(
        "vision-evidence-1",
        "vision",
        now,
        "camera",
        "camera-0",
        quality=0.9,
        metadata={"detections": []},
    )
    event = PerceptualEvent(
        "vision-event-1",
        "vision",
        now,
        "camera-0",
        (evidence,),
        (),
        payload={
            "detections": [
                {"label": "mug", "confidence": 0.91, "bbox": {"x1": 0, "y1": 0, "x2": 50, "y2": 50}},
            ],
            "frame_shape": [480, 640],
            "labels": ["mug"],
            "scene_labels": [],
            "behaviors": [],
            "boundary_entity_ids": [],
            "boundary_behaviors": [],
            "entities": [],
        },
    )

    pipeline._persist_event(event)

    assert pipeline._world_query is not None
    value = pipeline._world_query.property_value("det:mug", "label")
    assert value == "mug"
