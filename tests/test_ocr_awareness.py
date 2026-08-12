from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import BoundingBox, Detection, EvidenceRef, PerceptualEvent
from egg_companion.runtime import CompanionRuntime


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


def test_ocr_configuration_recognizes_text_bearing_objects(tmp_path) -> None:
    runtime = object.__new__(CompanionRuntime)
    runtime.config = _config(tmp_path)

    assert runtime._label_implies_text("flat screen television")
    assert runtime._label_implies_text("hard-cover book")
    assert runtime._label_implies_text("store sign")
    assert runtime._label_implies_text("aluminum can")
    assert runtime._label_implies_text("water bottle")
    assert not runtime._label_implies_text("wooden chair")
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
