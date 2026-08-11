from datetime import datetime, timezone

from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.models import EvidenceRef, PerceptualEvent
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
    assert not runtime._label_implies_text("wooden chair")
    assert runtime._ocr_fragments("First heading\nSecond nested line. Final line.", 8) == [
        "First heading",
        "Second nested line.",
        "Final line.",
    ]


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
