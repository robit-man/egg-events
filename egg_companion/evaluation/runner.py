from __future__ import annotations

import json
from pathlib import Path

from egg_companion.evaluation.event_boundaries import boundary_metrics, static_scene_event_rate
from egg_companion.evaluation.identity_metrics import identity_metrics
from egg_companion.evaluation.interaction_metrics import interaction_metrics
from egg_companion.evaluation.asr_metrics import asr_grounding_metrics
from egg_companion.evaluation.object_learning_metrics import object_learning_metrics
from egg_companion.evaluation.retrieval_metrics import retrieval_metrics
from egg_companion.evaluation.runtime_metrics import runtime_metrics


def load_trace(path: str | Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as trace_file:
        trace = json.load(trace_file)
    if not isinstance(trace, dict):
        raise ValueError("evaluation trace must be a JSON object")
    return trace


def evaluate_trace(trace: dict[str, object]) -> tuple[bool, dict[str, object]]:
    boundary = trace.get("event_boundaries", {})
    if not isinstance(boundary, dict):
        raise ValueError("event_boundaries must be an object")
    event = boundary_metrics(
        [float(value) for value in boundary.get("expected", [])],
        [float(value) for value in boundary.get("predicted", [])],
        float(boundary.get("tolerance_seconds", 0.5)),
    )
    event["static_scene_events_per_minute"] = static_scene_event_rate(
        int(boundary.get("static_event_count", 0)),
        float(boundary.get("static_duration_seconds", 300)),
    )
    identity = identity_metrics(list(trace.get("identity", [])))
    retrieval = retrieval_metrics(list(trace.get("retrieval", [])))
    interaction = interaction_metrics(list(trace.get("interactions", [])))
    object_learning = object_learning_metrics(list(trace.get("object_learning", [])))
    asr = asr_grounding_metrics(list(trace.get("asr", [])))
    runtime = runtime_metrics(list(trace.get("runtime", [])))
    thresholds = {
        "boundary_f1": 0.8,
        "static_events_per_minute": 1.0,
        "false_merge_rate": 0.0,
        "provenance_coverage": 1.0,
        "answer_support_rate": 0.8,
        "unsolicited_speech_rate": 0.0,
        "correction_retention": 1.0,
        "object_provenance": 1.0,
        "object_recall_accuracy": 0.8,
        "asr_grounding": 1.0,
        **(trace.get("thresholds", {}) if isinstance(trace.get("thresholds"), dict) else {}),
    }
    gates = {
        "event_boundaries": float(event["f1"]) >= float(thresholds["boundary_f1"]),
        "static_scene": float(event["static_scene_events_per_minute"])
        <= float(thresholds["static_events_per_minute"]),
        "identity": float(identity["false_merge_rate"])
        <= float(thresholds["false_merge_rate"]),
        "retrieval_provenance": float(retrieval["provenance_coverage"])
        >= float(thresholds["provenance_coverage"]),
        "retrieval_support": float(retrieval["answer_support_rate"])
        >= float(thresholds["answer_support_rate"]),
        "interaction": float(interaction["unsolicited_speech_rate"])
        <= float(thresholds["unsolicited_speech_rate"]),
        "corrections": float(interaction["correction_retention"])
        >= float(thresholds["correction_retention"]),
        "object_provenance": float(object_learning["segmentation_provenance_coverage"])
        >= float(thresholds["object_provenance"]),
        "object_recall": float(object_learning["user_label_recall_accuracy"])
        >= float(thresholds["object_recall_accuracy"]),
        "asr_grounding": float(asr["accepted_grounding_coverage"])
        >= float(thresholds["asr_grounding"]),
    }
    passed = all(gates.values())
    return passed, {
        "status": "pass" if passed else "fail",
        "event_boundaries": event,
        "identity": identity,
        "retrieval": retrieval,
        "interaction": interaction,
        "object_learning": object_learning,
        "asr": asr,
        "runtime": runtime,
        "gates": gates,
        "thresholds": thresholds,
    }
