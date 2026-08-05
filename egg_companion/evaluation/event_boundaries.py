from __future__ import annotations


def boundary_metrics(
    expected: list[float], predicted: list[float], tolerance_seconds: float = 0.5
) -> dict[str, float | int]:
    unmatched = set(range(len(expected)))
    true_positives = 0
    for prediction in predicted:
        match = min(
            unmatched,
            key=lambda index: abs(expected[index] - prediction),
            default=None,
        )
        if match is not None and abs(expected[match] - prediction) <= tolerance_seconds:
            unmatched.remove(match)
            true_positives += 1
    precision = true_positives / len(predicted) if predicted else float(not expected)
    recall = true_positives / len(expected) if expected else float(not predicted)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": true_positives,
        "false_positives": len(predicted) - true_positives,
        "false_negatives": len(expected) - true_positives,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def static_scene_event_rate(event_count: int, duration_seconds: float) -> float:
    return round(event_count / max(duration_seconds / 60.0, 1 / 60.0), 4)
