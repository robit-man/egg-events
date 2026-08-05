from __future__ import annotations

from collections import defaultdict


def identity_metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    predictions: dict[str, set[str]] = defaultdict(set)
    truths: dict[str, set[str]] = defaultdict(set)
    confirmed = 0
    recalled = 0
    for record in records:
        truth = str(record.get("truth_id") or "")
        prediction = str(record.get("predicted_id") or "")
        if not truth:
            continue
        if bool(record.get("confirmed", True)):
            confirmed += 1
        if prediction:
            predictions[prediction].add(truth)
            truths[truth].add(prediction)
            recalled += 1
    false_merges = sum(max(0, len(values) - 1) for values in predictions.values())
    false_splits = sum(max(0, len(values) - 1) for values in truths.values())
    return {
        "samples": len(records),
        "false_merges": false_merges,
        "false_splits": false_splits,
        "false_merge_rate": round(false_merges / max(len(predictions), 1), 4),
        "false_split_rate": round(false_splits / max(len(truths), 1), 4),
        "confirmed_profile_recall": round(recalled / max(confirmed, 1), 4),
    }
