from __future__ import annotations


def object_learning_metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    eligible = [record for record in records if record.get("expected_label")]
    correct = [
        record
        for record in eligible
        if str(record.get("expected_label")).casefold()
        == str(record.get("recalled_label")).casefold()
    ]
    provenance = [record for record in records if record.get("mask_checksum")]
    return {
        "samples": len(records),
        "user_label_recall_accuracy": round(len(correct) / max(len(eligible), 1), 4),
        "segmentation_provenance_coverage": round(
            len(provenance) / max(len(records), 1), 4
        ),
    }
