from __future__ import annotations


def asr_grounding_metrics(windows: list[dict[str, object]]) -> dict[str, float | int]:
    accepted = [window for window in windows if bool(window.get("accepted"))]
    rejected = [window for window in windows if not bool(window.get("accepted"))]
    correctly_rejected = [
        window for window in rejected if not bool(window.get("vad")) or bool(window.get("echo"))
    ]
    grounded = [
        window
        for window in accepted
        if bool(window.get("vad"))
        and isinstance(window.get("rms"), (int, float))
        and bool(window.get("evidence_id"))
    ]
    return {
        "windows": len(windows),
        "accepted": len(accepted),
        "rejected": len(rejected),
        "silent_echo_rejection_rate": round(
            len(correctly_rejected) / max(len(rejected), 1), 4
        ),
        "accepted_grounding_coverage": round(len(grounded) / max(len(accepted), 1), 4),
    }
