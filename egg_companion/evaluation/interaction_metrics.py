from __future__ import annotations

import re


def _fingerprint(response: object) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(response or "").casefold()))


def interaction_metrics(records: list[dict[str, object]]) -> dict[str, float | int]:
    spoken = [record for record in records if bool(record.get("spoken"))]
    unsolicited = [record for record in spoken if not bool(record.get("directed"))]
    fingerprints = [_fingerprint(record.get("response")) for record in spoken]
    duplicates = len(fingerprints) - len(set(value for value in fingerprints if value))
    undirected = [record for record in records if not bool(record.get("directed"))]
    corrections = [record for record in records if bool(record.get("correction_required"))]
    retained = [record for record in corrections if bool(record.get("correction_retained"))]
    return {
        "events": len(records),
        "unsolicited_speech_rate": round(len(unsolicited) / max(len(records), 1), 4),
        "duplicate_reply_rate": round(duplicates / max(len(spoken), 1), 4),
        "undirected_speech_suppression": round(
            sum(not bool(record.get("spoken")) for record in undirected) / max(len(undirected), 1),
            4,
        ),
        "correction_retention": round(len(retained) / max(len(corrections), 1), 4),
    }
