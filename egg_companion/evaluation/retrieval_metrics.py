from __future__ import annotations


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * percentile)))
    return ordered[index]


def retrieval_metrics(queries: list[dict[str, object]]) -> dict[str, float | int]:
    supported_hits = 0
    hits_total = 0
    relevant_found = 0
    relevant_total = 0
    latencies: list[float] = []
    for query in queries:
        relevant = {str(value) for value in query.get("relevant_ids", [])}
        hits = query.get("hits", [])
        hit_ids: set[str] = set()
        if isinstance(hits, list):
            for hit in hits:
                if not isinstance(hit, dict):
                    continue
                hits_total += 1
                hit_ids.add(str(hit.get("owner_id") or ""))
                if hit.get("provenance"):
                    supported_hits += 1
        relevant_total += len(relevant)
        relevant_found += len(relevant & hit_ids)
        latencies.append(float(query.get("latency_ms") or 0.0))
    return {
        "queries": len(queries),
        "provenance_coverage": round(supported_hits / max(hits_total, 1), 4),
        "answer_support_rate": round(relevant_found / max(relevant_total, 1), 4),
        "p50_latency_ms": round(_percentile(latencies, 0.50), 3),
        "p95_latency_ms": round(_percentile(latencies, 0.95), 3),
    }
