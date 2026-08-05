from __future__ import annotations


def _average(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def runtime_metrics(samples: list[dict[str, object]]) -> dict[str, float | int]:
    def values(name: str) -> list[float]:
        return [
            float(sample[name])
            for sample in samples
            if isinstance(sample.get(name), (int, float))
        ]

    db_sizes = values("db_bytes")
    media_sizes = values("media_bytes")
    return {
        "samples": len(samples),
        "average_camera_fps": _average(values("camera_fps")),
        "average_asr_delay_ms": _average(values("asr_delay_ms")),
        "peak_gpu_memory_mb": max(values("gpu_memory_mb"), default=0.0),
        "peak_ram_mb": max(values("ram_mb"), default=0.0),
        "db_growth_bytes": max(db_sizes, default=0.0) - min(db_sizes, default=0.0),
        "media_growth_bytes": max(media_sizes, default=0.0) - min(media_sizes, default=0.0),
    }
