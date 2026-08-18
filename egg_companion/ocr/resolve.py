"""OCR V2 resolution layer — merges local and Omnius results."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OcrResolution:
    """Final merged OCR output from multiple proposers."""

    text: str
    engine: str
    confidence: float
    local_text: str | None = None
    local_confidence: float | None = None
    local_engine: str | None = None
    omnius_text: str | None = None
    omnius_confidence: float | None = None
    source_count: int = 0
    regions: tuple[dict[str, object], ...] = ()
    image_size: tuple[int, int] | None = None

    @property
    def engines_used(self) -> list[str]:
        engines = []
        if self.local_engine:
            engines.append(self.local_engine)
        if self.omnius_text is not None:
            engines.append("omnius-ocr")
        return engines


def _normalize_text(text: str) -> str:
    """Collapse whitespace, strip, cap at 2000 chars."""
    normalized = "\n".join(
        " ".join(line.split())
        for line in text.splitlines()
        if line.strip()
    )
    return normalized[:2000]


def _text_similarity(a: str, b: str) -> float:
    """Jaccard similarity of character 3-grams."""
    if not a or not b:
        return 0.0
    a_lower = a.casefold()
    b_lower = b.casefold()
    grams_a = {a_lower[i:i+3] for i in range(len(a_lower) - 2)}
    grams_b = {b_lower[i:i+3] for i in range(len(b_lower) - 2)}
    if not grams_a or not grams_b:
        return 1.0 if a_lower == b_lower else 0.0
    intersection = grams_a & grams_b
    union = grams_a | grams_b
    return len(intersection) / len(union) if union else 0.0


def resolve_text_observations(
    local_result: dict[str, object] | None,
    omnius_result: dict[str, object] | None,
    *,
    confidence_threshold: float = 0.72,
) -> OcrResolution | None:
    """Merge local and Omnius OCR results into a single resolution.

    Decision logic:
    1. If only one source has text → use it.
    2. If both have text and are similar (jaccard > 0.6) → use higher confidence.
    3. If both have text but differ → use the one with higher confidence,
       but prefer Omnius if its confidence is within 0.1 of local.
    4. If neither has text → return None.
    """
    local_text = _normalize_text(str(local_result.get("text") or "")) if local_result else ""
    omnius_text = _normalize_text(str(omnius_result.get("text") or "")) if omnius_result else ""

    local_has = bool(local_text)
    omnius_has = bool(omnius_text)

    if not local_has and not omnius_has:
        return None

    local_conf = float(local_result.get("confidence", 0.5)) if local_result else 0.0
    omnius_conf = float(omnius_result.get("confidence", 0.5)) if omnius_result else 0.0
    local_engine = str(local_result.get("engine", "local-ocr")) if local_result else None
    regions = ()
    image_size = None

    if local_has and not omnius_has:
        regions = tuple(local_result.get("regions", [])) if local_result else ()
        image_size = _extract_image_size(local_result)
        return OcrResolution(
            text=local_text,
            engine=local_engine or "local-ocr",
            confidence=local_conf,
            local_text=local_text,
            local_confidence=local_conf,
            local_engine=local_engine,
            source_count=1,
            regions=regions,
            image_size=image_size,
        )

    if omnius_has and not local_has:
        regions = tuple(omnius_result.get("regions", [])) if omnius_result else ()
        image_size = _extract_image_size(omnius_result)
        return OcrResolution(
            text=omnius_text,
            engine="omnius-ocr",
            confidence=omnius_conf,
            omnius_text=omnius_text,
            omnius_confidence=omnius_conf,
            source_count=1,
            regions=regions,
            image_size=image_size,
        )

    # Both have text — compare
    similarity = _text_similarity(local_text, omnius_text)

    if similarity > 0.6:
        # Agree on content — use higher confidence
        if omnius_conf >= local_conf - 0.1:
            chosen_text = omnius_text
            chosen_engine = "omnius-ocr"
            chosen_conf = omnius_conf
        else:
            chosen_text = local_text
            chosen_engine = local_engine or "local-ocr"
            chosen_conf = local_conf
    else:
        # Disagree — prefer Omnius if confident enough
        if omnius_conf >= confidence_threshold:
            chosen_text = omnius_text
            chosen_engine = "omnius-ocr"
            chosen_conf = omnius_conf
        else:
            chosen_text = local_text
            chosen_engine = local_engine or "local-ocr"
            chosen_conf = local_conf

    regions = tuple(local_result.get("regions", [])) if local_result else ()
    image_size = _extract_image_size(local_result) or _extract_image_size(omnius_result)

    return OcrResolution(
        text=chosen_text,
        engine=chosen_engine,
        confidence=chosen_conf,
        local_text=local_text,
        local_confidence=local_conf,
        local_engine=local_engine,
        omnius_text=omnius_text,
        omnius_confidence=omnius_conf,
        source_count=2,
        regions=regions,
        image_size=image_size,
    )


def _extract_image_size(result: dict[str, object] | None) -> tuple[int, int] | None:
    if result is None:
        return None
    size = result.get("image_size")
    if isinstance(size, (list, tuple)) and len(size) == 2:
        try:
            return (int(size[0]), int(size[1]))
        except (ValueError, TypeError):
            return None
    return None
