from __future__ import annotations

import hashlib
import re
from collections import deque

from egg_companion.models import InteractionDecision


class InteractionPolicy:
    """Speech gate with explicit grounding and duplicate-output suppression."""

    def __init__(self, history_size: int = 12) -> None:
        self._spoken_fingerprints: deque[str] = deque(maxlen=history_size)

    def evaluate(
        self, transcript: str, response: str, *, directed: bool = True
    ) -> InteractionDecision:
        normalized = " ".join(response.strip().split())
        if not normalized or normalized == "[[SILENT]]":
            return InteractionDecision(False, "cognition selected silence")
        fingerprint = hashlib.sha256(self._semantic_form(normalized).encode("utf-8")).hexdigest()[
            :16
        ]
        if not transcript.strip():
            return InteractionDecision(
                False, "no human utterance grounded the response", fingerprint
            )
        if not directed:
            return InteractionDecision(False, "utterance was not directed to Egg", fingerprint)
        if fingerprint in self._spoken_fingerprints:
            return InteractionDecision(False, "duplicate response suppressed", fingerprint)
        self._spoken_fingerprints.append(fingerprint)
        return InteractionDecision(True, "contextual response permitted", fingerprint)

    @staticmethod
    def _semantic_form(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))
