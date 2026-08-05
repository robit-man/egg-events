from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


DialogueAct = Literal[
    "question", "correction", "person_naming", "object_naming", "command",
    "acknowledgement", "conversation",
]


@dataclass(frozen=True)
class DialogueEvidence:
    transcript: str
    doa_aligned: bool = False
    seconds_since_tts: float | None = None
    interaction_pending: bool = False
    language_directed: bool | None = None


@dataclass(frozen=True)
class DialogueDecision:
    directed: bool
    act: DialogueAct
    components: dict[str, float]
    reason: str


class DialogueClassifier:
    """Classifies grounded speech without requiring a hard-coded wake word."""

    QUESTION_WORDS = {"what", "when", "where", "who", "why", "how", "can", "could", "would"}
    COMMAND_WORDS = {"show", "tell", "remember", "forget", "stop", "start", "find"}

    def classify(self, evidence: DialogueEvidence) -> DialogueDecision:
        normalized = " ".join(evidence.transcript.strip().split())
        words = {word.strip(".,!?;:\"'").casefold() for word in normalized.split()}
        if words & {"actually", "correction", "wrong", "isn't", "not"}:
            act: DialogueAct = "correction"
        elif words & {"i'm", "called", "name"} and words & {"i", "me", "my", "i'm"}:
            act = "person_naming"
        elif words & {"this", "that", "object"} and words & {"is", "called", "name"}:
            act = "object_naming"
        elif normalized.endswith("?") or words & self.QUESTION_WORDS:
            act = "question"
        elif words & self.COMMAND_WORDS:
            act = "command"
        elif words <= {"yes", "no", "okay", "ok", "thanks", "thank", "you"}:
            act = "acknowledgement"
        else:
            act = "conversation"
        components = {
            "doa_alignment": 1.0 if evidence.doa_aligned else 0.0,
            "pending_interaction": 1.0 if evidence.interaction_pending else 0.0,
            "language_direction": (
                1.0 if evidence.language_directed is True
                else 0.0 if evidence.language_directed is False
                else 0.5
            ),
            "tts_echo_risk": (
                1.0
                if evidence.seconds_since_tts is not None and evidence.seconds_since_tts < 2.0
                else 0.0
            ),
        }
        directed_score = (
            0.35 * components["doa_alignment"]
            + 0.35 * components["pending_interaction"]
            + 0.45 * components["language_direction"]
            - 0.65 * components["tts_echo_risk"]
        )
        directed = bool(normalized and directed_score >= 0.35)
        reason = (
            "grounded dialogue evidence indicates Egg was addressed"
            if directed
            else "insufficient directed-speech evidence"
        )
        return DialogueDecision(directed, act, components, reason)
