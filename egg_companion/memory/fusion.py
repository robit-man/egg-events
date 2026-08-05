from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FusionResult:
    score: float
    components: dict[str, float]
    outcome: str


class EvidenceFusion:
    """Transparent late fusion; face evidence is required for confirmed person recall."""

    @staticmethod
    def person(
        face_similarity: float | None,
        clip_similarity: float | None = None,
        continuity: float = 0.0,
        user_alias: bool = False,
    ) -> FusionResult:
        components = {
            "face_similarity": max(0.0, min(1.0, face_similarity or 0.0)),
            "clip_similarity": max(0.0, min(1.0, clip_similarity or 0.0)),
            "same_camera_continuity": max(0.0, min(1.0, continuity)),
            "user_named_alias": 1.0 if user_alias else 0.0,
        }
        score = (
            0.70 * components["face_similarity"]
            + 0.12 * components["clip_similarity"]
            + 0.10 * components["same_camera_continuity"]
            + 0.08 * components["user_named_alias"]
        )
        if face_similarity is None or components["face_similarity"] < 0.45:
            outcome = "appearance_track" if score < 0.55 else "hypothesis"
        else:
            outcome = "recalled" if score >= 0.55 else "hypothesis"
        return FusionResult(round(score, 4), components, outcome)

    @staticmethod
    def object(masked_clip_similarity: float, continuity: float = 0.0) -> FusionResult:
        components = {
            "masked_clip_similarity": max(0.0, min(1.0, masked_clip_similarity)),
            "same_camera_continuity": max(0.0, min(1.0, continuity)),
        }
        score = 0.9 * components["masked_clip_similarity"] + 0.1 * components[
            "same_camera_continuity"
        ]
        return FusionResult(
            round(score, 4), components, "recalled" if score >= 0.78 else "hypothesis"
        )
