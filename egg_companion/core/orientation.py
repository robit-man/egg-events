from __future__ import annotations

from collections.abc import Sequence


def upright_pose_score(keypoints: Sequence[Sequence[float]]) -> float:
    if len(keypoints) < 13:
        return 0.0
    shoulders = (keypoints[5], keypoints[6])
    hips = (keypoints[11], keypoints[12])
    shoulder_y = (shoulders[0][1] + shoulders[1][1]) / 2
    hip_y = (hips[0][1] + hips[1][1]) / 2
    torso = max(0.0, min(1.0, (hip_y - shoulder_y) * 2.5))
    level = max(0.0, 1.0 - abs(shoulders[0][1] - shoulders[1][1]) * 5)
    return 0.8 * torso + 0.2 * level


def select_rotation(scores: dict[int, float]) -> int | None:
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if len(ranked) < 2 or ranked[0][1] < 0.55 or ranked[0][1] - ranked[1][1] < 0.12:
        return None
    return ranked[0][0]
