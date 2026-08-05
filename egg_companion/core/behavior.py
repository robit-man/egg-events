from __future__ import annotations

from collections.abc import Sequence


def classify_pose(keypoints: Sequence[Sequence[float]] | None) -> str | None:
    """Classify coarse, non-identifying behavior from COCO body keypoints."""
    if not keypoints or len(keypoints) < 17:
        return None
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_wrist, right_wrist = keypoints[9], keypoints[10]
    left_hip, right_hip = keypoints[11], keypoints[12]
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    if min(left_wrist[1], right_wrist[1]) < shoulder_y:
        return "waving"
    if abs(hip_y - shoulder_y) < 0.10:
        return "seated"
    return "standing"
