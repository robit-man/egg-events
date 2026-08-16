from __future__ import annotations

from collections.abc import Sequence


def _visible(keypoint: Sequence[float]) -> bool:
    """Return True if keypoint has non-zero coordinates (visible in frame)."""
    return len(keypoint) >= 3 and keypoint[0] > 0 and keypoint[1] > 0 and keypoint[2] > 0.3


def classify_pose(keypoints: Sequence[Sequence[float]] | None) -> str | None:
    """Classify coarse, non-identifying behavior from COCO body keypoints.

    Waving detection requires *both* arms clearly raised above shoulders with
    a meaningful margin to avoid false positives from normal gestures (pointing,
    stretching, scratching, reaching).  Partially visible keypoints are ignored.
    """
    if not keypoints or len(keypoints) < 17:
        return None
    left_shoulder, right_shoulder = keypoints[5], keypoints[6]
    left_wrist, right_wrist = keypoints[9], keypoints[10]
    left_elbow, right_elbow = keypoints[7], keypoints[8]
    left_hip, right_hip = keypoints[11], keypoints[12]
    shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
    hip_y = (left_hip[1] + right_hip[1]) / 2
    torso = abs(hip_y - shoulder_y)
    if torso < 0.05:
        return "seated"
    margin = torso * 0.25
    shoulders_clear = (
        _visible(left_shoulder)
        and _visible(right_shoulder)
        and _visible(left_wrist)
        and _visible(right_wrist)
    )
    if shoulders_clear:
        left_raised = left_wrist[1] < left_shoulder[1] - margin
        right_raised = right_wrist[1] < right_shoulder[1] - margin
        if left_raised and right_raised:
            elbows_raised = (
                _visible(left_elbow)
                and _visible(right_elbow)
                and left_elbow[1] < shoulder_y
                and right_elbow[1] < shoulder_y
            )
            if elbows_raised:
                return "waving"
    return "standing"
