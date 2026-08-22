from __future__ import annotations

from collections.abc import Sequence

# COCO keypoint indices used for a coarse gaze/facing-direction estimate.
_NOSE, _LEFT_EYE, _RIGHT_EYE, _LEFT_EAR, _RIGHT_EAR = 0, 1, 2, 3, 4


def _conf(keypoint: Sequence[float]) -> float:
    return keypoint[2] if len(keypoint) >= 3 else 0.0


def classify_gaze(
    keypoints: Sequence[Sequence[float]] | None,
    confidence_threshold: float = 0.5,
) -> dict[str, object] | None:
    """Classify a coarse facing-direction / gaze state from COCO face keypoints.

    This is a v1 heuristic, not a fitted gaze-vector model: it estimates
    horizontal head orientation from the nose position relative to the
    eye midpoint (normalized by inter-eye distance), the way many
    lightweight pose-only systems approximate yaw without a dedicated
    face mesh or iris model. It answers "is this person oriented toward
    the camera" (useful as a social-attention signal for a companion
    robot), not precise eye-direction/degrees -- that would need a real
    face-landmark or gaze-regression model (e.g. MediaPipe iris,
    L2CS-Net), which isn't justified as an always-on per-frame cost
    across three cameras on this hardware today.

    Returns None if there aren't enough confident keypoints to say
    anything at all (distinct from returning state="unknown", which means
    keypoints exist but are inconclusive/profile).
    """
    if not keypoints or len(keypoints) <= _RIGHT_EAR:
        return None

    nose = keypoints[_NOSE]
    left_eye = keypoints[_LEFT_EYE]
    right_eye = keypoints[_RIGHT_EYE]
    left_ear = keypoints[_LEFT_EAR]
    right_ear = keypoints[_RIGHT_EAR]

    nose_conf = _conf(nose)
    left_eye_conf = _conf(left_eye)
    right_eye_conf = _conf(right_eye)
    one_eye_visible = (
        left_eye_conf >= confidence_threshold or right_eye_conf >= confidence_threshold
    )
    if nose_conf < confidence_threshold or not one_eye_visible:
        return None

    eyes_visible = (
        left_eye_conf >= confidence_threshold and right_eye_conf >= confidence_threshold
    )
    if not eyes_visible:
        # Only one eye confidently visible -- near-profile view. We can't
        # compute a meaningful eye-midpoint offset, but ear asymmetry alone
        # is a reasonably strong "turned away" signal at this point.
        return {
            "state": "looking_away",
            "yaw_offset": None,
            "confidence": round(min(nose_conf, max(left_eye_conf, right_eye_conf)), 3),
        }

    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    eye_dist = abs(right_eye[0] - left_eye[0])
    if eye_dist < 1e-4:
        return {"state": "unknown", "yaw_offset": None, "confidence": 0.0}

    # Sign convention: negative == nose shifted toward the left edge of the
    # image relative to the eye midpoint; positive == toward the right edge.
    # This is image-relative (observer's frame), not the subject's own
    # anatomical left/right, which is what actually matters operationally
    # (which side of frame they're oriented toward).
    yaw_offset = (nose[0] - eye_mid_x) / eye_dist
    confidence = round(min(nose_conf, left_eye_conf, right_eye_conf), 3)

    if abs(yaw_offset) <= 0.18:
        state = "facing_camera"
    elif abs(yaw_offset) > 0.5:
        state = "looking_away"
    elif yaw_offset < 0:
        state = "looking_left"
    else:
        state = "looking_right"

    return {
        "state": state,
        "yaw_offset": round(yaw_offset, 4),
        "confidence": confidence,
    }
