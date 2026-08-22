import unittest

from egg_companion.core.gaze import classify_gaze


def _make_face_keypoints(
    nose: tuple[float, float, float] = (0.5, 0.4, 0.9),
    left_eye: tuple[float, float, float] = (0.46, 0.35, 0.9),
    right_eye: tuple[float, float, float] = (0.54, 0.35, 0.9),
    left_ear: tuple[float, float, float] = (0.4, 0.36, 0.7),
    right_ear: tuple[float, float, float] = (0.6, 0.36, 0.7),
) -> list[list[float]]:
    points: list[list[float]] = [[0.5, 0.5, 0.5] for _ in range(17)]
    points[0] = list(nose)
    points[1] = list(left_eye)
    points[2] = list(right_eye)
    points[3] = list(left_ear)
    points[4] = list(right_ear)
    return points


class GazeTests(unittest.TestCase):
    def test_nose_centered_between_eyes_is_facing_camera(self) -> None:
        points = _make_face_keypoints()
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "facing_camera")
        self.assertAlmostEqual(result["yaw_offset"], 0.0, delta=0.05)

    def test_nose_shifted_left_of_eye_midpoint_is_looking_left(self) -> None:
        points = _make_face_keypoints(nose=(0.47, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_left")
        self.assertLess(result["yaw_offset"], 0.0)

    def test_nose_shifted_right_of_eye_midpoint_is_looking_right(self) -> None:
        points = _make_face_keypoints(nose=(0.53, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_right")
        self.assertGreater(result["yaw_offset"], 0.0)

    def test_extreme_offset_is_looking_away(self) -> None:
        points = _make_face_keypoints(nose=(0.7, 0.4, 0.9))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_away")

    def test_only_one_eye_visible_is_looking_away(self) -> None:
        points = _make_face_keypoints(right_eye=(0.54, 0.35, 0.05))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["state"], "looking_away")

    def test_low_confidence_nose_returns_none(self) -> None:
        points = _make_face_keypoints(nose=(0.5, 0.4, 0.1))
        self.assertIsNone(classify_gaze(points))

    def test_neither_eye_visible_returns_none(self) -> None:
        points = _make_face_keypoints(
            left_eye=(0.46, 0.35, 0.05), right_eye=(0.54, 0.35, 0.05),
        )
        self.assertIsNone(classify_gaze(points))

    def test_none_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_gaze(None))

    def test_insufficient_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_gaze([[0.5, 0.5, 0.9] for _ in range(3)]))

    def test_confidence_is_min_of_nose_and_eyes(self) -> None:
        points = _make_face_keypoints(nose=(0.5, 0.4, 0.55))
        result = classify_gaze(points)
        assert result is not None
        self.assertEqual(result["confidence"], 0.55)
