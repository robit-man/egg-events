import unittest

from egg_companion.core.behavior import classify_pose


def _make_coco_keypoints(
    left_shoulder: tuple[float, float, float] = (0.4, 0.4, 0.9),
    right_shoulder: tuple[float, float, float] = (0.6, 0.4, 0.9),
    left_elbow: tuple[float, float, float] = (0.3, 0.3, 0.8),
    right_elbow: tuple[float, float, float] = (0.7, 0.3, 0.8),
    left_wrist: tuple[float, float, float] = (0.3, 0.15, 0.9),
    right_wrist: tuple[float, float, float] = (0.7, 0.15, 0.9),
    left_hip: tuple[float, float, float] = (0.4, 0.7, 0.9),
    right_hip: tuple[float, float, float] = (0.6, 0.7, 0.9),
) -> list[list[float]]:
    points: list[list[float]] = [[0.5, 0.5, 0.5] for _ in range(17)]
    points[5] = list(left_shoulder)
    points[6] = list(right_shoulder)
    points[7] = list(left_elbow)
    points[8] = list(right_elbow)
    points[9] = list(left_wrist)
    points[10] = list(right_wrist)
    points[11] = list(left_hip)
    points[12] = list(right_hip)
    return points


class BehaviorTests(unittest.TestCase):
    def test_both_arms_raised_above_shoulders_is_waving(self) -> None:
        points = _make_coco_keypoints()
        self.assertEqual(classify_pose(points), "waving")

    def test_single_wrist_above_shoulder_is_standing(self) -> None:
        points = _make_coco_keypoints(
            left_wrist=(0.3, 0.15, 0.9),
            right_wrist=(0.7, 0.45, 0.9),
            right_elbow=(0.7, 0.5, 0.8),
        )
        self.assertEqual(classify_pose(points), "standing")

    def test_arms_raised_without_elbows_is_standing(self) -> None:
        points = _make_coco_keypoints(
            left_elbow=(0.3, 0.55, 0.8),
            right_elbow=(0.7, 0.55, 0.8),
        )
        self.assertEqual(classify_pose(points), "standing")

    def test_low_confidence_keypoints_is_standing(self) -> None:
        points = _make_coco_keypoints(
            left_wrist=(0.3, 0.15, 0.1),
            right_wrist=(0.7, 0.15, 0.1),
        )
        self.assertEqual(classify_pose(points), "standing")

    def test_short_torso_is_seated(self) -> None:
        points = _make_coco_keypoints(
            left_shoulder=(0.4, 0.5, 0.9),
            right_shoulder=(0.6, 0.5, 0.9),
            left_hip=(0.4, 0.52, 0.9),
            right_hip=(0.6, 0.52, 0.9),
        )
        self.assertEqual(classify_pose(points), "seated")

    def test_none_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_pose(None))

    def test_insufficient_keypoints_returns_none(self) -> None:
        self.assertIsNone(classify_pose([[0.5, 0.5, 0.5] for _ in range(10)]))
