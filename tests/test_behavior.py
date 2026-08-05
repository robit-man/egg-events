import unittest

from egg_companion.core.behavior import classify_pose


class BehaviorTests(unittest.TestCase):
    def test_wrist_above_shoulder_is_waving(self) -> None:
        points = [[0.5, 0.5] for _ in range(17)]
        points[5] = [0.4, 0.4]
        points[6] = [0.6, 0.4]
        points[9] = [0.3, 0.2]
        points[10] = [0.7, 0.5]
        points[11] = [0.4, 0.7]
        points[12] = [0.6, 0.7]

        self.assertEqual(classify_pose(points), "waving")
