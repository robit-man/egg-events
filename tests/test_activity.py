import unittest

from egg_companion.config import ActivityConfig
from egg_companion.core.activity import ActivityGovernor


class ActivityGovernorTests(unittest.TestCase):
    def test_full_alertness_before_any_activity_recorded(self) -> None:
        governor = ActivityGovernor(ActivityConfig())

        self.assertEqual(governor.scale(1000.0), 1.0)

    def test_falls_off_toward_idle_floor_after_sustained_quiet(self) -> None:
        settings = ActivityConfig(idle_floor=0.2, decay_seconds=10, novelty_threshold=0.1)
        governor = ActivityGovernor(settings)

        governor.note_visual(novelty=1.0, has_presence=True, now=0.0)
        quiet_scale = governor.scale(1000.0)

        self.assertAlmostEqual(quiet_scale, 0.2, places=3)

    def test_presence_without_novelty_holds_full_alertness(self) -> None:
        settings = ActivityConfig(idle_floor=0.2, decay_seconds=10, novelty_threshold=0.5)
        governor = ActivityGovernor(settings)

        governor.note_visual(novelty=0.0, has_presence=True, now=0.0)
        governor.note_visual(novelty=0.0, has_presence=True, now=5.0)

        self.assertAlmostEqual(governor.scale(5.0), 1.0, places=3)

    def test_absence_and_no_novelty_does_not_refresh_activity(self) -> None:
        settings = ActivityConfig(idle_floor=0.2, decay_seconds=10, novelty_threshold=0.5)
        governor = ActivityGovernor(settings)

        governor.note_visual(novelty=0.0, has_presence=True, now=0.0)
        governor.note_visual(novelty=0.0, has_presence=False, now=1.0)

        self.assertLess(governor.scale(50.0), 0.25)

    def test_new_activity_snaps_alertness_back_to_full(self) -> None:
        settings = ActivityConfig(idle_floor=0.2, decay_seconds=5, novelty_threshold=0.1)
        governor = ActivityGovernor(settings)

        governor.note_visual(novelty=1.0, has_presence=True, now=0.0)
        self.assertLess(governor.scale(100.0), 0.25)

        governor.note_visual(novelty=1.0, has_presence=True, now=100.0)

        self.assertAlmostEqual(governor.scale(100.0), 1.0, places=3)

    def test_audio_speech_counts_as_activity(self) -> None:
        settings = ActivityConfig(idle_floor=0.2, decay_seconds=5, novelty_threshold=0.5)
        governor = ActivityGovernor(settings)

        governor.note_audio(speech_detected=True, now=10.0)

        self.assertAlmostEqual(governor.scale(10.0), 1.0, places=3)

    def test_disabled_governor_always_reports_full_alertness(self) -> None:
        settings = ActivityConfig(enabled=False, idle_floor=0.2, decay_seconds=5)
        governor = ActivityGovernor(settings)

        governor.note_visual(novelty=1.0, has_presence=True, now=0.0)

        self.assertEqual(governor.scale(1000.0), 1.0)

    def test_scaled_fps_and_interval_move_in_opposite_directions_when_idle(self) -> None:
        settings = ActivityConfig(idle_floor=0.25, decay_seconds=5, novelty_threshold=0.1)
        governor = ActivityGovernor(settings)
        governor.note_visual(novelty=1.0, has_presence=True, now=0.0)

        fps = governor.scaled_fps(8.0, 1000.0)
        interval = governor.scaled_interval(20.0, 1000.0)

        self.assertAlmostEqual(fps, 2.0, places=2)
        self.assertAlmostEqual(interval, 80.0, places=2)


if __name__ == "__main__":
    unittest.main()
