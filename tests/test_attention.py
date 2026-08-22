from datetime import datetime, timedelta, timezone
import unittest

from egg_companion.core.attention import AttentionManager
from egg_companion.models import BoundingBox, Detection, Observation
from egg_companion.models import GraphCognitiveSignal


class AttentionTests(unittest.TestCase):
    def test_new_person_is_prioritized(self) -> None:
        manager = AttentionManager(track_ttl_seconds=10, min_priority=0.1)
        person = Detection("person", 0.9, BoundingBox(0.1, 0.1, 0.7, 0.9))
        observation = Observation("front", datetime.now(timezone.utc), (person,))

        target = manager.select(observation)[0]

        self.assertEqual(target.novelty, 1.0)
        self.assertGreater(target.priority, 0.5)
        self.assertEqual(target.reason, "new person")

    def test_expired_tracks_become_novel_again(self) -> None:
        manager = AttentionManager(track_ttl_seconds=1, min_priority=0.1)
        person = Detection("person", 0.9, BoundingBox(0.1, 0.1, 0.7, 0.9))
        now = datetime.now(timezone.utc)
        manager.select(Observation("front", now, (person,)))

        target = manager.select(Observation("front", now + timedelta(seconds=2), (person,)))[0]

        self.assertEqual(target.novelty, 1.0)

    def test_detector_label_jitter_does_not_reenter_attention(self) -> None:
        manager = AttentionManager(track_ttl_seconds=10, min_priority=0.35)
        now = datetime.now(timezone.utc)
        first = Detection(
            "television", 0.8, BoundingBox(100, 100, 300, 300), {"frame_shape": [1080, 1920]}
        )
        jitter = Detection(
            "tv genre", 0.75, BoundingBox(102, 102, 302, 302), {"frame_shape": [1080, 1920]}
        )
        manager.select(Observation("front", now, (first,)))
        assert manager.select(Observation("front", now + timedelta(seconds=1), (jitter,))) == []

    def test_continuing_person_habituates_below_selection_threshold(self) -> None:
        manager = AttentionManager(track_ttl_seconds=10, min_priority=0.35)
        now = datetime.now(timezone.utc)
        person = Detection(
            "person", 0.9, BoundingBox(100, 100, 500, 900),
            {"frame_shape": [1080, 1920], "identity_id": "person-001"},
        )
        manager.select(Observation("front", now, (person,)))
        assert manager.select(Observation("front", now + timedelta(seconds=1), (person,))) == []

    def test_graph_familiarity_prevents_expired_stable_track_from_false_novelty(self) -> None:
        manager = AttentionManager(track_ttl_seconds=1, min_priority=0.35)
        now = datetime.now(timezone.utc)
        person = Detection(
            "person", 0.9, BoundingBox(100, 100, 500, 900),
            {"frame_shape": [1080, 1920], "identity_id": "person-001"},
        )
        manager.select(Observation("front", now, (person,)))
        signal = GraphCognitiveSignal(
            "person-001", familiarity=0.95, structural_relevance=0.6,
            knowledge_gap=0.3, evidence_count=30, edge_count=10,
        )

        assert manager.select(
            Observation("front", now + timedelta(seconds=2), (person,)),
            {"person-001": signal},
        ) == []

        selected = manager.select(
            Observation("front", now + timedelta(seconds=4), (person,)),
            {"person-001": signal},
            {"focus_terms": ["person"], "focus_entity_ids": ["person-001"]},
        )
        assert selected
        assert "conversation-relevant" in selected[0].reason

    def test_focus_camera_ids_boosts_priority_for_that_camera_only(self) -> None:
        manager = AttentionManager(track_ttl_seconds=10, min_priority=0.1)
        now = datetime.now(timezone.utc)
        thing = Detection("cup", 0.5, BoundingBox(10, 10, 30, 30))

        baseline = manager.select(Observation("camera-video1", now, (thing,)))[0]

        manager2 = AttentionManager(track_ttl_seconds=10, min_priority=0.1)
        boosted = manager2.select(
            Observation("camera-video1", now, (thing,)),
            None,
            {"focus_camera_ids": ["camera-video1"]},
        )[0]
        assert boosted.priority > baseline.priority
        assert "camera-focus-requested" in boosted.reason

    def test_focus_camera_ids_does_not_boost_other_cameras(self) -> None:
        manager = AttentionManager(track_ttl_seconds=10, min_priority=0.1)
        now = datetime.now(timezone.utc)
        thing = Detection("cup", 0.5, BoundingBox(10, 10, 30, 30))

        not_focused = manager.select(
            Observation("camera-video2", now, (thing,)),
            None,
            {"focus_camera_ids": ["camera-video1"]},
        )[0]
        assert "camera-focus-requested" not in not_focused.reason

    def test_gazing_at_camera_boosts_priority(self) -> None:
        now = datetime.now(timezone.utc)
        looking = Detection(
            "person", 0.6, BoundingBox(100, 100, 300, 500),
            {"frame_shape": [1080, 1920], "gaze": {"state": "facing_camera", "confidence": 0.9}},
        )
        away = Detection(
            "person", 0.6, BoundingBox(100, 100, 300, 500),
            {"frame_shape": [1080, 1920], "gaze": {"state": "looking_away", "confidence": 0.9}},
        )

        looking_target = AttentionManager(track_ttl_seconds=10, min_priority=0.1).select(
            Observation("front", now, (looking,))
        )[0]
        away_target = AttentionManager(track_ttl_seconds=10, min_priority=0.1).select(
            Observation("front", now, (away,))
        )[0]
        assert looking_target.priority > away_target.priority
        assert "gazing at viewer" in looking_target.reason
