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
