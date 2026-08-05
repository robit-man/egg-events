from egg_companion.memory.fusion import EvidenceFusion


def test_clip_only_person_similarity_never_confirms_identity() -> None:
    result = EvidenceFusion.person(None, clip_similarity=0.99, continuity=1.0, user_alias=True)

    assert result.outcome != "recalled"
    assert result.components["face_similarity"] == 0.0


def test_masked_object_fusion_exposes_components() -> None:
    result = EvidenceFusion.object(0.9, continuity=0.8)

    assert result.outcome == "recalled"
    assert result.components == {"masked_clip_similarity": 0.9, "same_camera_continuity": 0.8}
