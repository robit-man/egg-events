from egg_companion.cognition.conversation import ConversationTurnController


def test_stale_response_cannot_claim_playback() -> None:
    turns = ConversationTurnController()
    turns.finalize_audio_turn(
        "newest turn",
        utterance_id="utterance-1",
        started_at=1.0,
        ended_at=2.0,
    )

    assert turns.begin_playback("stale response", expected_revision=0) is None
    assert turns.snapshot()["floor"] == "processing"


def test_false_barge_resumes_same_logical_playback_from_bound_cursor() -> None:
    turns = ConversationTurnController()
    playback = turns.begin_playback(
        "The response continues after a false barge.",
        expected_revision=0,
        playback_id="playback-1",
        started_at=1.0,
    )
    assert playback is not None

    barge = turns.speech_started(started_at=2.0)
    assert barge is not None
    turns.bind_barge_cursor(barge.barge_id, 1.25)
    heard = turns.finalize_audio_turn(
        "background speech",
        utterance_id="utterance-1",
        started_at=2.0,
        ended_at=2.5,
        barge_id=barge.barge_id,
    )

    resumed = turns.resolve_barge(barge.barge_id, "resume")

    assert heard.revision == 1
    assert resumed is not None
    assert resumed.playback_id == playback.playback_id
    assert resumed.status == "playing"
    assert resumed.resume_seconds == 1.25
    assert turns.active_barge is None


def test_genuine_barge_terminates_playback_once_and_preserves_audible_order() -> None:
    turns = ConversationTurnController()
    playback = turns.begin_playback(
        "An agent utterance that is cut short.",
        expected_revision=0,
        playback_id="playback-1",
        started_at=1.0,
    )
    assert playback is not None
    barge = turns.speech_started(started_at=2.0)
    assert barge is not None
    turns.finalize_audio_turn(
        "Wait, that is wrong.",
        utterance_id="utterance-1",
        started_at=2.0,
        ended_at=2.6,
        barge_id=barge.barge_id,
    )

    terminal = turns.resolve_barge(barge.barge_id, "interrupted", ended_at=2.6)
    duplicate = turns.resolve_barge(barge.barge_id, "interrupted", ended_at=2.7)

    assert terminal is not None
    assert terminal.status == "interrupted"
    assert duplicate is None
    assert [(turn.role, turn.status) for turn in turns.history()] == [
        ("agent", "interrupted"),
        ("heard", "final"),
    ]
    assert turns.active_playback is None


def test_followup_speech_keeps_the_existing_pending_barge_identity() -> None:
    turns = ConversationTurnController()
    turns.begin_playback("active response", expected_revision=0, playback_id="playback-1")
    first = turns.speech_started(started_at=1.0)

    second = turns.speech_started(started_at=1.5)

    assert first is not None
    assert second == first


def test_pending_acoustic_input_blocks_response_publication_before_asr() -> None:
    turns = ConversationTurnController()
    turns.speech_started(started_at=1.0)

    assert turns.begin_playback("too early", expected_revision=0) is None

    turns.reject_audio_input()
    assert turns.begin_playback("safe after rejection", expected_revision=0) is not None


def test_new_acoustic_onset_invalidates_an_older_barge_decision() -> None:
    turns = ConversationTurnController()
    turns.begin_playback("active response", expected_revision=0, playback_id="playback-1")
    barge = turns.speech_started(started_at=1.0)
    assert barge is not None
    first = turns.finalize_audio_turn(
        "first fragment",
        utterance_id="utterance-1",
        started_at=1.0,
        ended_at=1.4,
        barge_id=barge.barge_id,
    )
    assert turns.barge_decision_current(barge.barge_id, first.revision)

    turns.speech_started(started_at=1.5)

    assert not turns.barge_decision_current(barge.barge_id, first.revision)
