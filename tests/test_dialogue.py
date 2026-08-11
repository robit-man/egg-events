import json

from egg_companion.cognition.dialogue import (
    DialogueClassifier,
    DialogueEvidence,
    parse_interruption_decision,
)


def test_dialogue_uses_multisignal_direction_without_wake_word() -> None:
    decision = DialogueClassifier().classify(
        DialogueEvidence(
            "Could you find my mug?", doa_aligned=True, language_directed=True
        )
    )

    assert decision.directed
    assert decision.act == "question"


def test_recent_tts_echo_is_not_directed_dialogue() -> None:
    decision = DialogueClassifier().classify(
        DialogueEvidence(
            "That is your mug.", doa_aligned=True, seconds_since_tts=0.4,
            language_directed=False,
        )
    )

    assert not decision.directed
    assert decision.components["tts_echo_risk"] == 1.0


def test_timing_metadata_does_not_override_semantic_direction() -> None:
    decision = DialogueClassifier().classify(
        DialogueEvidence(
            "Wait, that is the wrong mug.",
            doa_aligned=True,
            seconds_since_tts=0.1,
            language_directed=True,
            playback_overlap=True,
            interruption_genuine=True,
        )
    )

    assert decision.directed
    assert decision.components["tts_echo_risk"] == 1.0


def test_interruption_contract_is_strict_and_typed() -> None:
    valid = {
        "version": 1,
        "genuine": True,
        "confidence": 0.94,
        "reason": "heard_correction",
        "summary": "The heard speaker corrected Egg's active response.",
        "should_cancel_playback": True,
    }

    decision = parse_interruption_decision(json.dumps(valid))

    assert decision is not None
    assert decision.genuine
    assert parse_interruption_decision(json.dumps({**valid, "extra": "no"})) is None
    assert parse_interruption_decision(json.dumps({**valid, "genuine": "true"})) is None
