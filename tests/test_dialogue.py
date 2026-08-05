from egg_companion.cognition.dialogue import DialogueClassifier, DialogueEvidence


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
