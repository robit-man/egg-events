from egg_companion.adapters.omnius import OmniusClient


def test_vlm_object_response_requires_bounded_json_label() -> None:
    assert OmniusClient.parse_object_classification('{"label":"ceramic mug","confidence":0.91}') == ("ceramic mug", 0.91)
    assert OmniusClient.parse_object_classification('{"label":null,"confidence":0.91}') is None
    assert OmniusClient.parse_object_classification('a mug') is None
    assert OmniusClient.parse_object_classification('{"label":"mug","confidence":1.2}') is None


def test_person_name_parser_requires_explicit_bounded_json_name() -> None:
    assert OmniusClient.parse_person_name('{"name":"Ada Lovelace"}') == "Ada Lovelace"
    assert OmniusClient.parse_person_name('{"name":null}') is None
    assert OmniusClient.parse_person_name('My name is Ada') is None


def test_asr_grounding_rejects_no_speech_low_probability_and_repetition() -> None:
    silence = {"segments": [{"no_speech_prob": 0.8, "avg_logprob": -0.2, "compression_ratio": 1.0}]}
    unlikely = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -1.2, "compression_ratio": 1.0}]}
    repetitive = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 2.5}]}
    grounded = {"segments": [{"no_speech_prob": 0.1, "avg_logprob": -0.2, "compression_ratio": 1.1}]}

    assert OmniusClient.transcription_rejection_reason(silence) == "high no-speech probability"
    assert OmniusClient.transcription_rejection_reason(unlikely) == "low average token probability"
    assert OmniusClient.transcription_rejection_reason(repetitive) == "repetitive transcript compression"
    assert OmniusClient.transcription_rejection_reason(grounded) is None
    assert not OmniusClient.transcription_is_grounded(silence)
    assert OmniusClient.transcription_is_grounded(grounded)
