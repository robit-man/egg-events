from __future__ import annotations

import importlib.util
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "jetson_whisper_server.py"
SPEC = importlib.util.spec_from_file_location("egg_jetson_whisper_server", SOURCE)
assert SPEC is not None and SPEC.loader is not None
SERVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SERVER)


def _result(text: str, log_prob: float, no_speech: float) -> dict[str, object]:
    return {
        "text": text,
        "language": "en",
        "segments": [
            {
                "text": text,
                "avg_logprob": log_prob,
                "no_speech_prob": no_speech,
                "compression_ratio": 1.1,
            }
        ],
    }


def test_dual_whisper_prefers_accurate_pass_when_models_agree() -> None:
    selected, reason, agreement = SERVER.choose_dual_result(
        _result("we should test the microphone", -0.25, 0.02),
        _result("We should test the microphone.", -0.18, 0.01),
    )

    assert reason is None
    assert selected["text"] == "We should test the microphone."
    assert agreement > 0.95


def test_dual_whisper_rejects_weak_disagreement() -> None:
    selected, reason, agreement = SERVER.choose_dual_result(
        _result("the microphone is working", -0.30, 0.03),
        _result("thanks everyone for joining us", -0.90, 0.40),
    )

    assert selected == {}
    assert reason == "dual Whisper disagreement on weak base decode"
    assert agreement < 0.22


def test_legitimately_spoken_phrase_is_not_semantically_blacklisted() -> None:
    assert SERVER.rejection_reason(
        _result("Thanks for watching!", -0.1, 0.01)
    ) is None
