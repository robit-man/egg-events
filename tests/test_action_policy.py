"""Regression tests for policy-gated action dispatch.

PolicyValidator and ActionStore existed as complete, tested scaffolding
but were never instantiated anywhere in the live runtime -- nothing ever
called PolicyValidator.validate() before a real effect happened. _speak()
is the one true chokepoint for every speech-producing action (speak,
ask_clarifying_question, ask_identity_clarification -- ~9 call sites
across the runtime), so it's wired to propose+validate via
MemoryPipeline.propose_action() before ever synthesizing audio, and to
record the outcome via MemoryPipeline.record_action_execution() after.
"""

from __future__ import annotations

import asyncio

import pytest

from egg_companion.config import EggConfig
from egg_companion.memory.pipeline import MemoryPipeline
from egg_companion.memory.store import MemoryStore
from egg_companion.runtime import CompanionRuntime
from egg_companion.world.policy import PolicyRule


def _config(tmp_path) -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "camera_discovery": {"enabled": False},
            "memory": {"storage_dir": str(tmp_path / "memory")},
        }
    )


def _runtime_with_pipeline(tmp_path):
    config = _config(tmp_path)
    store = MemoryStore(config.memory)
    pipeline = MemoryPipeline(config, store)
    assert pipeline._action_store is not None
    assert pipeline._policy_validator is not None
    runtime = object.__new__(CompanionRuntime)
    runtime._memory = pipeline
    return runtime, pipeline


class TestProposeAction:
    """MemoryPipeline.propose_action creates, persists, and policy-checks."""

    def test_creates_and_persists_proposal(self, tmp_path) -> None:
        _, pipeline = _runtime_with_pipeline(tmp_path)
        proposal, violations = pipeline.propose_action(
            "speak", inputs={"text": "hello"},
        )
        assert proposal.action_type == "speak"
        assert violations == []
        stored = pipeline._action_store.get_proposal(proposal.proposal_id)
        assert stored is not None
        assert stored.inputs == {"text": "hello"}

    def test_fails_open_with_no_world_model(self, tmp_path) -> None:
        """Without a durable ledger there's nothing to check policy
        against -- must not crash, must not fabricate a block."""
        _, pipeline = _runtime_with_pipeline(tmp_path)
        pipeline._action_store = None
        pipeline._policy_validator = None
        proposal, violations = pipeline.propose_action("speak", inputs={"text": "hi"})
        assert proposal.action_type == "speak"
        assert violations == []


class TestRecordActionPopulatesFrequencyLog:
    """Regression test: policy_action_log previously had a schema and a
    read query but nothing ever wrote to it, so max_per_minute rules were
    silently permanent no-ops no matter how many times an action fired."""

    def test_record_action_populates_policy_action_log(self, tmp_path) -> None:
        _, pipeline = _runtime_with_pipeline(tmp_path)
        validator = pipeline._policy_validator
        proposal, violations = pipeline.propose_action("speak", inputs={"text": "hi"})
        assert violations == []
        row = validator._conn.execute(
            "SELECT COUNT(*) FROM policy_action_log WHERE action_type = 'speak' AND proposal_id = ?",
            (proposal.proposal_id,),
        ).fetchone()
        assert row[0] == 1

    def test_frequency_limit_engages_after_being_populated(self, tmp_path) -> None:
        _, pipeline = _runtime_with_pipeline(tmp_path)
        validator = pipeline._policy_validator
        validator.register(PolicyRule(
            rule_id="test-speak-cap", name="test speak cap", description="",
            action_type="speak", conditions_json='{"max_per_minute": 1}',
            block=True,
        ))

        _, first_violations = pipeline.propose_action("speak", inputs={"text": "one"})
        assert not any(v.blocked for v in first_violations)

        _, second_violations = pipeline.propose_action("speak", inputs={"text": "two"})
        assert any(v.blocked for v in second_violations)


class TestSpeakPolicyGating:
    """_speak() must propose+validate before synthesizing audio, and
    record the execution outcome afterward -- without ever touching the
    real TTS/audio stack (that's exercised by _speak_impl, stubbed here)."""

    def test_allowed_speech_calls_impl_and_records_success(self, tmp_path) -> None:
        runtime, pipeline = _runtime_with_pipeline(tmp_path)
        calls = []

        async def fake_impl(text, expected_revision=None):
            calls.append(text)
            return True

        runtime._speak_impl = fake_impl

        spoken = asyncio.run(runtime._speak("  hello   there  "))
        assert spoken is True
        assert calls == ["hello there"]

        executions = pipeline._action_store.recent_executions()
        assert len(executions) == 1
        assert executions[0]["success"] is True

    def test_blocked_speech_never_calls_impl(self, tmp_path) -> None:
        runtime, pipeline = _runtime_with_pipeline(tmp_path)
        pipeline._policy_validator.register(PolicyRule(
            rule_id="test-block-all-speech", name="block all speech", description="",
            action_type="speak", conditions_json='{"max_per_minute": 0}',
            block=True,
        ))
        calls = []

        async def fake_impl(text, expected_revision=None):
            calls.append(text)
            return True

        runtime._speak_impl = fake_impl

        spoken = asyncio.run(runtime._speak("this should never be spoken"))
        assert spoken is False
        assert calls == []
        # Blocked proposals don't get an execution recorded -- the effect
        # never happened, so there's nothing to record.
        assert pipeline._action_store.recent_executions() == []

    def test_failed_impl_records_failure(self, tmp_path) -> None:
        runtime, pipeline = _runtime_with_pipeline(tmp_path)

        async def failing_impl(text, expected_revision=None):
            return False

        runtime._speak_impl = failing_impl

        spoken = asyncio.run(runtime._speak("hello"))
        assert spoken is False

        executions = pipeline._action_store.recent_executions()
        assert len(executions) == 1
        assert executions[0]["success"] is False

    def test_empty_text_short_circuits_before_any_proposal(self, tmp_path) -> None:
        runtime, pipeline = _runtime_with_pipeline(tmp_path)

        async def fake_impl(text, expected_revision=None):
            raise AssertionError("_speak_impl must not run for empty text")

        runtime._speak_impl = fake_impl

        spoken = asyncio.run(runtime._speak("   "))
        assert spoken is False
        assert pipeline._action_store.pending_proposals() == []
        assert pipeline._action_store.recent_executions() == []

    def test_cancellation_still_records_execution(self, tmp_path) -> None:
        """A cancelled speak attempt must still be recorded (as a failure)
        before the CancelledError propagates -- the finally block must not
        be skipped."""
        runtime, pipeline = _runtime_with_pipeline(tmp_path)

        async def cancelling_impl(text, expected_revision=None):
            raise asyncio.CancelledError()

        runtime._speak_impl = cancelling_impl

        with pytest.raises(asyncio.CancelledError):
            asyncio.run(runtime._speak("hello"))

        executions = pipeline._action_store.recent_executions()
        assert len(executions) == 1
        assert executions[0]["success"] is False
