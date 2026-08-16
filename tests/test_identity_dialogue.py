import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from egg_companion.config import EggConfig
from egg_companion.models import AttentionTarget, BoundingBox, Detection, Observation
from egg_companion.runtime import CompanionRuntime, _PendingIdentityQuestion


def _config() -> EggConfig:
    return EggConfig.model_validate(
        {
            "audio": {"input_device": "default", "doa_mode": "disabled"},
            "omnius": {"model": "test", "voice_model": "test"},
            "identity": {"enabled": False},
            "object_learning": {"enabled": False},
            "memory": {"enabled": False},
            "camera_discovery": {"enabled": False},
        }
    )


def _unnamed_target(profile_id: str = "person-specific") -> AttentionTarget:
    now = datetime.now(timezone.utc)
    return AttentionTarget(
        "track-1",
        Detection(
            "person",
            0.98,
            BoundingBox(0, 0, 100, 100),
            {
                "identity_id": profile_id,
                "identity_kind": "face",
                "identity_persistent": True,
                "identity_needs_name": True,
                "identity_sightings": 1,
            },
        ),
        1.0,
        1.0,
        "new person",
        "front",
        now,
    )


def test_stable_unnamed_face_is_asked_once_only_after_audible_publication() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        spoken: list[str] = []

        runtime.identities = SimpleNamespace(
            profile_record=lambda profile_id: {
                "profile_id": profile_id,
                "first_seen": datetime.now(timezone.utc),
                "last_seen": datetime.now(timezone.utc),
                "sightings": 3,
                "samples": 3,
                "last_camera": "front",
            }
        )

        async def compose(*args, **kwargs):
            return {
                "speak": True,
                "question": "Hi—what would you like me to call you?",
                "reason": "A natural introduction is appropriate.",
                "confidence": 0.9,
            }

        runtime._omnius.compose_identity_question = compose  # type: ignore[method-assign]

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._speak = speak  # type: ignore[method-assign]

        assert await runtime._maybe_ask_identity_name(_unnamed_target())
        assert not await runtime._maybe_ask_identity_name(_unnamed_target())
        assert spoken == ["Hi—what would you like me to call you?"]
        snapshot = runtime.telemetry.snapshot(runtime.config)
        assert snapshot["identity_dialogue"]["state"] == "awaiting_name"
        assert snapshot["identity_dialogue"]["profile_id"] == "person-specific"

    asyncio.run(scenario())


def test_failed_question_publication_does_not_consume_the_profile() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())

        runtime.identities = SimpleNamespace(
            profile_record=lambda profile_id: {
                "profile_id": profile_id,
                "first_seen": datetime.now(timezone.utc),
                "last_seen": datetime.now(timezone.utc),
                "sightings": 3,
                "samples": 3,
                "last_camera": "front",
            }
        )

        async def compose(*args, **kwargs):
            return {
                "speak": True,
                "question": "Would you tell me what you prefer to be called?",
                "reason": "A grounded introduction is appropriate.",
                "confidence": 0.9,
            }

        runtime._omnius.compose_identity_question = compose  # type: ignore[method-assign]

        async def cannot_speak(text: str, expected_revision: int | None = None) -> bool:
            return False

        runtime._speak = cannot_speak  # type: ignore[method-assign]

        assert not await runtime._maybe_ask_identity_name(_unnamed_target())
        assert "person-specific" not in runtime._identity_name_questions
        assert runtime._pending_identity_name is None

    asyncio.run(scenario())


def test_bare_name_answer_bypasses_general_router_and_targets_prompted_face() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        prompted = _PendingIdentityQuestion(
            "person-prompted", "front", datetime.now(timezone.utc), float("inf")
        )
        runtime._pending_identity_name = prompted
        accepted: list[tuple[str, str, str]] = []

        async def interpret(text: str, *, prompted: bool = False) -> str | None:
            assert prompted
            assert text == "Troy"
            return "Troy"

        async def route(*args, **kwargs):
            raise AssertionError("a prompted bare name must not wait on general routing")

        async def accept(
            profile_id: str,
            name: str,
            transcript: str,
            expected_revision: int,
            camera_id: str | None,
        ) -> bool:
            accepted.append((profile_id, name, camera_id or ""))
            return True

        runtime._omnius.interpret_person_naming = interpret  # type: ignore[method-assign]
        runtime._omnius.reason_about_utterance = route  # type: ignore[method-assign]
        runtime._accept_identity_name = accept  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Troy", utterance_id="heard-1", started_at=1.0, ended_at=1.2
        )

        await runtime._handle_audio_turn(turn)

        assert accepted == [("person-prompted", "Troy", "front")]

    asyncio.run(scenario())


def test_name_binding_uses_exact_profile_and_acknowledges_with_name() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        named: list[tuple[str, str]] = []
        spoken: list[tuple[str, int | None]] = []

        class Identities:
            @staticmethod
            def name_profile(profile_id: str, name: str):
                named.append((profile_id, name))
                return SimpleNamespace(profile_id=profile_id, name=name)

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append((text, expected_revision))
            return True

        async def acknowledge(*args, **kwargs) -> str:
            return "Got it, Troy—I'll remember that."

        runtime.identities = Identities()  # type: ignore[assignment]
        runtime._speak = speak  # type: ignore[method-assign]
        runtime._omnius.compose_identity_acknowledgement = acknowledge  # type: ignore[method-assign]
        runtime._pending_identity_name = _PendingIdentityQuestion(
            "person-prompted", "front", datetime.now(timezone.utc), float("inf")
        )

        assert await runtime._accept_identity_name(
            "person-prompted", "Troy", "Troy", 7, "front"
        )
        assert named == [("person-prompted", "Troy")]
        assert spoken == [("Got it, Troy—I'll remember that.", 7)]
        assert runtime._pending_identity_name is None
        identity_state = runtime.telemetry.snapshot(runtime.config)["identity_dialogue"]
        assert identity_state["state"] == "named"
        assert identity_state["name"] == "Troy"

    asyncio.run(scenario())


def test_web_search_requires_model_authored_tool_route() -> None:
    assert CompanionRuntime._web_search_query(
        "Please search the web for the latest Jetson release", None
    ) is None
    assert CompanionRuntime._web_search_query("How are you?", None) is None
    assert CompanionRuntime._web_search_query(
        "anything", {"tool": "web_search", "tool_query": "current local weather"}
    ) == "current local weather"


def test_visual_turn_uses_frames_frozen_at_utterance_boundary() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        runtime.config.omnius.dialogue_router_enabled = True
        frozen = np.full((64, 64, 3), 40, dtype=np.uint8)
        replacement = np.full((64, 64, 3), 220, dtype=np.uint8)
        boundary = time.monotonic()
        runtime._latest_frames = {"front": (frozen, boundary)}
        snapshot = runtime._capture_turn_visual_snapshot("heard-vision", boundary)
        runtime._latest_frames["front"] = (replacement, time.monotonic())
        received: list[tuple[str, bytes, str]] = []
        spoken: list[str] = []

        async def route(utterance: str, context: str):
            assert "ASR-boundary=" in context
            return {
                "directed": True,
                "act": "question",
                "confidence": 0.99,
                "tool": "vision",
                "tool_query": utterance,
            }

        async def answer(frames, utterance: str, scene: str):
            received.extend(frames)
            return {
                "answer": "You are holding a dark object.",
                "grounded": True,
                "confidence": 0.9,
                "supporting_camera_ids": ["front"],
                "observations": ["A dark object is visible."],
                "uncertainty": None,
            }

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.reason_about_utterance = route  # type: ignore[method-assign]
        runtime._omnius.answer_visual_question_analysis = answer  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "What am I holding?",
            utterance_id="heard-vision",
            started_at=boundary - 0.5,
            ended_at=boundary,
        )

        await runtime._handle_audio_turn(turn)

        assert snapshot.frames[0].frame.mean() == 40
        assert [item[0] for item in received] == ["front"]
        assert spoken == ["You are holding a dark object."]

    import time
    import numpy as np

    asyncio.run(scenario())


def test_model_authored_question_requires_and_addresses_named_visible_person() -> None:
    async def scenario() -> None:
        config = _config()
        config.attention.proactive_speech_enabled = True
        runtime = CompanionRuntime(config)
        spoken: list[str] = []

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._speak = speak  # type: ignore[method-assign]

        async def compose_curiosity(candidate, visible_people, scene, history, strategy):
            return {
                "speak": True,
                "question": "Cole, what do you use that keyboard for?",
                "reason": "The grounded unresolved relationship is relevant now.",
                "confidence": 0.9,
            }

        runtime._omnius.compose_curiosity_question = compose_curiosity  # type: ignore[method-assign]
        now = datetime.now(timezone.utc)
        object_detection = Detection(
            "keyboard", 0.95, BoundingBox(0, 0, 50, 50), {"object_id": "object-1"}
        )
        runtime._latest_observations = {
            "front": Observation("front", now, (object_detection,))
        }
        result = {
            "observation_policy": {
                "open_questions": [
                    {
                        "summary": "I've seen the keyboard a few times. What do you use it for?",
                        "reason": "The dream agent chose to ask while the object is present.",
                        "action": "ask",
                        "predicate": "used_for",
                        "confidence": 0.9,
                        "entity_ids": ["object-1"],
                        "evidence_ids": [],
                        "context_ids": [],
                    }
                ],
                "attend_to": [],
            }
        }
        assert not await runtime._maybe_ask_default_mode_question(result)

        person_detection = Detection(
            "person", 0.99, BoundingBox(0, 0, 100, 100),
            {
                "identity_id": "person-1",
                "identity": "Cole",
                "identity_persistent": True,
                "identity_needs_name": False,
            },
        )
        runtime._latest_observations["front"] = Observation(
            "front", now, (object_detection, person_detection)
        )
        assert await runtime._maybe_ask_default_mode_question(result)
        assert spoken == [
            "Cole, what do you use that keyboard for?"
        ]

    asyncio.run(scenario())
