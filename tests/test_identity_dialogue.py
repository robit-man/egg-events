import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

from egg_companion.adapters.omnius import OmniusClient
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
    assert OmniusClient.parse_realtime_tool_call(
        "Please search the web for the latest Jetson release"
    ) is None
    assert OmniusClient.parse_realtime_tool_call("How are you?") is None
    assert OmniusClient.parse_realtime_tool_handoff(
        "[[TOOL:WEB_SEARCH|current local weather]]"
    ) == ("web_search", "current local weather")


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

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            if "CURRENT CAMERA TOOL RESULT" not in context:
                return "[[TOOL:VISION]]"
            return "You are holding a dark object."

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
        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
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


def test_realtime_model_intent_signal_routes_live_vision_without_heuristics() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        frozen = np.full((64, 64, 3), 40, dtype=np.uint8)
        boundary = time.monotonic()
        runtime._latest_frames = {"front": (frozen, boundary)}
        runtime._capture_turn_visual_snapshot("heard-semantic-vision", boundary)
        replies = []
        spoken = []

        async def conversation(*args, allow_tool_requests=True, **kwargs):
            replies.append(allow_tool_requests)
            context = args[1]
            if "CURRENT CAMERA TOOL RESULT" not in context:
                return "[[TOOL:VISION]]"
            return "A dark object is visible in front of me."

        async def answer(frames, utterance: str, scene: str):
            assert "Recent ordered conversation" in scene
            return {
                "answer": "A dark object is visible in front of me.",
                "grounded": True,
                "confidence": 0.9,
                "supporting_camera_ids": ["front"],
                "observations": ["A dark object is visible."],
                "uncertainty": None,
            }

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        async def forbidden_router(*args, **kwargs):
            raise AssertionError("the serial dialogue router must remain disabled")

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._omnius.answer_visual_question_analysis = answer  # type: ignore[method-assign]
        runtime._omnius.reason_about_utterance = forbidden_router  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Could you inspect the thing I mean?",
            utterance_id="heard-semantic-vision",
            started_at=boundary - 0.5,
            ended_at=boundary,
        )

        await runtime._handle_audio_turn(turn)

        assert replies == [True, True]
        assert spoken == ["A dark object is visible in front of me."]

    import time
    import numpy as np

    asyncio.run(scenario())


def test_realtime_model_can_chain_camera_evidence_into_selected_ocr_region() -> None:
    async def scenario() -> None:
        import cv2
        import numpy as np
        import time

        runtime = CompanionRuntime(_config())
        frozen = np.full((80, 120, 3), 47, dtype=np.uint8)
        boundary = time.monotonic()
        runtime._latest_frames = {"front": (frozen, boundary)}
        runtime._capture_turn_visual_snapshot("heard-ocr-chain", boundary)
        calls: list[bool] = []
        ocr_candidates = []
        full_frame_ocr_calls = []
        spoken: list[str] = []

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            calls.append(allow_tool_requests)
            if "CURRENT CAMERA TOOL RESULT" not in context:
                return "[[TOOL:VISION]]"
            if "ADVANCED CAMERA OCR TOOL RESULT" not in context:
                return runtime._omnius._realtime_tool_marker(
                    "ocr",
                    {
                        "question": "Read the small status text on the front display.",
                        "camera_ids": ["front"],
                        "region_ids": ["front-display"],
                    },
                )
            assert '"text": "READY 42"' in context
            return "The front display says, ready forty-two."

        async def inspect(frames, utterance: str, scene: str):
            return {
                "answer": "A small display is visible, but its text needs OCR.",
                "grounded": True,
                "confidence": 0.88,
                "supporting_camera_ids": ["front"],
                "camera_observations": [
                    {
                        "camera_id": "front",
                        "observations": ["A small status display is visible."],
                    }
                ],
                "text_candidates": [
                    {
                        "region_id": "front-display",
                        "camera_id": "front",
                        "bbox": [0.2, 0.2, 0.8, 0.7],
                        "description": "small status display",
                        "confidence": 0.9,
                        "visible_text": None,
                        "needs_ocr": True,
                    }
                ],
                "uncertainty": "The characters are too small for the VLM.",
            }

        async def read_regions(candidate, *, explicit_read_request=False):
            ocr_candidates.append(candidate)
            decoded = cv2.imdecode(
                np.frombuffer(candidate.image_png, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            assert decoded is not None and round(float(decoded.mean())) == 47
            assert explicit_read_request
            assert candidate.camera_id == "front"
            assert candidate.vlm_text_regions[0]["region_id"] == "front-display"
            return None

        async def read_full_frame(
            image_png, *, explicit_read_request=False, vlm_text_positive=False
        ):
            full_frame_ocr_calls.append(image_png)
            assert explicit_read_request and vlm_text_positive
            return {
                "text": "READY 42",
                "confidence": 0.96,
                "engine": "test-advanced-ocr",
                "regions": [
                    {
                        "region_id": "front-display",
                        "text": "READY 42",
                        "confidence": 0.96,
                    }
                ],
            }

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._omnius.answer_visual_question_analysis = inspect  # type: ignore[method-assign]
        runtime._ocr_vlm_detected_regions = read_regions  # type: ignore[method-assign]
        runtime._run_advanced_ocr = read_full_frame  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "What does that display say?",
            utterance_id="heard-ocr-chain",
            started_at=boundary - 0.4,
            ended_at=boundary,
        )

        await runtime._handle_audio_turn(turn)

        assert calls == [True, True, True]
        assert len(ocr_candidates) == 1
        assert len(full_frame_ocr_calls) == 1
        assert spoken == ["The front display says, ready forty-two."]

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_web_evidence_back_into_reply() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        contexts = []
        spoken = []

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            contexts.append((context, allow_tool_requests))
            if "WEB SEARCH TOOL EVIDENCE" not in context:
                return "[[TOOL:WEB_SEARCH|current test launch news]]"
            assert "WEB SEARCH TOOL EVIDENCE" in context
            return "The retrieved headline says the test launch succeeded."

        async def web_search(query: str):
            assert query == "current test launch news"
            return "Test launch succeeded — https://example.test/news"

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._omnius.web_search_with_pages = web_search  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Tell me what happened online.",
            utterance_id="heard-semantic-web",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["The retrieved headline says the test launch succeeded."]

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_memory_recall_unavailable_status() -> None:
    """With memory disabled (this file's default _config), the model-selected

    memory tool must still dispatch cleanly and tell the model plainly that
    no memory is available, rather than crashing or inventing a sighting.
    The happy-path recall (memory enabled, real results) is covered in
    tests/test_environmental_cognition.py, which already constructs a
    runtime with a real memory pipeline.
    """

    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        assert runtime._memory is None
        contexts = []
        spoken = []

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            contexts.append((context, allow_tool_requests))
            if "OBJECT MEMORY TOOL STATUS" not in context:
                return "[[TOOL:MEMORY|my keys]]"
            assert "OBJECT MEMORY TOOL STATUS" in context
            return "I don't have any memory of that."

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Where did you last see my keys?",
            utterance_id="heard-semantic-memory",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["I don't have any memory of that."]

    asyncio.run(scenario())


def test_heard_turn_speaks_a_fallback_when_reasoning_raises() -> None:
    """A failed/timed-out model call must still produce audible feedback,

    never silent turn drop. Regression test for a real production incident:
    under heavy load the realtime chat request timed out, and the turn was
    logged and discarded with nothing spoken back.
    """

    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        spoken = []

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            raise TimeoutError("simulated backend timeout")

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Are you there?",
            utterance_id="heard-reasoning-timeout",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert len(spoken) == 1
        assert spoken[0]

    asyncio.run(scenario())


def test_realtime_model_intent_signal_routes_read_only_shell_through_omnius() -> None:
    async def scenario() -> None:
        runtime = CompanionRuntime(_config())
        contexts = []
        spoken = []

        async def conversation(utterance, context, history, *, allow_tool_requests=True):
            contexts.append((context, allow_tool_requests))
            if "READ-ONLY SHELL TOOL EVIDENCE" not in context:
                return "[[TOOL:SHELL|git status --short]]"
            assert "READ-ONLY SHELL TOOL EVIDENCE" in context
            return "The working tree is clean."

        async def plan(request: str, context: str):
            raise AssertionError("native shell command should not require a second model plan")

        async def run(command: str, working_dir: str):
            assert command == "git status --short"
            return "working tree clean"

        async def speak(text: str, expected_revision: int | None = None) -> bool:
            spoken.append(text)
            return True

        runtime._omnius.conversation_reply = conversation  # type: ignore[method-assign]
        runtime._omnius.plan_read_only_shell_command = plan  # type: ignore[method-assign]
        runtime._omnius.run_read_only_shell = run  # type: ignore[method-assign]
        runtime._speak = speak  # type: ignore[method-assign]
        turn = runtime._conversation_turns.finalize_audio_turn(
            "Inspect the repository status.",
            utterance_id="heard-semantic-shell",
            started_at=1.0,
            ended_at=1.2,
        )

        await runtime._handle_audio_turn(turn)

        assert [allow for _, allow in contexts] == [True, True]
        assert spoken == ["The working tree is clean."]

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
