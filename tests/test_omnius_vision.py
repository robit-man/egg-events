import asyncio
import io
import json
import wave

from egg_companion.adapters.omnius import OmniusClient
from egg_companion.config import OmniusConfig


def test_vlm_object_response_requires_bounded_json_label() -> None:
    assert OmniusClient.parse_object_classification('{"label":"ceramic mug","confidence":0.91}') == ("ceramic mug", 0.91)
    assert OmniusClient.parse_object_classification('{"label":null,"confidence":0.91}') is None
    assert OmniusClient.parse_object_classification('a mug') is None
    assert OmniusClient.parse_object_classification('{"label":"mug","confidence":1.2}') is None


def test_vlm_object_analysis_reports_grounded_text_regions_without_changing_label_api() -> None:
    payload = (
        '{"label":"cotton shirt","confidence":0.91,"visible_text":true,'
        '"text_regions":["shirt front"]}'
    )

    assert OmniusClient.parse_object_analysis(payload) == {
        "label": "cotton shirt",
        "confidence": 0.91,
        "visible_text": True,
        "text_regions": ["shirt front"],
    }
    assert OmniusClient.parse_object_classification(payload) == (
        "cotton shirt",
        0.91,
    )


def test_temporal_person_comparison_requires_bounded_explainable_json() -> None:
    response = (
        '{"same_person":true,"confidence":0.93,'
        '"analysis":"The dark jacket and carried cup persist across both masks.",'
        '"displacement_analysis":"The mask centroid moves 18 px right and 3 px down.",'
        '"visible_correspondences":["dark jacket","same carried cup"]}'
    )

    assert OmniusClient.parse_temporal_person_comparison(response) == {
        "same_person": True,
        "confidence": 0.93,
        "analysis": "The dark jacket and carried cup persist across both masks.",
        "displacement_analysis": "The mask centroid moves 18 px right and 3 px down.",
        "visible_correspondences": ["dark jacket", "same carried cup"],
    }
    assert OmniusClient.parse_temporal_person_comparison("same person") is None
    assert OmniusClient.parse_temporal_person_comparison(
        '{"same_person":true,"confidence":1.2,"analysis":"same",'
        '"displacement_analysis":"right"}'
    ) is None
    assert OmniusClient.parse_temporal_person_comparison(
        '{"same_person":"yes","confidence":0.8,"analysis":"same",'
        '"displacement_analysis":"right"}'
    ) is None


def test_identity_merge_comparison_requires_bounded_explainable_json() -> None:
    """The VLM gate for offline dream identity merges (added after a
    face-embedding-only consensus mislabeled a well-established profile)
    must reject malformed/underspecified completions the same way the
    other Ornith comparison parsers do."""
    response = (
        '{"same_person":true,"confidence":0.82,'
        '"analysis":"Matching jawline, brow shape, and a visible scar above the left eyebrow.",'
        '"visible_correspondences":["scar above left eyebrow","jawline shape"],'
        '"visible_conflicts":[]}'
    )

    assert OmniusClient.parse_identity_merge_comparison(response) == {
        "same_person": True,
        "confidence": 0.82,
        "analysis": "Matching jawline, brow shape, and a visible scar above the left eyebrow.",
        "visible_correspondences": ["scar above left eyebrow", "jawline shape"],
        "visible_conflicts": [],
    }
    assert OmniusClient.parse_identity_merge_comparison("same person") is None
    assert OmniusClient.parse_identity_merge_comparison(
        '{"same_person":true,"confidence":1.4,"analysis":"same"}'
    ) is None
    assert OmniusClient.parse_identity_merge_comparison(
        '{"same_person":"maybe","confidence":0.8,"analysis":"same"}'
    ) is None
    assert OmniusClient.parse_identity_merge_comparison(
        '{"same_person":false,"confidence":0.9,"analysis":""}'
    ) is None


def test_audio_classification_parser_accepts_only_numeric_yamnet_output() -> None:
    output = (
        'Audio scene classification:\n'
        '{"success":true,"classifications":['
        '{"class":"Speech","score":0.6703},'
        '{"class":"Vehicle","score":0.0914}],'
        '"total_classes":521,"duration_s":6.0}'
    )

    parsed = OmniusClient._parse_audio_classification(output)

    assert parsed == {
        "classifications": [
            {"label": "Speech", "confidence": 0.6703},
            {"label": "Vehicle", "confidence": 0.0914},
        ],
        "total_classes": 521,
        "duration_s": 6.0,
    }
    assert OmniusClient._parse_audio_classification("mock ambient evidence") is None


def test_audio_classification_parser_accepts_structured_omnius_10628_data() -> None:
    parsed = OmniusClient._normalize_audio_classification(
        {
            "classifications": [
                {"label": "Speech", "confidence": 0.81},
                {"class": "Music", "score": 0.19},
            ],
            "total_classes": 521,
            "duration_seconds": 4.5,
            "model": "yamnet",
            "backend": "tensorrt-fp16",
            "taxonomy": "AudioSet-521",
        }
    )

    assert parsed == {
        "classifications": [
            {"label": "Speech", "confidence": 0.81},
            {"label": "Music", "confidence": 0.19},
        ],
        "total_classes": 521,
        "duration_s": 4.5,
        "model": "yamnet",
        "backend": "tensorrt-fp16",
        "taxonomy": "AudioSet-521",
    }


def test_pause_daemon_listen_uses_voice_stop_without_disabling_tts(monkeypatch) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            assert url.endswith("/v1/voice/stop")
            assert "json" not in kwargs
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(voice_model="supertonic"))

    asyncio.run(client.pause_daemon_listen())


def test_model_catalog_lists_locally_pulled_ollama_models(monkeypatch) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def raise_for_status(self):
            return None

        async def json(self):
            return {
                "models": [
                    {
                        "name": "robit/ornith-1.5:9b",
                        "size": 6550814941,
                        "details": {
                            "family": "qwen35",
                            "parameter_size": "9.0B",
                            "quantization_level": "Q4_K_M",
                        },
                        "capabilities": ["completion", "tools", "vision"],
                    },
                    {"name": "", "size": 123},
                ]
            }

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def get(self, url, **kwargs):
            assert url.endswith("/api/tags")
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    catalog = asyncio.run(client.model_catalog())

    assert catalog == [
        {
            "name": "robit/ornith-1.5:9b",
            "size": 6550814941,
            "family": "qwen35",
            "parameter_size": "9.0B",
            "quantization_level": "Q4_K_M",
            "capabilities": ["completion", "tools", "vision"],
        }
    ]


def test_audio_classifier_health_accepts_structured_503_readiness(monkeypatch) -> None:
    class Response:
        status = 503

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {
                "ready": False,
                "backend": "tensorrt-fp16",
                "last_error": "CUDA unavailable",
            }

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def get(self, url, **kwargs):
            assert url.endswith("/v1/audio/classify/health")
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    result = asyncio.run(client.audio_classifier_health())

    assert result == {
        "supported": True,
        "ready": False,
        "backend": "tensorrt-fp16",
        "last_error": "CUDA unavailable",
    }


def test_realtime_chat_streams_content_deltas_and_calls_on_delta(monkeypatch) -> None:
    lines = [
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n',
        b'data: {"choices":[{"delta":{"content":" there"}}]}\n',
        b"data: [DONE]\n",
    ]

    class StreamContent:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    class Response:
        status = 200

        def __init__(self):
            self.content = StreamContent(lines)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def text(self):
            return ""

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            assert kwargs["json"]["stream"] is True
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))
    deltas: list[str] = []

    reply = asyncio.run(
        client.conversation_reply(
            "hi", "context", [], allow_tool_requests=False, on_delta=deltas.append
        )
    )

    assert deltas == ["Hello", " there"]
    assert reply == "Hello there"


def test_realtime_chat_streams_fragmented_tool_call_arguments(monkeypatch) -> None:
    lines = [
        b'data: {"choices":[{"delta":{"tool_calls":'
        b'[{"index":0,"function":{"name":"search_current_web"}}]}}]}\n',
        b'data: {"choices":[{"delta":{"tool_calls":'
        b'[{"index":0,"function":{"arguments":"{\\"qu"}}]}}]}\n',
        b'data: {"choices":[{"delta":{"tool_calls":'
        b'[{"index":0,"function":{"arguments":"ery\\": \\"weather\\"}"}}]}}]}\n',
        b"data: [DONE]\n",
    ]

    class StreamContent:
        def __init__(self, items):
            self._items = list(items)

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self._items:
                raise StopAsyncIteration
            return self._items.pop(0)

    class Response:
        status = 200

        def __init__(self):
            self.content = StreamContent(lines)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def text(self):
            return ""

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    reply = asyncio.run(
        client.conversation_reply(
            "look up the weather", "context", [],
            allow_tool_requests=True, on_delta=lambda _delta: None,
        )
    )

    assert OmniusClient.parse_realtime_tool_call(reply) == {
        "tool": "web_search",
        "arguments": {"query": "weather"},
    }


def test_realtime_chat_falls_back_to_non_streaming_on_transport_failure(monkeypatch) -> None:
    class FailingStreamResponse:
        status = 200

        async def __aenter__(self):
            raise ConnectionError("stream reset")

        async def __aexit__(self, *_):
            return None

    class NonStreamResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"choices": [{"message": {"content": "fallback reply"}}]}

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            return FailingStreamResponse() if kwargs["json"]["stream"] else NonStreamResponse()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    reply = asyncio.run(
        client.conversation_reply(
            "hi", "context", [], allow_tool_requests=False, on_delta=lambda _delta: None
        )
    )

    assert reply == "fallback reply"


def test_environmental_assessment_uses_the_shared_model_context_budget(monkeypatch) -> None:
    from io import BytesIO

    from PIL import Image

    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"message": {"content": "not valid json, only the request matters here"}}

    class Session:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    image = BytesIO()
    Image.new("RGB", (640, 480), "orange").save(image, format="JPEG")
    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    asyncio.run(
        client.assess_environmental_change(
            [("front", image.getvalue(), "2026-01-01T00:00:00+00:00")],
            {"kind": "person_count_changed"},
            None,
        )
    )

    assert len(requests) == 1
    options = requests[0][1]["json"]["options"]
    assert options["num_ctx"] == client.config.model_num_ctx


def test_realtime_chat_and_environmental_assessment_request_identical_num_ctx(
    monkeypatch,
) -> None:
    """`model` and `vision_model` are the same loaded Ollama instance in this
    deployment (one GPU, one serving slot). Any two calls against it that
    request different num_ctx values force a full model reload between
    them -- confirmed directly on this device: 48 reloads in two hours,
    audible churning, 40-90s+ turn latency, dropped replies, all from
    exactly that mismatch. Every payload must request the same num_ctx.
    """
    from io import BytesIO

    from PIL import Image

    requests = []

    class ChatResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"choices": [{"message": {"content": "hello"}}]}

    class EnvResponse:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"message": {"content": "not valid json, only the request matters here"}}

    class Session:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return EnvResponse() if url.endswith("/api/chat") else ChatResponse()

    image = BytesIO()
    Image.new("RGB", (640, 480), "orange").save(image, format="JPEG")
    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    asyncio.run(client.conversation_reply("hi", "context", [], allow_tool_requests=False))
    asyncio.run(
        client.assess_environmental_change(
            [("front", image.getvalue(), "2026-01-01T00:00:00+00:00")],
            {"kind": "person_count_changed"},
            None,
        )
    )

    assert len(requests) == 2

    def _num_ctx(body: dict) -> int:
        # The Omnius gateway (/v1/chat) takes num_ctx top-level; raw Ollama
        # (/api/chat) nests it under "options" -- same underlying knob.
        return body.get("num_ctx", body.get("options", {}).get("num_ctx"))

    num_ctx_values = {_num_ctx(req[1]["json"]) for req in requests}
    assert len(num_ctx_values) == 1, (
        f"requests targeting the same shared model must agree on num_ctx, got {num_ctx_values}"
    )


def test_cognition_health_uses_lightweight_backend_readiness(monkeypatch) -> None:
    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"status": "ready", "backend": "reachable", "type": "ollama"}

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def get(self, url, **kwargs):
            assert url.endswith("/health/ready")
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    result = asyncio.run(client.cognition_health())

    assert result == {
        "supported": True,
        "status": "ready",
        "backend": "reachable",
        "type": "ollama",
    }


def test_audio_classifier_prefers_new_structured_endpoint(monkeypatch) -> None:
    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {
                "result": {
                    "success": True,
                    "data": {
                        "classifications": [
                            {"label": "Speech", "confidence": 0.9}
                        ]
                    },
                }
            }

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs, self.timeout.total))
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    result = asyncio.run(
        client._call_audio_classifier(
            {"action": "classify", "file": "/tmp/input.wav"},
            timeout_seconds=90,
        )
    )

    assert result["data"]["classifications"][0]["label"] == "Speech"
    assert requests[0][0].endswith("/v1/audio/classify")
    assert requests[0][1]["json"]["timeout_ms"] == 90000
    assert requests[0][2] == 95


def test_advanced_ocr_uses_omnius_10629_structured_contract(monkeypatch) -> None:
    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {
                "result": {
                    "success": True,
                    "data": {
                        "text": "WELCOME\nGate 3",
                        "confidence": 91.5,
                        "regions": {"header": "WELCOME", "body": "Gate 3"},
                        "variant": "adaptive_31_psm11",
                        "variants_tested": 24,
                    },
                }
            }

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs, self.timeout.total))
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(
        OmniusConfig(model="test", voice_model="test", timeout_seconds=90)
    )

    result = asyncio.run(client.ocr_advanced("/tmp/screen.png"))

    assert result == {
        "text": "WELCOME\nGate 3",
        "vision_used": False,
        "engine": "omnius-ocr-image-advanced",
        "confidence": 0.915,
        "regions": [
            {"name": "header", "text": "WELCOME"},
            {"name": "body", "text": "Gate 3"},
        ],
        "variant": "adaptive_31_psm11",
        "variants_tested": 24,
    }
    assert requests[0][0].endswith("/v1/ocr/advanced")
    assert requests[0][1]["json"]["args"] == {
        "image": "/tmp/screen.png",
        "language": "eng",
        "regions": True,
    }
    assert requests[0][1]["json"]["timeout_ms"] == 90000
    assert requests[0][2] == 95


def test_direct_tool_timeout_is_forwarded_to_omnius_executor(monkeypatch) -> None:
    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {"result": {"success": True, "output": "ready"}}

    class Session:
        def __init__(self, *, timeout):
            self.timeout = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs, self.timeout.total))
            return Response()

    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    result = asyncio.run(
        client._call_tool("audio_analyze", {"action": "classify"}, timeout_seconds=90)
    )

    assert result["output"] == "ready"
    assert requests[0][1]["json"]["timeout_ms"] == 90000
    assert requests[0][2] == 95


def test_person_name_parser_requires_explicit_bounded_json_name() -> None:
    assert OmniusClient.parse_person_name('{"name":"Ada Lovelace"}') == "Ada Lovelace"
    assert OmniusClient.parse_person_name('{"name":null}') is None
    assert OmniusClient.parse_person_name('My name is Ada') is None


def test_dialogue_router_preserves_web_search_tool_contract() -> None:
    async def scenario() -> None:
        client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

        async def structured(prompt: str) -> str:
            return (
                '{"directed":true,"act":"question","confidence":0.98,'
                '"tool":"web_search","tool_query":"current Jetson Linux release"}'
            )

        client._structured_chat = structured  # type: ignore[method-assign]
        result = await client.reason_about_utterance("What is current?", "scene")
        assert result is not None
        assert result["tool"] == "web_search"
        assert result["tool_query"] == "current Jetson Linux release"

    asyncio.run(scenario())


def test_dialogue_router_can_select_live_vision_without_phrase_rules() -> None:
    async def scenario() -> None:
        client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

        async def structured(prompt: str, **kwargs) -> str:
            return (
                '{"directed":true,"act":"question","confidence":0.99,'
                '"tool":"vision","tool_query":"inspect the presently indicated item"}'
            )

        client._structured_chat = structured  # type: ignore[method-assign]
        result = await client.reason_about_utterance("Could you check this?", "scene")
        assert result is not None
        assert result["tool"] == "vision"
        assert result["tool_query"] == "inspect the presently indicated item"

    asyncio.run(scenario())


def test_realtime_tool_contract_executes_polite_capability_requests_semantically() -> None:
    async def scenario() -> None:
        client = OmniusClient(OmniusConfig(model="test", voice_model="test"))
        captured: list[dict[str, str]] = []

        async def realtime_chat(messages, *, allow_tool_requests, on_delta=None) -> str:
            assert allow_tool_requests
            captured.extend(messages)
            return "[[TOOL:WEB_SEARCH]]"

        client._realtime_chat = realtime_chat  # type: ignore[method-assign]
        reply = await client.conversation_reply(
            "Can you look up the news?", "No current news evidence is present.", []
        )

        assert reply == "[[TOOL:WEB_SEARCH]]"
        contract = next(
            item["content"]
            for item in captured
            if item["role"] == "system" and "polite request to perform it now" in item["content"]
        )
        assert "polite request to perform it now" in contract
        assert "instead of answering only that you can" in contract
        assert "explicitly asks only about capabilities" in contract
        assert "semantic decisions from complete context" in contract

    asyncio.run(scenario())


def test_realtime_tool_signals_are_exact_and_semantic_only() -> None:
    assert OmniusClient.parse_realtime_tool_request("[[TOOL:VISION]]") == "vision"
    assert OmniusClient.parse_realtime_tool_request(" [[ tool: web_search ]] ") == "web_search"
    assert OmniusClient.parse_realtime_tool_handoff(
        "[[TOOL:WEB_SEARCH|top current news headlines]]"
    ) == ("web_search", "top current news headlines")
    assert OmniusClient.parse_realtime_tool_request("[[TOOL:SHELL]]") == "shell"
    assert OmniusClient.parse_realtime_tool_request("I could use the shell") is None
    assert OmniusClient.parse_realtime_tool_request("I need local evidence. [[TOOL:SHELL]]") == "shell"
    assert OmniusClient.parse_realtime_tool_handoff(
        "[[TOOL:SHELL|systemctl status egg]]"
    ) == ("shell", "systemctl status egg")
    assert OmniusClient.parse_realtime_tool_request(
        "[[TOOL:SHELL]] [[TOOL:WEB_SEARCH]]"
    ) is None
    marker = OmniusClient._realtime_tool_marker(
        "ocr",
        {
            "question": "Read the status display.",
            "camera_ids": ["front"],
            "region_ids": ["display-1"],
        },
    )
    assert OmniusClient.parse_realtime_tool_call(marker) == {
        "tool": "ocr",
        "arguments": {
            "question": "Read the status display.",
            "camera_ids": ["front"],
            "region_ids": ["display-1"],
        },
    }
    assert OmniusClient.parse_realtime_tool_request(marker) == "ocr"
    assert any(
        item["function"]["name"] == "read_current_camera_text"
        for item in OmniusClient._realtime_tool_definitions()
    )
    assert OmniusClient.parse_realtime_tool_request("[[TOOL:MEMORY]]") == "memory"
    assert OmniusClient.parse_realtime_tool_handoff("[[TOOL:MEMORY|my keys]]") == (
        "memory",
        "my keys",
    )
    memory_marker = OmniusClient._realtime_tool_marker("memory", {"query": "my keys"})
    assert OmniusClient.parse_realtime_tool_call(memory_marker) == {
        "tool": "memory",
        "arguments": {"query": "my keys"},
    }
    assert any(
        item["function"]["name"] == "recall_object_memory"
        for item in OmniusClient._realtime_tool_definitions()
    )
    timed_memory_marker = OmniusClient._realtime_tool_marker(
        "memory",
        {
            "query": "my keys",
            "since": "2026-08-24T00:00:00+00:00",
            "until": "2026-08-31T00:00:00+00:00",
        },
    )
    assert OmniusClient.parse_realtime_tool_call(timed_memory_marker) == {
        "tool": "memory",
        "arguments": {
            "query": "my keys",
            "since": "2026-08-24T00:00:00+00:00",
            "until": "2026-08-31T00:00:00+00:00",
        },
    }
    recall_schema = next(
        item["function"]
        for item in OmniusClient._realtime_tool_definitions()
        if item["function"]["name"] == "recall_object_memory"
    )
    assert {"since", "until"} <= set(recall_schema["parameters"]["properties"])
    assert "enum" not in recall_schema["parameters"]["properties"]["since"]
    assert OmniusClient._normalized_iso_datetime("2026-08-24T00:00:00+00:00") == (
        "2026-08-24T00:00:00+00:00"
    )
    assert OmniusClient._normalized_iso_datetime("yesterday") is None
    assert OmniusClient._normalized_iso_datetime(None) is None
    assert OmniusClient.parse_realtime_tool_request("[[TOOL:PAST_OCR]]") == "past_ocr"
    assert OmniusClient.parse_realtime_tool_handoff("[[TOOL:PAST_OCR|the sign]]") == (
        "past_ocr",
        "the sign",
    )
    past_ocr_marker = OmniusClient._realtime_tool_marker(
        "past_ocr", {"query": "the sign", "since": "2026-08-24T00:00:00+00:00"}
    )
    assert OmniusClient.parse_realtime_tool_call(past_ocr_marker) == {
        "tool": "past_ocr",
        "arguments": {"query": "the sign", "since": "2026-08-24T00:00:00+00:00"},
    }
    assert any(
        item["function"]["name"] == "read_past_camera_text"
        for item in OmniusClient._realtime_tool_definitions()
    )


def test_web_search_urls_are_parsed_only_from_typed_result_fields() -> None:
    evidence = (
        "1. First\n   URL: https://example.com/one\n\n"
        "Natural language mentioning https://ignored.example/ is not a URL field.\n\n"
        "2. Second\n   URL: https://example.com/two\n"
    )
    assert OmniusClient.web_search_result_urls(evidence) == [
        "https://example.com/one",
        "https://example.com/two",
    ]


def test_read_only_shell_policy_is_structural_not_intent_routing() -> None:
    assert OmniusClient.validate_read_only_shell_command("git status --short")[0]
    assert OmniusClient.validate_read_only_shell_command(
        "systemctl --user status egg-companion.service"
    )[0]
    assert not OmniusClient.validate_read_only_shell_command("git reset --hard")[0]
    assert not OmniusClient.validate_read_only_shell_command("ls | head")[0]
    assert not OmniusClient.validate_read_only_shell_command("cat /etc/shadow")[0]


def test_visual_question_packs_all_frozen_views_into_one_labeled_contact_sheet(
    monkeypatch,
) -> None:
    from io import BytesIO
    from PIL import Image

    requests = []

    class Response:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        async def json(self):
            return {
                "message": {
                    "content": json.dumps(
                        {
                            "answer": "A current object is visible.",
                            "grounded": True,
                            "confidence": 0.9,
                            "supporting_camera_ids": ["front"],
                            "camera_observations": [
                                {
                                    "camera_id": "front",
                                    "observations": ["The front tile contains the object."],
                                },
                                {"camera_id": "side", "observations": []},
                            ],
                            "text_candidates": [],
                            "uncertainty": None,
                        }
                    )
                }
            }

    class Session:
        def __init__(self, *, timeout):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_):
            return None

        def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    image = BytesIO()
    Image.new("RGB", (640, 480), "orange").save(image, format="JPEG")
    monkeypatch.setattr("egg_companion.adapters.omnius.aiohttp.ClientSession", Session)
    client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

    result = asyncio.run(
        client.answer_visual_question_analysis(
            [
                ("front", image.getvalue(), "2026-01-01T00:00:00+00:00"),
                ("side", image.getvalue(), "2026-01-01T00:00:00+00:00"),
            ],
            "What is here?",
            "two current views",
        )
    )

    assert result is not None and result["supporting_camera_ids"] == ["front"]
    content = requests[0][1]["json"]["messages"][0]["content"]
    assert "row 1, column 1" in content and "row 1, column 2" in content
    assert len(requests[0][1]["json"]["messages"][0]["images"]) == 1


def test_social_reflection_requires_revisable_evidence_bound_contract() -> None:
    payload = json.dumps(
        {
            "momentary_affect": {
                "label": "frustrated but engaged",
                "valence": -0.45,
                "arousal": 0.72,
                "confidence": 0.78,
                "evidence": "The speaker directly criticizes the delayed reply.",
            },
            "communicative_behavior": {
                "summary": "The speaker gives direct corrective feedback.",
                "confidence": 0.91,
                "evidence": "The utterance names the failure and requests a change.",
            },
            "relationship_update": {
                "summary": "This turn supplies a preference for rapid grounded answers.",
                "confidence": 0.8,
                "evidence": "The correction explicitly contrasts delay with the desired behavior.",
            },
            "response_feedback": {
                "summary": "The prior response was too slow and insufficiently visual.",
                "confidence": 0.88,
                "evidence": "The speaker reports both defects in the current turn.",
            },
            "strategy_revision": {
                "directive": "Prioritize current sensor evidence and answer directly before elaborating.",
                "rationale": "Explicit feedback favored fast grounded replies.",
                "confidence": 0.86,
            },
            "profile_updates": [
                {
                    "subject_id": "person-001",
                    "summary": "Directly communicates desired system behavior.",
                    "sentiment_trajectory": "Frustration in this turn remains engagement with the task.",
                    "communication_patterns": ["Provides concrete corrective feedback."],
                    "interaction_preferences": ["Explicitly requested rapid grounded answers."],
                    "uncertainties": ["No evidence yet that this preference generalizes."],
                    "confidence": 0.81,
                    "evidence": "The current utterance explicitly requests the change.",
                }
            ],
        }
    )

    parsed = OmniusClient.parse_social_reflection(payload)

    assert parsed is not None
    assert parsed["strategy_revision"]["directive"].startswith("Prioritize")
    assert parsed["profile_updates"][0]["subject_id"] == "person-001"
    assert OmniusClient.parse_social_reflection(
        payload.replace('"confidence": 0.78', '"confidence": 1.8', 1)
    ) is None


def test_narrative_dream_contracts_are_strict_and_model_owned() -> None:
    async def scenario() -> None:
        client = OmniusClient(OmniusConfig(model="test", voice_model="test"))
        responses = iter(
            [
                '{"tool_requests":[{"tool":"memory_search","query":"prior encounter",'
                '"entity_ids":[],"evidence_ids":[],"purpose":"retrieve associated context"}],'
                '"planning_summary":"The model chose one local retrieval."}',
                '{"narrative_summary":"A grounded account.",'
                '"story_update":"The account changed after retrieved context.",'
                '"themes":[],"episodes":[]}',
                '{"unresolved_questions":[],"learned_context":[],'
                '"observation_policy":{"summary":"Remain responsive to grounded change.",'
                '"attend_to":[{"summary":"Ask about the artifact.","reason":"The model selected it.",'
                '"action":"ask","predicate":"used_for","confidence":0.9,"entity_ids":[],'
                '"evidence_ids":[],"context_ids":[]}],"deprioritize":[],"open_questions":[]},'
                '"constitution_update":null}',
                '{"accepted":true,"constitution":"Integrate evidence through explicit, revisable associations.",'
                '"review_summary":"The update changes the general method without encoding a lived fact."}',
                '{"narrative_summary":"A grounded account.",'
                '"story_update":"A changed account.","themes":[],"episodes":[],'
                '"unexpected":"rejected"}',
                '{}',
            ]
        )

        async def structured(prompt: str, *, max_tokens: int) -> str:
            assert prompt
            assert max_tokens > 0
            return next(responses)

        client._narrative_structured_chat = structured  # type: ignore[method-assign]
        plan = await client.plan_narrative_dream({}, {}, {})
        assert plan is not None
        assert plan["tool_requests"][0]["tool"] == "memory_search"
        synthesis = await client.synthesize_narrative_dream({}, {}, {}, plan, [])
        assert synthesis is not None
        assert synthesis["narrative_summary"] == "A grounded account."
        assert synthesis["observation_policy"]["attend_to"][0]["action"] == "ask"
        review = await client.review_narrative_constitution_update(
            {}, "Integrate evidence through explicit associations."
        )
        assert review is not None and review["accepted"] is True
        assert await client.synthesize_narrative_dream({}, {}, {}, plan, []) is None

    asyncio.run(scenario())


def test_split_narrative_contracts_reject_semantic_shape_drift() -> None:
    assert OmniusClient._parse_narrative_core(
        '{"narrative_summary":"Grounded.","story_update":"Changed.",'
        '"themes":[],"episodes":[]}'
    ) is not None
    assert OmniusClient._parse_narrative_reflection(
        '{"unresolved_questions":[],"learned_context":[],"observation_policy":'
        '{"summary":"Continue grounding.","attend_to":[],"deprioritize":[],'
        '"open_questions":[]},"constitution_update":null}'
    ) is not None
    assert OmniusClient._parse_narrative_core(
        '{"narrative_summary":"Grounded.","story_update":"Changed.",'
        '"themes":[],"episodes":[],"invented":true}'
    ) is None


def test_narrative_reflection_normalizes_model_owned_flat_policy() -> None:
    parsed = OmniusClient._parse_narrative_reflection(
        json.dumps(
            {
                "unresolved_questions": [],
                "learned_context": [],
                "observation_policy": [
                    {
                        "summary": "Retrieve the linked encounter.",
                        "reason": "The model selected the unresolved relationship.",
                        "action": "retrieve",
                        "predicate": None,
                        "confidence": 0.8,
                        "entity_ids": ["person-1"],
                        "evidence_ids": [],
                    },
                    {
                        "summary": "Reduce attention to unsupported repetition.",
                        "reason": "The model found no semantic link.",
                        "action": "deprioritize",
                        "predicate": None,
                        "confidence": 0.7,
                        "entity_ids": [],
                        "evidence_ids": [],
                    },
                ],
                "constitution_update": None,
            }
        )
    )
    assert parsed is not None
    assert parsed["observation_policy"]["attend_to"][0]["action"] == "retrieve"
    assert parsed["observation_policy"]["deprioritize"][0]["action"] == "deprioritize"
    assert parsed["observation_policy"]["attend_to"][0]["context_ids"] == []
    assert OmniusClient._parse_narrative_reflection(
        '{"unresolved_questions":[],"learned_context":[],"observation_policy":'
        '{"summary":"Continue grounding.","attend_to":[],"deprioritize":[],'
        '"open_questions":[]},"constitution_update":null,"invented":true}'
    ) is None


def test_proactive_answer_is_interpreted_by_model_contract() -> None:
    async def scenario() -> None:
        client = OmniusClient(OmniusConfig(model="test", voice_model="test"))

        async def structured(prompt: str) -> str:
            assert "What is it used for?" in prompt
            return (
                '{"relation":"answer","value":"inspecting circuit boards",'
                '"reply":"Thanks, I will remember that relationship."}'
            )

        client._structured_chat = structured  # type: ignore[method-assign]
        result = await client.interpret_proactive_answer(
            "What is it used for?", "I use it for inspecting circuit boards.", "used_for"
        )
        assert result is not None
        assert result["relation"] == "answer"
        assert result["value"] == "inspecting circuit boards"

    asyncio.run(scenario())


def test_narrative_prompt_capacity_boundary_is_transparent_json() -> None:
    encoded = OmniusClient._bounded_prompt_json({"ledger": "x" * 200}, 30)
    parsed = json.loads(encoded)
    assert parsed["capacity_truncated"] is True
    assert parsed["original_characters"] > 30
    assert len(parsed["serialized_prefix"]) == 30


def test_narrative_plan_capacity_is_bounded_without_inventing_fields() -> None:
    raw = json.dumps(
        {
            "tool_requests": [
                {
                    "tool": "graph_inspect",
                    "query": None,
                    "entity_ids": [f"entity-{index}" for index in range(20)],
                    "evidence_ids": [],
                    "purpose": "p" * 500,
                }
            ],
            "planning_summary": "s" * 1500,
        }
    )
    parsed = OmniusClient._parse_narrative_plan(raw)
    assert parsed is not None
    assert len(parsed["tool_requests"][0]["entity_ids"]) == 12
    assert len(parsed["tool_requests"][0]["purpose"]) == 300
    assert len(parsed["planning_summary"]) == 1000
    assert OmniusClient._parse_narrative_plan('{"tool_requests":[],"planning_summary":"ok","extra":1}') is None


def test_narrative_synthesis_normalizes_explicit_empty_reference_slots() -> None:
    raw = json.dumps(
        {
            "version": 1,
            "narrative_summary": "A grounded day.",
            "story_update": {
                "summary": "The account gained one supported relationship.",
                "confidence": 0.8,
                "entity_ids": ["person-1"],
                "evidence_ids": ["evidence-1"],
            },
            "themes": [
                {
                    "label": "shared work",
                    "summary": "A person and tool appeared in one supported episode.",
                    "confidence": 0.7,
                    "entity_ids": ["person-1"],
                    "evidence_ids": ["evidence-1"],
                }
            ],
            "episodes": [],
            "unresolved_questions": [],
            "learned_context": [],
            "observation_policy": {
                "summary": "Observe future grounded changes.",
                "attend_to": [],
                "deprioritize": [],
                "open_questions": [],
            },
            "constitution_update": None,
        }
    )
    parsed = OmniusClient._parse_narrative_synthesis(raw)
    assert parsed is not None
    assert parsed["story_update"] == "The account gained one supported relationship."
    assert parsed["story_update_detail"]["context_ids"] == []
    assert parsed["themes"][0]["context_ids"] == []


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


def test_asr_grounding_falls_back_to_text_repetition_when_engine_omits_quality_metadata() -> None:
    # Engines such as transcribe-cli never populate no_speech_prob/avg_logprob/
    # compression_ratio on segments, which would otherwise make the checks
    # above silently unreachable and accept any non-empty hallucinated text.
    repetition_loop = {
        "text": "Allah Allah Allah Allah Allah Allah",
        "segments": [{"id": 0, "start": 0, "end": 3, "text": "Allah Allah Allah Allah Allah Allah"}],
    }
    real_sentence = {
        "text": "Can you turn on the kitchen light",
        "segments": [{"id": 0, "start": 0, "end": 2, "text": "Can you turn on the kitchen light"}],
    }

    assert OmniusClient.transcription_rejection_reason(repetition_loop) == "repetitive transcript compression"
    assert OmniusClient.transcription_rejection_reason(real_sentence) is None


def test_asr_grounding_does_not_blacklist_legitimately_spoken_phrases() -> None:
    spoken = {
        "text": "Thank you for watching!",
        "segments": [
            {
                "id": 0,
                "start": 0,
                "end": 2,
                "text": "Thank you for watching!",
                "avg_logprob": -0.1,
                "no_speech_prob": 0.01,
                "compression_ratio": 1.1,
            }
        ],
    }

    assert OmniusClient.transcription_rejection_reason(spoken) is None


def test_asr_grounding_honors_backend_acoustic_rejection() -> None:
    rejected = {
        "text": "Thank you for watching!",
        "rejection_reason": "dual Whisper disagreement on weak base decode",
    }

    assert OmniusClient.transcription_rejection_reason(rejected) == (
        "dual Whisper disagreement on weak base decode"
    )


def test_asr_grounding_rejects_sparse_text_from_fragmented_max_window() -> None:
    sparse = {
        "text": "Ah! Ah! Shit!",
        "duration": 12,
        "segments": [
            {"id": 0, "start": 0, "end": 1, "text": "Ah!"},
            {"id": 1, "start": 2, "end": 3, "text": "Ah!"},
            {"id": 2, "start": 4, "end": 5, "text": "Shit!"},
        ],
    }
    substantive = {
        "text": "Please stop and tell me why the camera moved toward the window just now",
        "duration": 12,
        "segments": [],
    }
    evidence = {"duration": 12, "boundary_reason": "max_utterance"}

    assert OmniusClient.transcription_rejection_reason(sparse, evidence) == (
        "sparse transcript over max-length acoustic window"
    )
    assert OmniusClient.transcription_rejection_reason(substantive, evidence) is None


def test_asr_grounding_rejects_live_repetition_sparse_and_language_mismatch() -> None:
    repeated = {
        "text": "I can't believe it. I can't believe it. I can't believe it.",
        "segments": [
            {"text": "I can't believe it."},
            {"text": "I can't believe it."},
            {"text": "I can't believe it."},
        ],
    }
    max_window = {"duration": 6, "boundary_reason": "max_utterance"}
    assert OmniusClient.transcription_rejection_reason(repeated, max_window) == (
        "repetitive transcript loop"
    )
    assert OmniusClient.transcription_rejection_reason(
        {"text": "Thank you.", "duration": 6}, max_window
    ) == "sparse transcript over max-length acoustic window"
    assert OmniusClient.transcription_rejection_reason(
        {"text": "JR東日本E233系電車", "duration": 6},
        {**max_window, "requested_language": "en"},
    ) == "transcript script conflicts with requested language"


def test_large_whisper_is_blocked_from_live_jetson_runtime() -> None:
    model = {
        "id": "large-v3",
        "readiness": {"weightsReady": True, "device": "cuda:0"},
    }
    assert "memory budget" in str(OmniusClient._live_asr_unavailable_reason(model))

    base = {
        "id": "base",
        "readiness": {"weightsReady": True, "device": "cuda:0"},
    }
    assert OmniusClient._live_asr_unavailable_reason(base) is None


def test_asr_rejects_digital_silence_before_calling_backend() -> None:
    wav_audio = io.BytesIO()
    with wave.open(wav_audio, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(16000)
        target.writeframes(b"\x00\x00" * 48000)
    client = OmniusClient(OmniusConfig(model="test-model", voice_model="test-voice"))

    transcript = asyncio.run(client.transcribe(wav_audio.getvalue()))

    assert transcript is None
    assert client.last_transcription_metadata["rejection_reason"] == "digital silence input"
    assert client.last_transcription_metadata["accepted"] is False


def test_asr_rejects_failed_source_acoustic_evidence_after_normalization() -> None:
    evidence = {
        "source_rms": 0.0008,
        "minimum_rms": 0.001,
        "speech_detected": True,
        "wav_rms": 0.08,
        "wav_peak": 0.2,
    }

    assert OmniusClient.acoustic_rejection_reason(evidence) == (
        "source RMS below admission threshold"
    )


def test_asr_rejects_near_floor_ambient_window_before_backend() -> None:
    evidence = {
        "source_rms": 0.061,
        "minimum_rms": 0.05,
        "speech_detected": True,
        "speech_ratio": 0.38,
        "boundary_reason": "max_utterance",
        "wav_rms": 0.12,
        "wav_peak": 0.98,
    }

    assert OmniusClient.acoustic_rejection_reason(evidence) == (
        "near-threshold max-window ambience"
    )


def test_rejected_asr_segment_metadata_does_not_retain_hallucinated_text() -> None:
    segments = [{"id": 0, "start": 0, "end": 12, "text": "Thank you for watching!"}]
    redacted = OmniusClient._segment_metadata(segments, redact_text=True)

    assert redacted == [{"id": 0, "start": 0, "end": 12}]
    assert "watching" not in str(redacted).casefold()


def test_dedicated_asr_backend_rejection_is_authoritative() -> None:
    assert OmniusClient.transcription_rejection_reason(
        {
            "text": "plausible but uncorroborated words",
            "rejection_reason": "dual Whisper disagreement on weak base decode",
        }
    ) == "dual Whisper disagreement on weak base decode"
