from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import os
import tempfile
import time
import wave
import zlib

import aiohttp
import numpy as np

from egg_companion.config import OmniusConfig
from egg_companion.cognition.dialogue import (
    InterruptionDecision,
    parse_interruption_decision,
)


class OmniusClient:
    _UNSAFE_LIVE_ASR_MODELS = {
        "large-v3": "exceeds the live Jetson memory budget and can terminate Omnius",
    }
    _KNOWN_SILENCE_HALLUCINATIONS = {
        "ご視聴ありがとうございました",
        "thank you for watching",
        "thanks for watching",
    }

    def __init__(self, config: OmniusConfig) -> None:
        self.config = config
        self._conversation: list[dict[str, str]] = []
        self._model_gate = asyncio.Lock()
        # ASR must stay responsive while conversational inference is running;
        # its own gate preserves one-at-a-time transcription ordering without
        # letting an obsolete reply hold ingress behind the chat model.
        self._asr_gate = asyncio.Lock()
        self._ocr_gate = asyncio.Lock()
        self.last_transcription_metadata: dict[str, object] = {}
        self._voice_catalog_cache: dict[str, object] | None = None
        self._voice_catalog_cached_at = 0.0

    def _headers(self) -> dict[str, str]:
        if not self.config.bearer_token_env:
            return {}
        token = os.getenv(self.config.bearer_token_env)
        if not token:
            raise RuntimeError(f"required token environment variable is unset: {self.config.bearer_token_env}")
        return {"Authorization": f"Bearer {token}"}

    def _asr_base_url(self) -> str:
        return str(self.config.asr_base_url or self.config.base_url).rstrip("/")

    async def health(self) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(f"{str(self.config.base_url).rstrip('/')}/health", headers=self._headers()) as response:
                response.raise_for_status()

    async def cognition_health(self) -> dict[str, object]:
        """Read backend readiness without consuming a full model generation.

        Omnius 1.0.629 advertises ``/health/ready`` specifically as its backend
        reachability probe.  Periodically generating a synthetic chat turn made
        the monitor contend with live vision and could hold every health result
        for the complete 120-second inference timeout.
        """
        timeout = aiohttp.ClientTimeout(total=5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{str(self.config.base_url).rstrip('/')}/health/ready",
                headers=self._headers(),
            ) as response:
                if response.status == 404:
                    return {"supported": False, "status": "unknown"}
                if response.status not in {200, 503}:
                    detail = (await response.text())[:500]
                    raise RuntimeError(
                        f"Omnius cognition health HTTP {response.status}: {detail}"
                    )
                payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Omnius cognition health is not an object")
        return {**payload, "supported": True}

    async def audio_classifier_health(self) -> dict[str, object]:
        """Return the dedicated persistent classifier state added in Omnius 1.0.628.

        A 503 response is a valid readiness payload, not a transport failure. A
        404 identifies an older Omnius release so callers can retain the legacy
        direct-tool fallback without presenting it as a warm classifier.
        """
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{str(self.config.base_url).rstrip('/')}/v1/audio/classify/health",
                headers=self._headers(),
            ) as response:
                if response.status == 404:
                    return {"supported": False, "ready": None}
                if response.status not in {200, 503}:
                    detail = (await response.text())[:500]
                    raise RuntimeError(
                        f"Omnius audio classifier health HTTP {response.status}: {detail}"
                    )
                payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Omnius audio classifier health is not an object")
        return {**payload, "supported": True}

    async def chat_contract_probe(self) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "Return exactly READY."},
                {"role": "user", "content": "Verify the local chat contract."},
            ],
            "stream": False,
            "temperature": 0,
            "max_tokens": 8,
            "tools": False,
        }
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/chat",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius chat contract HTTP {response.status}: {detail}")
                    result = await response.json()
        try:
            content = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius chat contract returned an invalid completion") from error
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("Omnius chat contract returned an empty completion")
        return content.strip()

    async def voice_state(self) -> dict[str, object]:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            async with session.get(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/state", headers=self._headers()
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Omnius voice state is not an object")
        return payload

    async def voice_catalog(self, *, force: bool = False) -> dict[str, object]:
        # Model discovery may stat several local model trees on first access.
        # Keep it outside the low-latency health budget and fetch independent
        # catalog surfaces concurrently so the dashboard remains responsive.
        state = await self.voice_state()
        if (
            not force
            and self._voice_catalog_cache is not None
            and time.monotonic() - self._voice_catalog_cached_at < 300
        ):
            return {**self._voice_catalog_cache, "state": state}

        timeout = aiohttp.ClientTimeout(total=90)
        base_url = str(self.config.base_url).rstrip("/")
        asr_base_url = self._asr_base_url()

        async def get_json(
            session: aiohttp.ClientSession,
            path: str,
            *,
            service_url: str = base_url,
            unavailable: dict[str, object] | None = None,
        ) -> dict[str, object]:
            async with session.get(f"{service_url}{path}", headers=self._headers()) as response:
                if response.status == 404 and unavailable is not None:
                    return unavailable
                response.raise_for_status()
                payload = await response.json()
            if not isinstance(payload, dict):
                raise RuntimeError(f"Omnius returned an invalid voice payload for {path}")
            return payload

        async with aiohttp.ClientSession(timeout=timeout) as session:
            tts, asr, supertonic, dedicated_asr_state = await asyncio.gather(
                get_json(session, "/v1/voice/models"),
                get_json(
                    session, "/v1/voice/asr-models", service_url=asr_base_url
                ),
                get_json(
                    session,
                    "/v1/voice/supertonic-settings",
                    unavailable={"supported": False, "settings": {}, "options": {"voices": []}},
                ),
                get_json(
                    session,
                    "/v1/voice/state",
                    service_url=asr_base_url,
                    unavailable={},
                ) if asr_base_url != base_url else asyncio.sleep(0, result={}),
            )
        if dedicated_asr_state:
            state = {
                **state,
                **{
                    key: value for key, value in dedicated_asr_state.items()
                    if key.startswith("asr") or key in {"device", "lastError", "loadedAt"}
                },
            }
        models = asr.get("models")
        if isinstance(models, list):
            asr = {
                **asr,
                "models": [
                    {
                        **model,
                        "liveEligible": self._live_asr_unavailable_reason(model) is None,
                        "liveReason": self._live_asr_unavailable_reason(model),
                    }
                    if isinstance(model, dict)
                    else model
                    for model in models
                ],
            }
        self._voice_catalog_cache = {"tts": tts, "asr": asr, "supertonic": supertonic}
        self._voice_catalog_cached_at = time.monotonic()
        return {**self._voice_catalog_cache, "state": state}

    async def asr_catalog(self) -> dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=15)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                f"{self._asr_base_url()}/v1/voice/asr-models",
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Omnius ASR catalog is not an object")
        return payload

    async def ensure_asr_model(self, model_id: str) -> None:
        asr = await self.asr_catalog()
        models = asr.get("models")
        selected = next(
            (
                model
                for model in models
                if isinstance(model, dict) and model.get("id") == model_id
            ),
            None,
        ) if isinstance(models, list) else None
        reason = self._live_asr_unavailable_reason(selected)
        if reason:
            raise RuntimeError(f"ASR model {model_id} is unavailable for live use: {reason}")
        if asr.get("current") == model_id:
            return
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{self._asr_base_url()}/v1/voice/asr-models/switch",
                json={"modelId": model_id},
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Omnius ASR model switch HTTP {response.status}: {detail}")

    @classmethod
    def _live_asr_unavailable_reason(cls, model: object) -> str | None:
        if not isinstance(model, dict):
            return "model is not present in the Omnius catalog"
        model_id = str(model.get("id") or "")
        if model_id in cls._UNSAFE_LIVE_ASR_MODELS:
            return cls._UNSAFE_LIVE_ASR_MODELS[model_id]
        readiness = model.get("readiness")
        if not isinstance(readiness, dict) or readiness.get("weightsReady") is not True:
            if isinstance(readiness, dict) and readiness.get("lastError"):
                return str(readiness["lastError"])
            return "model weights are not ready"
        return None

    async def configure_supertonic_voice(self, voice_name: str | None) -> bool:
        if self.config.voice_model != "supertonic" or not voice_name:
            return False
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.patch(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/supertonic-settings",
                json={"voiceName": voice_name},
                headers=self._headers(),
            ) as response:
                if response.status in {404, 405, 501}:
                    return False
                response.raise_for_status()
        return True

    async def ensure_voice_ready(self, model_id: str | None = None) -> None:
        model_id = model_id or self.config.voice_model
        state = await self.voice_state()
        if state.get("voiceReady") is True and state.get("voiceModelId") == model_id:
            return
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/models/switch",
                json={"modelId": model_id, "enable": True},
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Omnius voice model switch HTTP {response.status}: {detail}")
        state = await self.voice_state()
        if state.get("voiceReady") is not True:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/voice/start",
                    json={"modelId": model_id},
                    headers=self._headers(),
                ) as response:
                    if response.status not in {200, 404}:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius voice start HTTP {response.status}: {detail}")
            state = await self.voice_state()
        if state.get("voiceReady") is not True:
            raise RuntimeError(f"Omnius voice did not become ready: {state}")

    async def companion_reply(self, scene: str) -> str:
        return await self._chat(
            "Offer one concise, useful spoken observation or next step. "
            f"Observed scene: {scene}"
        )

    async def conversation_reply(
        self,
        utterance: str,
        scene: str,
        history: list[dict[str, object]] | None = None,
    ) -> str:
        ordered_history = json.dumps(history or [], ensure_ascii=False)
        return await self._chat(
            "Local speech, already verified as human speech by VAD:\n"
            f"{utterance!r}\n\nEmbodied context:\n{scene}\n\n"
            f"Complete ordered audible conversation ledger:\n{ordered_history}\n\n"
            "The ledger distinguishes completed and interrupted agent utterances. "
            "Never answer in the human's voice, answer Egg's own prior question, or treat an "
            "interrupted agent utterance as fully heard. "
            "When the embodied context supplies a user-provided preferred name, use it naturally "
            "when useful, without repeating it in every reply. When WEB SEARCH TOOL EVIDENCE is "
            "present, ground current facts in that evidence and do not read URLs aloud. "
            "Decide from the conversational history and context whether this is directed to Egg. "
            "If it is not directed to Egg or does not merit an audible interruption, "
            "reply exactly [[SILENT]].",
            remember=False,
            include_memory=False,
        )

    async def classify_interruption(
        self,
        heard_text: str,
        agent_text: str,
        history: list[dict[str, object]],
        context: str,
    ) -> InterruptionDecision | None:
        raw = await self._structured_chat(
            "You are Egg's silent secondary interruption-triage analyst. Never answer the speaker. "
            "Decide whether new human audio overlapping Egg's playback is a genuine interruption. "
            "Use the complete ordered conversation, current agent playback, latest heard transcript, and "
            "embodied context. Treat timing as transport metadata only, never as semantic proof. "
            "A correction, stop/redirect command, answer to Egg, substantive question, or other change "
            "to what Egg must do next is genuine. Speaker echo, a backchannel, room noise, duplicated "
            "agent speech, or ambiguous audio is not.\n"
            f"Ordered conversation: {json.dumps(history, ensure_ascii=False)}\n"
            f"Current agent playback: {agent_text!r}\n"
            f"Latest heard-audio candidate: {heard_text!r}\n"
            f"Embodied context: {context[:1600]}\n"
            "Return exactly this JSON shape with no extra keys: "
            '{"version":1,"genuine":boolean,"confidence":number,"reason":string,'
            '"summary":string,"should_cancel_playback":boolean}.'
        )
        return parse_interruption_decision(raw)

    async def reason_about_utterance(
        self, utterance: str, context: str
    ) -> dict[str, object] | None:
        raw = await self._structured_chat(
            "Classify a VAD-verified utterance for interaction routing, without answering it.\n"
            f"Utterance: {utterance!r}\nContext: {context[:1200]}\n"
            "Return only JSON: {\"directed\":boolean,\"act\":\"question\"|\"correction\"|"
            "\"person_naming\"|\"object_naming\"|\"command\"|\"acknowledgement\"|"
            "\"conversation\",\"confidence\":number,\"tool\":\"none\"|\"web_search\","
            "\"tool_query\":string|null}. Select web_search for an explicit request to search or "
            "for current external information that cannot be grounded in the embodied context. "
            "Make tool_query a concise standalone search query."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(parsed.get("directed"), bool)
            or parsed.get("act") not in {
                "question", "correction", "person_naming", "object_naming", "command",
                "acknowledgement", "conversation",
            }
            or not isinstance(parsed.get("confidence"), (int, float))
            or not 0 <= float(parsed["confidence"]) <= 1
        ):
            return None
        tool = parsed.get("tool", "none")
        tool_query = parsed.get("tool_query")
        if tool not in {"none", "web_search"}:
            return None
        if tool == "web_search":
            if not isinstance(tool_query, str) or not tool_query.strip():
                return None
            tool_query = " ".join(tool_query.split())[:300]
        else:
            tool_query = None
        return {
            "directed": parsed["directed"],
            "act": parsed["act"],
            "confidence": float(parsed["confidence"]),
            "tool": tool,
            "tool_query": tool_query,
        }

    async def interpret_correction(
        self, utterance: str, candidate: dict[str, object]
    ) -> dict[str, str] | None:
        return await self.interpret_observation_feedback(utterance, candidate)

    async def interpret_person_naming(
        self, utterance: str, *, prompted: bool = False
    ) -> str | None:
        return await self.interpret_person_introduction(utterance, prompted=prompted)

    async def interpret_object_naming(self, utterance: str) -> str | None:
        return await self.interpret_held_object_label(utterance)

    async def observation_question(self, candidate: dict[str, object], scene: str) -> str:
        return await self._chat(
            "A stable but uncertain visual observation needs human calibration. "
            f"Candidate: {json.dumps(candidate)}\nContext: {scene}\n"
            "Ask one natural, specific question about whether that label is accurate."
        )

    async def interpret_observation_feedback(self, utterance: str, candidate: dict[str, object]) -> dict[str, str] | None:
        raw = await self._structured_chat(
            "Interpret this response to a pending visual calibration question.\n"
            f"Candidate: {json.dumps(candidate)}\nSpeaker response: {utterance!r}\n"
            "Return only JSON: {\"decision\":\"confirm\"|\"correct\"|\"unclear\",\"label\":string|null,\"reply\":string}. "
            "Use correct only when the speaker clearly supplies a replacement object label."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        decision = parsed.get("decision")
        label = parsed.get("label")
        reply = parsed.get("reply")
        if decision not in {"confirm", "correct", "unclear"} or not isinstance(reply, str) or not reply.strip():
            return None
        if decision == "correct" and (not isinstance(label, str) or not label.strip() or len(label) > 64):
            return None
        return {"decision": decision, "label": label.strip() if isinstance(label, str) else "", "reply": reply.strip()}

    async def interpret_held_object_label(self, utterance: str) -> str | None:
        raw = await self._structured_chat(
            "Determine whether the speaker explicitly gives a name to a physical object they are holding up to Egg. "
            f"Utterance: {utterance!r}\n"
            "Return only JSON: {\"label\": string|null}. "
            "Use a label only for an explicit object-identification statement; never treat a person's name, a request, "
            "or unrelated conversation as an object label."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        label = parsed.get("label")
        if not isinstance(label, str):
            return None
        normalized = " ".join(label.strip().split())
        return normalized if normalized and len(normalized) <= 64 else None

    async def interpret_person_introduction(
        self, utterance: str, *, prompted: bool = False
    ) -> str | None:
        raw = await self._structured_chat(
            "Determine whether the speaker provides their own preferred name. "
            f"Utterance: {utterance!r}\n"
            f"This {'is' if prompted else 'is not'} a direct response to Egg asking what to call them. "
            "Return only JSON: {\"name\": string|null}. A bare plausible personal name is valid only "
            "when this is a prompted response. Do not infer names, extract names of third parties, or "
            "treat descriptions such as 'I am tired' as a name."
        )
        return self.parse_person_name(raw)

    async def web_search(self, query: str, *, num_results: int = 5) -> str:
        """Invoke Omnius' policy-gated local web-search tool directly.

        The direct tool endpoint keeps voice turns responsive while retaining
        Omnius' auth, egress, and audit policy. Its output is evidence for a
        subsequent grounded conversational completion, never executable text.
        """
        normalized = " ".join(query.strip().split())
        if not normalized:
            raise ValueError("web search query is required")
        timeout = aiohttp.ClientTimeout(total=min(self.config.timeout_seconds, 20))
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/tools/web_search/call",
                json={
                    "args": {
                        "query": normalized[:300],
                        "num_results": max(1, min(int(num_results), 8)),
                        "provider": "duckduckgo",
                    }
                },
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(
                        f"Omnius web_search HTTP {response.status}: {detail}"
                    )
                result = await response.json()
        # Omnius' direct tool API wraps the canonical ToolResult with security
        # metadata. Retain compatibility with an unwrapped ToolResult as well.
        tool_result = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(tool_result, dict) or tool_result.get("success") is not True:
            detail = (
                tool_result.get("error", "invalid tool result")
                if isinstance(tool_result, dict)
                else "invalid tool result"
            )
            raise RuntimeError(f"Omnius web_search failed: {detail}")
        output = tool_result.get("output")
        if not isinstance(output, str) or not output.strip():
            raise RuntimeError("Omnius web_search returned no evidence")
        return output.strip()[:10000]

    async def analyze_audio_scene(
        self, wav_audio: bytes, *, top_k: int = 5
    ) -> dict[str, object]:
        """Run Omnius' grounded YAMNet/AudioSet classifier off the ASR path.

        Omnius can expose a broader ``comprehend`` action, but that action may
        legitimately report ``mock semantic scaffold`` when no live AV sidecar
        roles are loaded. Egg never promotes scaffold events into memory. This
        method intentionally consumes only the independent YAMNet classifier's
        numeric output and locally measured WAV facts.
        """
        acoustic = self._wav_acoustic_evidence(wav_audio)
        if not acoustic or float(acoustic.get("duration") or 0) <= 0:
            raise ValueError("audio comprehension requires a valid WAV payload")
        temporary_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                prefix="egg-audio-comprehension-", suffix=".wav", delete=False
            ) as temporary:
                temporary.write(wav_audio)
                temporary_path = temporary.name
            tool_result = await self._call_audio_classifier(
                {
                    "action": "classify",
                    "file": temporary_path,
                    "top_k": max(1, min(int(top_k), 20)),
                },
                timeout_seconds=max(30.0, self.config.timeout_seconds),
            )
        finally:
            if temporary_path:
                try:
                    os.unlink(temporary_path)
                except FileNotFoundError:
                    pass
        data = tool_result.get("data")
        parsed = self._normalize_audio_classification(data)
        if parsed is None:
            output = tool_result.get("output")
            parsed = (
                self._parse_audio_classification(output)
                if isinstance(output, str)
                else None
            )
        if parsed is None:
            raise RuntimeError("Omnius audio_analyze returned invalid YAMNet output")
        return {
            "classifications": parsed["classifications"],
            "total_classes": parsed.get("total_classes", 521),
            "duration_seconds": parsed.get("duration_s", acoustic.get("duration")),
            "acoustic": acoustic,
            "model": parsed.get("model", "google/yamnet/1"),
            "backend": parsed.get("backend"),
            "taxonomy": parsed.get("taxonomy", "AudioSet"),
            "semantic_quality": "grounded classifier",
            "mock_evidence_discarded": True,
        }

    async def _call_audio_classifier(
        self, args: dict[str, object], *, timeout_seconds: float
    ) -> dict[str, object]:
        """Prefer Omnius' warm structured audio endpoint with legacy fallback."""
        server_timeout_ms = max(
            1000, min(120000, int(round(timeout_seconds * 1000)))
        )
        timeout = aiohttp.ClientTimeout(total=timeout_seconds + 5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/audio/classify",
                json={"args": args, "timeout_ms": server_timeout_ms},
                headers=self._headers(),
            ) as response:
                if response.status == 404:
                    return await self._call_tool(
                        "audio_analyze", args, timeout_seconds=timeout_seconds
                    )
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(
                        f"Omnius audio classifier HTTP {response.status}: {detail}"
                    )
                result = await response.json()
        tool_result = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(tool_result, dict) or tool_result.get("success") is not True:
            detail = (
                tool_result.get("error", "invalid classifier result")
                if isinstance(tool_result, dict)
                else "invalid classifier result"
            )
            raise RuntimeError(f"Omnius audio classifier failed: {detail}")
        return tool_result

    async def _call_tool(
        self,
        name: str,
        args: dict[str, object],
        *,
        timeout_seconds: float,
    ) -> dict[str, object]:
        server_timeout_ms = max(
            1000, min(120000, int(round(timeout_seconds * 1000)))
        )
        # Omnius' direct tool executor defaults to 30 seconds independently of
        # the HTTP client timeout. Send its explicit bounded timeout so a warmup
        # or cold YAMNet pass is not killed while Egg is still waiting.
        timeout = aiohttp.ClientTimeout(total=timeout_seconds + 5)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/tools/{name}/call",
                json={"args": args, "timeout_ms": server_timeout_ms},
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(
                        f"Omnius {name} HTTP {response.status}: {detail}"
                    )
                result = await response.json()
        tool_result = result.get("result", result) if isinstance(result, dict) else result
        if not isinstance(tool_result, dict) or tool_result.get("success") is not True:
            detail = (
                tool_result.get("error", "invalid tool result")
                if isinstance(tool_result, dict)
                else "invalid tool result"
            )
            raise RuntimeError(f"Omnius {name} failed: {detail}")
        return tool_result

    @staticmethod
    def _parse_audio_classification(output: str) -> dict[str, object] | None:
        # runPythonScript prefixes its JSON with ``Audio scene classification:``.
        start = output.find("{")
        if start < 0:
            return None
        try:
            payload = json.loads(output[start:])
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict) or payload.get("success") is not True:
            return None
        return OmniusClient._normalize_audio_classification(payload)

    @staticmethod
    def _normalize_audio_classification(
        payload: object,
    ) -> dict[str, object] | None:
        if not isinstance(payload, dict):
            return None
        classifications = payload.get("classifications")
        if not isinstance(classifications, list):
            return None
        normalized: list[dict[str, object]] = []
        for item in classifications:
            if not isinstance(item, dict):
                continue
            raw_label = item.get("label", item.get("class"))
            raw_score = item.get("confidence", item.get("score"))
            if not isinstance(raw_label, str):
                continue
            try:
                score = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                continue
            label = " ".join(raw_label.split())[:120]
            if label:
                normalized.append({"label": label, "confidence": score})
        result: dict[str, object] = {
            "classifications": normalized,
            "total_classes": int(payload.get("total_classes") or 521),
            "duration_s": float(
                payload.get("duration_s", payload.get("duration_seconds")) or 0
            ),
        }
        for key in ("model", "backend", "taxonomy"):
            if isinstance(payload.get(key), str):
                result[key] = payload[key]
        return result

    @staticmethod
    def parse_person_name(content: object) -> str | None:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        name = parsed.get("name")
        if not isinstance(name, str):
            return None
        normalized = " ".join(name.strip().split())
        return normalized if normalized and len(normalized) <= 64 else None

    async def transcribe(
        self,
        wav_audio: bytes,
        *,
        acoustic_evidence: dict[str, object] | None = None,
        language: str = "auto",
    ) -> str | None:
        evidence = {
            **self._wav_acoustic_evidence(wav_audio),
            **dict(acoustic_evidence or {}),
            "requested_language": language,
        }
        acoustic_rejection = self.acoustic_rejection_reason(evidence)
        if acoustic_rejection is not None:
            self.last_transcription_metadata = {
                "duration": evidence.get("duration"),
                "language": None,
                "segments": [],
                "acoustic": evidence,
                "accepted": False,
                "rejection_reason": acoustic_rejection,
            }
            return None
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = {**self._headers(), "Content-Type": "audio/wav"}
        async with self._asr_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{self._asr_base_url()}/v1/voice/transcribe",
                    data=wav_audio,
                    headers=headers,
                    params={"language": language},
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius ASR HTTP {response.status}: {detail}")
                    payload = await response.json()
        text = payload.get("text")
        rejection_reason = self.transcription_rejection_reason(payload, evidence)
        if not isinstance(text, str) or not text.strip():
            rejection_reason = "empty transcript"
        segment_metadata = self._segment_metadata(
            payload.get("segments"), redact_text=rejection_reason is not None
        )
        self.last_transcription_metadata = {
            "duration": payload.get("duration"),
            "language": payload.get("language"),
            "segments": segment_metadata,
            "acoustic": evidence,
            "accepted": rejection_reason is None,
            "rejection_reason": rejection_reason,
        }
        if rejection_reason is not None:
            return None
        return text.strip()

    @staticmethod
    def transcription_is_grounded(
        payload: dict[str, object], acoustic_evidence: dict[str, object] | None = None
    ) -> bool:
        return OmniusClient.transcription_rejection_reason(payload, acoustic_evidence) is None

    @staticmethod
    def transcription_rejection_reason(
        payload: dict[str, object], acoustic_evidence: dict[str, object] | None = None
    ) -> str | None:
        backend_rejection = payload.get("rejection_reason")
        if isinstance(backend_rejection, str) and backend_rejection.strip():
            return backend_rejection.strip()
        text = payload.get("text")
        if isinstance(text, str) and OmniusClient._is_known_silence_hallucination(text):
            # Some backends return text with no segment array. This guard must
            # precede segment-quality handling or their most common silence
            # hallucination bypasses grounding entirely.
            return "known silence hallucination"
        segments = payload.get("segments")
        scored = (
            [segment for segment in segments if isinstance(segment, dict)]
            if isinstance(segments, list)
            else []
        )
        no_speech = [
            float(segment["no_speech_prob"])
            for segment in scored
            if isinstance(segment.get("no_speech_prob"), (int, float))
        ]
        log_prob = [
            float(segment["avg_logprob"])
            for segment in scored
            if isinstance(segment.get("avg_logprob"), (int, float))
        ]
        compression = [
            float(segment["compression_ratio"])
            for segment in scored
            if isinstance(segment.get("compression_ratio"), (int, float))
        ]
        if no_speech and sum(no_speech) / len(no_speech) >= 0.55:
            return "high no-speech probability"
        if log_prob and sum(log_prob) / len(log_prob) <= -1.0:
            return "low average token probability"
        if compression and max(compression) >= 2.4:
            return "repetitive transcript compression"
        if not compression:
            # Some ASR engines (e.g. transcribe-cli) never populate segment-level
            # quality metadata, silently making the checks above unreachable, so
            # every non-empty transcript would otherwise be accepted regardless
            # of quality. Recompute the same repetition signal directly from the
            # transcript text, using Whisper's own algorithm and threshold, to
            # still catch its most common hallucination class: word/phrase
            # repetition loops (e.g. "Allah Allah Allah Allah Allah Allah").
            if isinstance(text, str) and OmniusClient._text_compression_ratio(text) >= 2.4:
                return "repetitive transcript compression"
        if isinstance(text, str) and OmniusClient._is_repeated_transcript(text, scored):
            return "repetitive transcript loop"
        evidence = acoustic_evidence or {}
        duration = evidence.get("duration", payload.get("duration"))
        if (
            isinstance(text, str)
            and evidence.get("boundary_reason") == "max_utterance"
            and isinstance(duration, (int, float))
            and float(duration) >= 4
        ):
            symbol_count = sum(
                character.isalnum() for character in OmniusClient._normalized_transcript(text)
            )
            if symbol_count / float(duration) < 1.75:
                return "sparse transcript over max-length acoustic window"
        requested_language = evidence.get("requested_language")
        if (
            isinstance(text, str)
            and requested_language == "en"
            and OmniusClient._non_latin_letter_ratio(text) > 0.40
        ):
            return "transcript script conflicts with requested language"
        return None

    @staticmethod
    def acoustic_rejection_reason(evidence: dict[str, object]) -> str | None:
        speech_detected = evidence.get("speech_detected")
        if speech_detected is False:
            return "source VAD rejected speech"
        source_rms = evidence.get("source_rms", evidence.get("wav_rms"))
        minimum_rms = evidence.get("minimum_rms")
        if (
            isinstance(source_rms, (int, float))
            and isinstance(minimum_rms, (int, float))
            and math.isfinite(float(source_rms))
            and float(source_rms) < float(minimum_rms)
        ):
            return "source RMS below admission threshold"
        if (
            evidence.get("boundary_reason") == "max_utterance"
            and isinstance(source_rms, (int, float))
            and isinstance(minimum_rms, (int, float))
            and math.isfinite(float(source_rms))
            and math.isfinite(float(minimum_rms))
            and float(source_rms) < float(minimum_rms) * 1.5
        ):
            # A room-noise spike can scrape over the absolute RMS floor while
            # WebRTC VAD accumulates a fragmented utterance until the hard cap.
            # Do not spend a Whisper invocation on that near-floor window.
            return "near-threshold max-window ambience"
        wav_rms = evidence.get("wav_rms")
        wav_peak = evidence.get("wav_peak")
        if (
            isinstance(wav_rms, (int, float))
            and isinstance(wav_peak, (int, float))
            and float(wav_rms) <= 0.0001
            and float(wav_peak) <= 0.0005
        ):
            return "digital silence input"
        return None

    @staticmethod
    def _wav_acoustic_evidence(wav_audio: bytes) -> dict[str, object]:
        try:
            with wave.open(io.BytesIO(wav_audio), "rb") as source:
                channels = source.getnchannels()
                sample_width = source.getsampwidth()
                sample_rate = source.getframerate()
                frame_count = source.getnframes()
                frames = source.readframes(frame_count)
        except (EOFError, OSError, wave.Error):
            return {}
        if channels <= 0 or sample_width != 2 or sample_rate <= 0 or not frames:
            return {"duration": frame_count / sample_rate if sample_rate else None}
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768
        if channels > 1:
            complete = samples.size - samples.size % channels
            samples = samples[:complete].reshape(-1, channels).mean(axis=1)
        if not samples.size:
            return {"duration": frame_count / sample_rate}
        return {
            "duration": frame_count / sample_rate,
            "wav_rms": float(np.sqrt(np.mean(np.square(samples)))),
            "wav_peak": float(np.max(np.abs(samples))),
        }

    @staticmethod
    def _normalized_transcript(text: str) -> str:
        normalized = "".join(
            character if character.isalnum() or character.isspace() else " "
            for character in text.casefold()
        )
        return " ".join(normalized.split())

    @staticmethod
    def _is_known_silence_hallucination(text: str) -> bool:
        normalized = OmniusClient._normalized_transcript(text)
        words = normalized.split()
        if len(words) <= 10 and (
            "thanks for watching" in normalized
            or "thank you for watching" in normalized
            or "thank you so much for watching" in normalized
        ):
            return True
        for phrase in OmniusClient._KNOWN_SILENCE_HALLUCINATIONS:
            phrase_words = phrase.split()
            if words and len(words) % len(phrase_words) == 0 and words == phrase_words * (
                len(words) // len(phrase_words)
            ):
                return True
        return False

    @staticmethod
    def _segment_metadata(segments: object, *, redact_text: bool) -> list[dict[str, object]]:
        if not isinstance(segments, list):
            return []
        return [
            {
                key: value
                for key, value in segment.items()
                if not redact_text or key != "text"
            }
            for segment in segments
            if isinstance(segment, dict)
        ]

    @staticmethod
    def _text_compression_ratio(text: str) -> float:
        data = text.strip().encode("utf-8")
        if not data:
            return 0.0
        compressor = zlib.compressobj(level=9, wbits=-15)
        compressed = compressor.compress(data) + compressor.flush()
        return len(data) / max(len(compressed), 1)

    @staticmethod
    def _is_repeated_transcript(text: str, segments: list[dict[str, object]]) -> bool:
        segment_text = [
            OmniusClient._normalized_transcript(str(segment.get("text") or ""))
            for segment in segments
            if str(segment.get("text") or "").strip()
        ]
        if len(segment_text) >= 2 and len(set(segment_text)) == 1:
            return True
        words = OmniusClient._normalized_transcript(text).split()
        for width in range(1, len(words) // 2 + 1):
            if len(words) % width:
                continue
            repetitions = len(words) // width
            if repetitions >= 3 and words == words[:width] * repetitions:
                return True
        return False

    @staticmethod
    def _non_latin_letter_ratio(text: str) -> float:
        letters = [character for character in text if character.isalpha()]
        if not letters:
            return 0.0
        non_latin = sum(
            not (
                "A" <= character <= "Z"
                or "a" <= character <= "z"
                or "À" <= character <= "ɏ"
            )
            for character in letters
        )
        return non_latin / len(letters)

    async def ocr_advanced(self, image_path: str) -> dict[str, object] | None:
        """Read a local image with Omnius' managed multi-variant OCR pipeline.

        Omnius 1.0.629 replaced the old ``{imagePath}`` controller with the
        canonical direct-tool envelope and structured ``result.data``. Keep the
        former response parser as a compatibility path for older daemons.
        """
        server_timeout_ms = max(
            30000, min(120000, int(round(self.config.timeout_seconds * 1000)))
        )
        timeout = aiohttp.ClientTimeout(total=server_timeout_ms / 1000 + 5)
        async with self._ocr_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/ocr/advanced",
                    json={
                        "args": {
                            "image": image_path,
                            "language": "eng",
                            "regions": True,
                        },
                        "timeout_ms": server_timeout_ms,
                        "max_output_chars": 40000,
                    },
                    headers=self._headers(),
                ) as response:
                    if response.status == 501:
                        return None
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius OCR HTTP {response.status}: {detail}")
                    result = await response.json()
        if not isinstance(result, dict):
            return None
        tool_result = result.get("result")
        if isinstance(tool_result, dict):
            if tool_result.get("success") is not True:
                raise RuntimeError(
                    f"Omnius advanced OCR failed: "
                    f"{tool_result.get('error') or 'invalid tool result'}"
                )
            data = tool_result.get("data")
            if not isinstance(data, dict):
                return None
            text = data.get("text")
            if not isinstance(text, str) or not text.strip():
                return None
            raw_confidence = data.get("confidence")
            try:
                confidence = float(raw_confidence)
                if confidence > 1:
                    confidence /= 100
                confidence = max(0.0, min(1.0, confidence))
            except (TypeError, ValueError):
                confidence = None
            regions = data.get("regions")
            normalized_regions: list[dict[str, object]] = []
            if isinstance(regions, dict):
                normalized_regions = [
                    {"name": str(name), "text": value.strip()}
                    for name, value in regions.items()
                    if isinstance(value, str) and value.strip()
                ]
            elif isinstance(regions, list):
                normalized_regions = [
                    dict(region) for region in regions if isinstance(region, dict)
                ]
            return {
                "text": text.strip()[:2000],
                "vision_used": False,
                "engine": "omnius-ocr-image-advanced",
                "confidence": confidence,
                "regions": normalized_regions[:32],
                "variant": data.get("variant"),
                "variants_tested": data.get("variants_tested"),
            }

        # Omnius <=1.0.628 compatibility response.
        if result.get("success") is not True:
            return None
        text = result.get("ocrText")
        if not isinstance(text, str) or not text.strip():
            return None
        return {
            "text": text.strip()[:500],
            "vision_used": bool(result.get("visionUsed")),
            "engine": "omnius-legacy-advanced-ocr",
        }

    async def classify_masked_object_analysis(
        self,
        image_png: bytes,
        detector_label: str,
        detector_confidence: float,
    ) -> dict[str, object] | None:
        """Classify a mask and report grounded visual evidence of visible text."""
        image_data = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Classify only the opaque segmented object in this PNG; ignore transparent pixels and background. "
                        "Identify the specific physical item with a short ordinary noun phrase. Correct the detector "
                        "when its category is vague, stylistic, or unsupported by the pixels. Do not return a scene, "
                        "genre, material, person occupation, or visual style as the object label. "
                        "Independently inspect the opaque pixels for actually visible printed, written, or displayed "
                        "characters. Do not infer text merely because this kind of object commonly has a label, and "
                        "do not guess or transcribe unreadable characters. Return JSON only: "
                        "{\"label\":string|null,\"confidence\":number,\"visible_text\":boolean,"
                        "\"text_regions\":[string]}. text_regions is at most four short location descriptions "
                        "such as 'upper-left display' or 'shirt front'; it is empty when no text is visibly grounded. "
                        f"Detector candidate: {detector_label!r} at {detector_confidence:.2f}."
                    ),
                    "images": [image_data],
                },
            ],
            "stream": False,
            "format": "json",
            # This bounded classification does not need a hidden chain of thought.
            # Ornith otherwise spends the entire generation budget reasoning and
            # can finish with an empty content field before it emits the JSON.
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 64},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.vision_base_url).rstrip('/')}/api/chat", json=payload
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Ornith VLM HTTP {response.status}: {detail}")
                    result = await response.json()
        try:
            content = result["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius VLM returned an invalid completion") from error
        return self.parse_object_analysis(content)

    async def classify_masked_object(
        self, image_png: bytes, detector_label: str, detector_confidence: float
    ) -> tuple[str, float] | None:
        """Compatibility classification view over the richer visual analysis."""
        analysis = await self.classify_masked_object_analysis(
            image_png, detector_label, detector_confidence
        )
        if analysis is None:
            return None
        return str(analysis["label"]), float(analysis["confidence"])

    async def compare_temporal_person_detections(
        self,
        prior_png: bytes,
        current_png: bytes,
        geometry: dict[str, object],
    ) -> dict[str, object]:
        """Audit adjacent same-camera person-mask continuity with local Ornith.

        Geometry owns short-term tracking; this comparison produces inspectable
        visual and displacement reasoning. It is not face recognition and must
        not infer identity, demographics, or other sensitive attributes.
        """
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "These are two transparent-background instance-mask crops from "
                        "the same camera a few frames apart. Determine whether they show "
                        "one continuous physical person detection. Compare only visible, "
                        "non-sensitive continuity cues such as silhouette, clothing colors, "
                        "carried items, pose transition, and the supplied mask geometry. "
                        "Do not name the person or infer demographics, emotion, health, or "
                        "other sensitive traits. Keep analysis at most 45 words and "
                        "displacement_analysis at most 30 words. Return exactly these four "
                        "JSON fields and no prose: "
                        '{"same_person":boolean,"confidence":number,'
                        '"analysis":string,"displacement_analysis":string}. '
                        "Image 1 is prior; image 2 is current. Geometry: "
                        f"{json.dumps(geometry, sort_keys=True)[:1800]}"
                    ),
                    "images": [
                        base64.b64encode(prior_png).decode("ascii"),
                        base64.b64encode(current_png).decode("ascii"),
                    ],
                }
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 260},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.vision_base_url).rstrip('/')}/api/chat",
                    json=payload,
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(
                            f"Ornith temporal-person comparison HTTP {response.status}: {detail}"
                        )
                    result = await response.json()
        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                "Ornith temporal-person comparison returned an invalid completion"
            ) from error
        parsed = self.parse_temporal_person_comparison(content)
        if parsed is None:
            raise RuntimeError(
                "Ornith temporal-person comparison returned invalid analysis JSON"
            )
        return parsed

    @staticmethod
    def parse_temporal_person_comparison(
        content: object,
    ) -> dict[str, object] | None:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        same_person = parsed.get("same_person")
        confidence = parsed.get("confidence")
        analysis = parsed.get("analysis")
        displacement = parsed.get("displacement_analysis")
        if (
            not isinstance(same_person, bool)
            or not isinstance(confidence, (int, float))
            or not isinstance(analysis, str)
            or not isinstance(displacement, str)
            or not 0 <= float(confidence) <= 1
        ):
            return None
        normalized_analysis = " ".join(analysis.split())[:600]
        normalized_displacement = " ".join(displacement.split())[:400]
        if not normalized_analysis or not normalized_displacement:
            return None
        correspondences = parsed.get("visible_correspondences")
        return {
            "same_person": same_person,
            "confidence": float(confidence),
            "analysis": normalized_analysis,
            "displacement_analysis": normalized_displacement,
            "visible_correspondences": [
                " ".join(str(item).split())[:160]
                for item in correspondences[:8]
                if isinstance(item, str) and item.strip()
            ] if isinstance(correspondences, list) else [],
        }

    async def answer_visual_question(
        self, image_jpeg: bytes, utterance: str, scene: str
    ) -> str | None:
        """Answer a deictic question from one current camera frame."""
        image_data = base64.b64encode(image_jpeg).decode("ascii")
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are the visual perception path for an embodied companion. "
                        "Answer the speaker's question using only visible pixels in this current frame. "
                        "Be direct and conversational in one short sentence. If an item is partly occluded, "
                        "say what it most likely is and express uncertainty. If the pixels do not support an "
                        "answer, say what must be moved into view. Do not identify people or infer sensitive "
                        "traits. ASR can confuse 'what' with 'where'; when the utterance contains 'am I holding', "
                        "the intended question is what item is held. Return JSON only: "
                        '{"answer":string,"grounded":boolean,"confidence":number}.\n'
                        f"Speaker utterance: {utterance!r}\nCurrent detector context: {scene[:800]}"
                    ),
                    "images": [image_data],
                }
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 96},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.vision_base_url).rstrip('/')}/api/chat", json=payload
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(
                            f"Ornith visual question HTTP {response.status}: {detail}"
                        )
                    result = await response.json()
        try:
            content = result["message"]["content"]
            parsed = json.loads(content)
            answer = parsed.get("answer")
            confidence = float(parsed.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(answer, str):
            return None
        normalized = " ".join(answer.strip().split())
        if not normalized or len(normalized) > 320 or not 0 <= confidence <= 1:
            return None
        return normalized

    @staticmethod
    def parse_object_classification(content: object) -> tuple[str, float] | None:
        analysis = OmniusClient.parse_object_analysis(content)
        if analysis is None:
            return None
        return str(analysis["label"]), float(analysis["confidence"])

    @staticmethod
    def parse_object_analysis(content: object) -> dict[str, object] | None:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        label, confidence = parsed.get("label"), parsed.get("confidence")
        if not isinstance(label, str) or not isinstance(confidence, (int, float)):
            return None
        normalized = " ".join(label.strip().split())
        if not normalized or len(normalized) > 64 or not 0 <= float(confidence) <= 1:
            return None
        visible_text = parsed.get("visible_text", False)
        text_regions = parsed.get("text_regions", [])
        if not isinstance(visible_text, bool) or not isinstance(text_regions, list):
            return None
        normalized_regions = [
            " ".join(region.strip().split())[:80]
            for region in text_regions[:4]
            if isinstance(region, str) and region.strip()
        ]
        if bool(normalized_regions) != visible_text:
            # A visual-language hint can schedule OCR but must itself be
            # internally consistent before it enters provenance.
            visible_text = bool(normalized_regions)
        return {
            "label": normalized,
            "confidence": float(confidence),
            "visible_text": visible_text,
            "text_regions": normalized_regions,
        }

    async def audit_object_label(self, profile: dict[str, object]) -> dict[str, object] | None:
        """Cheap, text-only confidence audit of an already-labelled object.

        Uses the cognition model rather than the vision model: it triages which
        profiles are worth a real image-grounded VLM re-classification instead of
        overwriting a label itself. Returns None on any malformed response so the
        call site always falls back to the VLM path rather than trusting a silent pass.
        """
        raw = await self._structured_chat(
            "Audit whether a previously assigned object label is still plausible, given its history. "
            f"Profile: {json.dumps(profile)}\n"
            "Return only JSON: {\"consistent\": bool, \"confidence\": number, \"reason\": string}. "
            "Mark consistent=false if the label history shows repeated flip-flopping, the label reads "
            "as implausible for a hand-held or nearby object, or confidence is low with few samples."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        consistent, confidence, reason = parsed.get("consistent"), parsed.get("confidence"), parsed.get("reason")
        if not isinstance(consistent, bool) or not isinstance(confidence, (int, float)) or not isinstance(reason, str):
            return None
        if not 0 <= float(confidence) <= 1 or not reason.strip() or len(reason) > 200:
            return None
        return {"consistent": consistent, "confidence": float(confidence), "reason": reason.strip()}

    async def synthesize(self, text: str) -> bytes:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/tts",
                json={"text": text, "format": "wav", "model": self.config.voice_model},
                headers=self._headers(),
            ) as response:
                response.raise_for_status()
                audio = await response.read()
        if not audio.startswith(b"RIFF"):
            raise RuntimeError("Omnius TTS response is not a WAV payload")
        return audio

    async def _structured_chat(self, prompt: str) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested JSON object with no prose or markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0,
            "tools": False,
            "think": self.config.reasoning_enabled,
            "realtime": True,
            "realtime_options": {"max_history_messages": 4, "max_tokens": 128},
        }
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/chat",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius structured chat HTTP {response.status}: {detail}")
                    result = await response.json()
        try:
            reply = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius structured chat returned an invalid completion") from error
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Omnius structured chat returned an empty completion")
        return reply.strip()

    async def _chat(
        self, prompt: str, *, remember: bool = True, include_memory: bool = True
    ) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Egg, an embodied companion with continuous, natural discourse. Reply naturally, briefly, and helpfully. "
                        "Use only the current scene and conversation as evidence; never invent actions, objects, "
                        "locations, or facts. If the request cannot be answered from that context, ask a short "
                        "clarifying question. Every reply must address a concrete object in the observed scene or "
                        "the speaker's exact question; avoid vague phrases such as 'try something new', 'how can "
                        "I help', or generic invitations. Do not use stock greetings, acknowledgements, emojis, "
                        "or phrases such as 'I hear you clearly' or 'ready to listen'. Do not identify people, "
                        "infer sensitive traits, or describe personal appearance. Treat retrieved graph claims "
                        "as the only support for remembered facts, preserve stated uncertainty and contradictions, "
                        "and never claim first-person perceptual certainty beyond the supplied live evidence."
                    ),
                },
                *(self._conversation if include_memory else []),
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.6,
            "tools": False,
            "think": self.config.reasoning_enabled,
            "realtime": True,
            "realtime_options": {"max_history_messages": 12, "max_tokens": 160},
        }
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/chat",
                    json=payload,
                    headers=self._headers(),
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius chat HTTP {response.status}: {detail}")
                    result = await response.json()
        try:
            reply = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius returned an invalid chat completion") from error
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Omnius returned an empty chat completion")
        response = reply.strip()
        if remember:
            self._conversation.extend(
                ({"role": "user", "content": prompt}, {"role": "assistant", "content": response})
            )
            self._conversation = self._conversation[-12:]
        return response
