from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import math
import os
import tempfile
import time
import wave
import zlib
from datetime import datetime, timezone

import aiohttp
import numpy as np

from egg_companion.config import OmniusConfig
from egg_companion.cognition.dialogue import (
    InterruptionDecision,
    parse_interruption_decision,
)


logger = logging.getLogger(__name__)


class OmniusClient:
    _UNSAFE_LIVE_ASR_MODELS = {
        "large-v3": "exceeds the live Jetson memory budget and can terminate Omnius",
    }
    _DEFAULT_SYSTEM_PROMPT = (
        "You are Egg, an embodied companion. Reply briefly and naturally. "
        "Use only the observed scene and conversation as evidence. Never invent "
        "objects, actions, or facts. If unsure, ask a short clarifying question. "
        "Avoid stock phrases, emojis, and greetings."
    )

    def __init__(self, config: OmniusConfig) -> None:
        self.config = config
        self._conversation: list[dict[str, str]] = []
        self._system_prompt: str = self._DEFAULT_SYSTEM_PROMPT
        # Conversational gate: used by chat/reply calls. Background VLM calls
        # never hold this gate, so a human turn is never blocked by object
        # classification or person comparison.
        self._conversational_gate = asyncio.Lock()
        # Background gate: used by VLM calls (object classification, person
        # comparison, visual questions). These run when the conversational gate
        # is free. If speech starts, background tasks cancel via
        # _background_visual_tasks and release this gate.
        self._background_gate = asyncio.Lock()
        # Legacy alias kept for callers not yet migrated.
        self._model_gate = self._conversational_gate
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

    async def companion_reply(
        self,
        scene: str,
        *,
        history: list[dict[str, object]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "system",
                "content": (
                    "You are in proactive observation mode. The room is quiet and a person "
                    "is present. Offer one concise, useful spoken observation or next step "
                    "grounded in the following evidence. Never invent facts or connections."
                ),
            },
        ]
        if history:
            for turn in history[-10:]:
                role = "assistant" if turn.get("role") == "agent" else "user"
                text = str(turn.get("text") or "")
                if text:
                    messages.append({"role": role, "content": text})
        messages.append({"role": "user", "content": f"Observed scene: {scene}"})
        return await self._chat(messages=messages, remember=False, include_memory=False)

    async def conversation_reply(
        self,
        utterance: str,
        context: str,
        history: list[dict[str, object]] | None = None,
    ) -> str:
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt},
            {
                "role": "system",
                "content": (
                    "The following is your current cognitive context. Use it as evidence "
                    "but never treat it as stronger than its sources. Do not read URLs aloud."
                ),
            },
            {"role": "system", "content": context},
        ]
        for turn in (history or [])[-8:]:
            role = "assistant" if turn.get("role") == "agent" else "user"
            text = str(turn.get("text") or "")
            status = str(turn.get("status") or "")
            if not text:
                continue
            if role == "assistant" and status in {"interrupted", "superseded"}:
                messages.append({
                    "role": "assistant",
                    "content": f"[interrupted] {text}",
                })
            else:
                messages.append({"role": role, "content": text})
        messages.append({
            "role": "user",
            "content": (
                f"Local speech, already verified as human speech by VAD:\n{utterance!r}\n\n"
                "Decide from the conversational history and context whether this is directed "
                "to Egg. When the cognitive context supplies a user-provided preferred name, "
                "use it naturally when useful, without repeating it in every reply. When WEB "
                "SEARCH TOOL EVIDENCE is present in the context, ground current facts in that "
                "evidence and do not read URLs aloud. Never answer in the human's voice, "
                "answer Egg's own prior question, or treat an interrupted agent utterance as "
                "fully heard. If it is not directed to Egg or does not merit an audible "
                "interruption, reply exactly [[SILENT]]."
            ),
        })
        return await self._chat(messages=messages, remember=False, include_memory=False)

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
            "\"conversation\",\"confidence\":number,\"tool\":\"none\"|\"vision\"|\"web_search\","
            "\"tool_query\":string|null}. Select vision when an accurate reply depends on the "
            "speaker's presently visible scene, gesture, held or indicated item, spatial relation, "
            "display, or readable text. Select web_search for an explicit request to search or "
            "for current external information that cannot be grounded in the embodied context. "
            "Otherwise select none. tool must be exactly none, vision, or web_search. For vision, "
            "make tool_query the concise visual question without answering it. For web_search, make "
            "tool_query a concise standalone search query."
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
        if tool not in {"none", "vision", "web_search"}:
            return None
        if tool in {"vision", "web_search"}:
            if not isinstance(tool_query, str) or not tool_query.strip():
                tool_query = utterance if tool == "vision" else None
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

    async def plan_narrative_dream(
        self,
        daily_evidence: dict[str, object],
        constitution: dict[str, object],
        prior_policy: dict[str, object],
    ) -> dict[str, object] | None:
        """Let the cognition model decide which evidence tools it needs."""
        raw = await self._narrative_structured_chat(
            "Plan an evidence-grounded dream synthesis. The narrative constitution is mutable "
            "strategy, not evidence. Decide whether additional tools are needed before synthesis.\n"
            f"Constitution: {self._bounded_prompt_json(constitution, 1000)}\n"
            "Prior model-authored observation policy: "
            f"{self._bounded_prompt_json(prior_policy, 1000)}\n"
            f"Daily provenance ledger: {self._bounded_prompt_json(daily_evidence, 5000)}\n"
            "Return JSON with exactly: {\"tool_requests\":[{\"tool\":\"memory_search\"|"
            "\"graph_inspect\"|\"evidence_inspect\"|\"web_search\",\"query\":string|null,"
            "\"entity_ids\":[string],\"evidence_ids\":[string],\"purpose\":string}],"
            "\"planning_summary\":string}. Request web search only when external knowledge is "
            "needed to interpret evidence; local memory and artifacts are authoritative for lived events."
            " Return at most two requests, at most four IDs in each reference list, keep query and "
            "purpose under 180 characters, and finish the complete JSON under 1,800 characters.",
            max_tokens=512,
        )
        parsed = self._parse_narrative_plan(raw)
        if parsed is not None:
            return parsed
        repaired = await self._narrative_structured_chat(
            "Reformat the prior tool plan to the exact contract without adding facts. Return only "
            "{\"tool_requests\":[{\"tool\":\"memory_search\"|\"graph_inspect\"|"
            "\"evidence_inspect\"|\"web_search\",\"query\":string|null,\"entity_ids\":[string],"
            "\"evidence_ids\":[string],\"purpose\":string}],\"planning_summary\":string}. "
            "Use every field, no extra fields, at most two requests, no more than four IDs per "
            "reference list, and finish the complete JSON under 1,800 characters.\nPrior response: "
            + self._bounded_prompt_json(raw, 3000),
            max_tokens=512,
        )
        parsed = self._parse_narrative_plan(repaired)
        if parsed is None:
            logger.warning(
                "narrative planner violated its JSON contract; initial=%r repair=%r",
                raw[:2000],
                repaired[:2000],
            )
        return parsed

    @classmethod
    def _parse_narrative_plan(cls, raw: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if set(parsed) != {"tool_requests", "planning_summary"}:
            return None
        requests = parsed.get("tool_requests")
        summary = parsed.get("planning_summary")
        if (
            not isinstance(requests, list)
            or len(requests) > 4
            or not isinstance(summary, str)
            or not summary.strip()
        ):
            return None
        normalized: list[dict[str, object]] = []
        for request in requests:
            if (
                not isinstance(request, dict)
                or set(request) != {
                    "tool", "query", "entity_ids", "evidence_ids", "purpose"
                }
                or request.get("tool") not in {
                "memory_search", "graph_inspect", "evidence_inspect", "web_search"
                }
            ):
                return None
            query = request.get("query")
            entity_ids = request.get("entity_ids", [])
            evidence_ids = request.get("evidence_ids", [])
            purpose = request.get("purpose")
            if query is not None and (
                not isinstance(query, str) or not query.strip()
            ):
                return None
            if not isinstance(entity_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in entity_ids
            ):
                return None
            if not isinstance(evidence_ids, list) or not all(
                isinstance(value, str) and value.strip() for value in evidence_ids
            ):
                return None
            if not isinstance(purpose, str) or not purpose.strip():
                return None
            normalized.append(
                {
                    "tool": request["tool"],
                    "query": " ".join(query.split())[:300] if query else None,
                    "entity_ids": [value[:300] for value in entity_ids[:12]],
                    "evidence_ids": [value[:300] for value in evidence_ids[:12]],
                    "purpose": " ".join(purpose.split())[:300],
                }
            )
        return {
            "tool_requests": normalized,
            "planning_summary": " ".join(summary.split())[:1000],
        }

    async def synthesize_narrative_dream(
        self,
        daily_evidence: dict[str, object],
        constitution: dict[str, object],
        prior_policy: dict[str, object],
        plan: dict[str, object],
        tool_results: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Produce learned narrative semantics and a self-revisable future policy."""
        core_context = (
            "Synthesize one day into an evolving associative world account. Do not expose hidden "
            "reasoning. Do not use lexical frequency or detector repetition as meaning by itself. "
            "Distinguish sensory evidence, heard speech, agent speech, tool evidence, inference, and "
            "uncertainty. The observation policy must be learned from this evidence and the prior "
            "policy, not copied as a static checklist. Treat social-reflection affect and behavioral "
            "records as time-local uncertain interpretations, never fixed traits. Use response-feedback "
            "evidence to assess whether Egg's communication worked, and let supported outcomes revise "
            "future interaction strategy, curiosity, tone, and question choice. Constitution changes "
            "must improve the general "
            "future method rather than encode facts or keywords from this day. Your response is the "
            "semantic interpretation pass: do not describe the account as still awaiting or pending "
            "model semantics merely because the provenance ledger carries that pre-synthesis state. "
            "Resolve what the evidence supports and state remaining epistemic uncertainty directly.\n"
            f"Constitution: {self._bounded_prompt_json(constitution, 600)}\n"
            f"Prior policy: {self._bounded_prompt_json(prior_policy, 600)}\n"
            f"Daily ledger: {self._bounded_prompt_json(daily_evidence, 2200)}\n"
            f"Plan: {self._bounded_prompt_json(plan, 500)}\n"
            f"Tool results: {self._bounded_prompt_json(tool_results, 1000)}\n"
        )
        core_raw = await self._narrative_structured_chat(
            core_context
            + "Return the narrative core only as JSON with exactly: narrative_summary string, "
            "story_update string, themes, episodes. themes contain label, summary, confidence, "
            "entity_ids, evidence_ids, context_ids. episodes contain title, summary, significance, "
            "confidence, started_at, ended_at, entity_ids, evidence_ids, context_ids. Use empty "
            "reference arrays when there is no supported reference. Return at most two themes and "
            "two episodes, selected by you. Keep each prose field under 500 characters and each "
            "reference array to at most four IDs so the complete JSON stays under 4,500 characters.",
            max_tokens=1200,
        )
        core = self._parse_narrative_core(core_raw)
        if core is None:
            core_repair = await self._narrative_structured_chat(
                "Re-author the prior narrative core to the exact requested JSON contract without "
                "adding evidence or conclusions. Preserve only supported content. Return exactly "
                "narrative_summary, story_update, themes, episodes; include confidence and empty "
                "entity_ids, evidence_ids, and context_ids arrays where no reference exists. "
                "Use at most two themes and two episodes, prose fields under 500 characters, no "
                "more than four IDs per reference array, and finish the JSON under 4,500 characters.\n"
                "Prior response: " + self._bounded_prompt_json(core_raw, 3000),
                max_tokens=1200,
            )
            core = self._parse_narrative_core(core_repair)
            if core is None:
                logger.warning(
                    "narrative core violated its JSON contract; initial=%r repair=%r",
                    core_raw[:2000],
                    core_repair[:2000],
                )
                return None
        reflection_context = (
            "Reflect on an evidence-grounded daily narrative to revise future attention and inquiry. "
            "Do not expose hidden reasoning. Preserve uncertainty and provenance. Changes to the "
            "constitution must improve the general method rather than encode this day's facts or "
            "keywords. Curiosity must arise from unresolved evidence and relationship context, not a "
            "canned question template. Compare conversation outcomes with prior interaction strategy "
            "and revise the method when feedback supports it.\n"
            f"Constitution: {self._bounded_prompt_json(constitution, 600)}\n"
            f"Prior policy: {self._bounded_prompt_json(prior_policy, 700)}\n"
            f"Daily ledger: {self._bounded_prompt_json(daily_evidence, 1800)}\n"
            f"Tool results: {self._bounded_prompt_json(tool_results, 800)}\n"
        )
        reflection_raw = await self._narrative_structured_chat(
            reflection_context
            + f"Narrative core already authored: {self._bounded_prompt_json(core, 1600)}\n"
            "Return reflective policy only as JSON with exactly: unresolved_questions, learned_context, "
            "observation_policy, constitution_update. unresolved_questions and learned_context contain "
            "summary, confidence, entity_ids, evidence_ids, context_ids. observation_policy has summary, "
            "attend_to, deprioritize, open_questions; every item contains summary, reason, action, predicate, "
            "confidence, entity_ids, evidence_ids, context_ids. predicate is a claim predicate or null. "
            "attend_to actions: observe, retrieve, ask, speak. deprioritize action: deprioritize. "
            "open_questions actions: observe, retrieve, ask. constitution_update is a general revised "
            "constitution string or null. Use empty arrays where appropriate and at most two items per list. "
            "Keep each prose field under 400 characters and each reference array to at most four IDs.",
            max_tokens=1000,
        )
        reflection = self._parse_narrative_reflection(reflection_raw)
        if reflection is None:
            reflection_repair = await self._narrative_structured_chat(
                "Re-author the prior reflective policy to the exact requested JSON contract without "
                "adding evidence or conclusions. Return exactly unresolved_questions, learned_context, "
                "observation_policy, constitution_update. observation_policy must be an object shaped "
                "exactly as {\"summary\":string,\"attend_to\":[],\"deprioritize\":[],"
                "\"open_questions\":[]}, never an array. Include every required field and empty "
                "reference arrays where no reference exists; predicate may be null. Use only the "
                "allowed actions and at most two items per list. Keep each prose field under 400 "
                "characters and each reference array to at most four IDs.\nPrior response: "
                + self._bounded_prompt_json(reflection_raw, 3000),
                max_tokens=1000,
            )
            reflection = self._parse_narrative_reflection(reflection_repair)
            if reflection is None:
                logger.warning(
                    "narrative reflection violated its JSON contract; initial=%r repair=%r",
                    reflection_raw[:2000],
                    reflection_repair[:2000],
                )
                return None
        combined = {"version": 1, **core, **reflection}
        return self._parse_narrative_synthesis(
            json.dumps(combined, ensure_ascii=False, separators=(",", ":"))
        )

    @classmethod
    def _parse_narrative_core(cls, raw: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {
            "narrative_summary", "story_update", "themes", "episodes"
        }:
            return None
        if not cls._bounded_text(parsed.get("narrative_summary"), 6000):
            return None
        story = parsed.get("story_update")
        if isinstance(story, dict):
            for key in ("entity_ids", "evidence_ids", "context_ids"):
                story.setdefault(key, [])
            if not cls._narrative_items(
                [story], maximum=1, text_fields={"summary": 2000}
            ):
                return None
        elif not cls._bounded_text(story, 8000):
            return None
        for key in ("themes", "episodes"):
            values = parsed.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        for reference_key in (
                            "entity_ids", "evidence_ids", "context_ids"
                        ):
                            item.setdefault(reference_key, [])
        if not cls._narrative_items(
            parsed.get("themes"), maximum=8,
            text_fields={"label": 200, "summary": 2000},
        ) or not cls._narrative_items(
            parsed.get("episodes"), maximum=8,
            text_fields={"title": 200, "summary": 2400, "significance": 1600},
            temporal=True,
        ):
            return None
        return parsed

    @classmethod
    def _parse_narrative_reflection(cls, raw: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {
            "unresolved_questions", "learned_context", "observation_policy",
            "constitution_update",
        }:
            return None
        flat_policy = parsed.get("observation_policy")
        if isinstance(flat_policy, list):
            for item in flat_policy:
                if isinstance(item, dict):
                    item.setdefault("predicate", None)
                    for reference_key in (
                        "entity_ids", "evidence_ids", "context_ids"
                    ):
                        item.setdefault(reference_key, [])
            if not cls._narrative_items(
                flat_policy,
                maximum=8,
                text_fields={"summary": 1200, "reason": 1600},
                enum_fields={
                    "action": {
                        "observe", "retrieve", "ask", "speak", "deprioritize"
                    }
                },
                nullable_text_fields={"predicate": 200},
            ):
                return None
            active = [
                item for item in flat_policy
                if item.get("action") != "deprioritize"
            ]
            deprioritized = [
                item for item in flat_policy
                if item.get("action") == "deprioritize"
            ]
            summaries = [str(item["summary"]) for item in flat_policy]
            parsed["observation_policy"] = {
                "summary": " ".join(summaries)[:4000],
                "attend_to": active,
                "deprioritize": deprioritized,
                "open_questions": [],
            }
        for key in ("unresolved_questions", "learned_context"):
            values = parsed.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        for reference_key in (
                            "entity_ids", "evidence_ids", "context_ids"
                        ):
                            item.setdefault(reference_key, [])
            if not cls._narrative_items(
                values, maximum=8, text_fields={"summary": 2000}
            ):
                return None
        policy = parsed.get("observation_policy")
        if not isinstance(policy, dict) or set(policy) != {
            "summary", "attend_to", "deprioritize", "open_questions"
        } or not cls._bounded_text(policy.get("summary"), 4000):
            return None
        policy_actions = {
            "attend_to": {"observe", "retrieve", "ask", "speak"},
            "deprioritize": {"deprioritize"},
            "open_questions": {"observe", "retrieve", "ask"},
        }
        for key, actions in policy_actions.items():
            values = policy.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        item.setdefault("predicate", None)
                        for reference_key in (
                            "entity_ids", "evidence_ids", "context_ids"
                        ):
                            item.setdefault(reference_key, [])
            if not cls._narrative_items(
                values, maximum=8,
                text_fields={"summary": 1200, "reason": 1600},
                enum_fields={"action": actions},
                nullable_text_fields={"predicate": 200},
            ):
                return None
        constitution = parsed.get("constitution_update")
        if constitution is not None and not cls._bounded_text(constitution, 6000):
            return None
        return parsed

    @classmethod
    def _parse_narrative_synthesis(cls, raw: str) -> dict[str, object] | None:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict):
            return None
        if set(parsed) != {
            "version",
            "narrative_summary",
            "story_update",
            "themes",
            "episodes",
            "unresolved_questions",
            "learned_context",
            "observation_policy",
            "constitution_update",
        } or parsed.get("version") != 1:
            return None
        story_update_detail: dict[str, object] | None = None
        if isinstance(parsed.get("story_update"), dict):
            story_update_detail = dict(parsed["story_update"])
            for reference_key in ("entity_ids", "evidence_ids", "context_ids"):
                story_update_detail.setdefault(reference_key, [])
            if not cls._narrative_items(
                [story_update_detail],
                maximum=1,
                text_fields={"summary": 2000},
            ):
                return None
            parsed = {**parsed, "story_update": story_update_detail.get("summary")}
        for key in ("themes", "episodes", "unresolved_questions", "learned_context"):
            values = parsed.get(key)
            if isinstance(values, list):
                for item in values:
                    if isinstance(item, dict):
                        for reference_key in (
                            "entity_ids", "evidence_ids", "context_ids"
                        ):
                            item.setdefault(reference_key, [])
        policy_value = parsed.get("observation_policy")
        if isinstance(policy_value, dict):
            for key in ("attend_to", "deprioritize", "open_questions"):
                values = policy_value.get(key)
                if isinstance(values, list):
                    for item in values:
                        if isinstance(item, dict):
                            item.setdefault("predicate", None)
                            for reference_key in (
                                "entity_ids", "evidence_ids", "context_ids"
                            ):
                                item.setdefault(reference_key, [])
        if not cls._bounded_text(parsed.get("narrative_summary"), 6000):
            return None
        if not cls._bounded_text(parsed.get("story_update"), 8000):
            return None
        if not cls._narrative_items(
            parsed.get("themes"),
            maximum=20,
            text_fields={"label": 200, "summary": 2000},
        ):
            return None
        if not cls._narrative_items(
            parsed.get("episodes"),
            maximum=48,
            text_fields={"title": 200, "summary": 2400, "significance": 1600},
            temporal=True,
        ):
            return None
        for key, maximum in (("unresolved_questions", 20), ("learned_context", 30)):
            if not cls._narrative_items(
                parsed.get(key), maximum=maximum, text_fields={"summary": 2000}
            ):
                return None
        policy = parsed.get("observation_policy")
        if (
            not isinstance(policy, dict)
            or set(policy) != {"summary", "attend_to", "deprioritize", "open_questions"}
            or not cls._bounded_text(policy.get("summary"), 4000)
        ):
            return None
        policy_actions = {
            "attend_to": {"observe", "retrieve", "ask", "speak"},
            "deprioritize": {"deprioritize"},
            "open_questions": {"observe", "retrieve", "ask"},
        }
        for key, allowed_actions in policy_actions.items():
            if not cls._narrative_items(
                policy.get(key),
                maximum=20,
                text_fields={"summary": 1200, "reason": 1600},
                enum_fields={"action": allowed_actions},
                nullable_text_fields={"predicate": 200},
            ):
                return None
        constitution_update = parsed.get("constitution_update")
        if constitution_update is not None and not cls._bounded_text(
            constitution_update, 6000
        ):
            return None
        if story_update_detail is not None:
            parsed["story_update_detail"] = story_update_detail
        return parsed

    async def review_narrative_constitution_update(
        self,
        current_constitution: dict[str, object],
        proposed_constitution: str,
    ) -> dict[str, object] | None:
        """Ask a separate model pass to review reversible prompt self-modification."""
        raw = await self._narrative_structured_chat(
            "Review a proposed update to an evolving narrative constitution. Accept or revise it "
            "only when it improves the general evidence-grounding, associative synthesis, tool-use, "
            "uncertainty, or attention method. It must remain applicable to future unknown experience "
            "and must not encode people, objects, phrases, conclusions, or keywords from one lived day. "
            "Do not expose hidden reasoning.\n"
            f"Current constitution: {self._bounded_prompt_json(current_constitution, 7000)}\n"
            f"Proposed constitution: {self._bounded_prompt_json(proposed_constitution, 7000)}\n"
            "Return only JSON with exactly: accepted (boolean), constitution (the accepted/revised "
            "string or null), review_summary (a concise inspectable explanation).",
            max_tokens=1024,
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if (
            not isinstance(parsed, dict)
            or set(parsed) != {"accepted", "constitution", "review_summary"}
            or not isinstance(parsed.get("accepted"), bool)
            or not self._bounded_text(parsed.get("review_summary"), 1200)
        ):
            return None
        constitution = parsed.get("constitution")
        if parsed["accepted"]:
            if not self._bounded_text(constitution, 6000):
                return None
        elif constitution is not None:
            return None
        return parsed

    async def self_assess_and_update_prompt(
        self,
        current_system_prompt: str,
        recent_interactions: list[dict[str, object]],
        cognitive_documents: dict[str, str],
        daily_recaps: list[str],
        constitution: dict[str, object],
    ) -> dict[str, object] | None:
        """Meta-cognitive self-assessment: review recent interaction outcomes, day-over-day
        recaps, and cognitive documents to produce an updated system prompt and directive
        revisions for communication strategy, observation policy, and interaction strategy.

        Returns dict with: system_prompt, communication_directive, observation_directive,
        interaction_directive, assessment_summary. Any field may be null to signal 'no change'.
        """
        interaction_summary = json.dumps(
            recent_interactions[-20:], ensure_ascii=False, default=str
        )[:3000]
        doc_summary = "\n\n".join(
            f"[{kind}]\n{text[:800]}"
            for kind, text in cognitive_documents.items()
        )[:2500]
        recap_text = "\n---\n".join(recap[:600] for recap in daily_recaps[-3:])[:1800]
        prompt = (
            "You are Egg performing a periodic self-assessment of your own prompt drivers. "
            "Review recent interaction outcomes, day-over-day recaps, and your current cognitive "
            "documents. Produce an updated system prompt that improves on the prior version based "
            "on what you have learned. Also produce revised directives for communication strategy, "
            "observation policy, and interaction strategy where the evidence supports a change. "
            "When the evidence is insufficient to justify a revision, return null for that field. "
            "The system prompt must remain grounded: it sets identity, grounding constraints, "
            "silence-gating rules, and the voice/persona tone. Do not encode specific people, "
            "objects, or episode details into the system prompt -- keep it as durable method. "
            "Do not expose hidden reasoning.\n\n"
            f"Current system prompt:\n{current_system_prompt[:2000]}\n\n"
            f"Recent interaction outcomes:\n{interaction_summary}\n\n"
            f"Cognitive documents:\n{doc_summary}\n\n"
            f"Day-over-day recaps:\n{recap_text}\n\n"
            f"Constitution: {self._bounded_prompt_json(constitution, 600)}\n\n"
            "Return JSON only with exactly: system_prompt (the full revised system prompt string "
            "or null), communication_directive (revised communication strategy directive or null), "
            "observation_directive (revised observation policy directive or null), "
            "interaction_directive (revised interaction strategy directive or null), "
            "assessment_summary (a concise explanation of what changed and why). "
            "Keep system_prompt under 1500 characters. Keep each directive under 400 characters."
        )
        raw = await self._narrative_structured_chat(prompt, max_tokens=1400)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("self-assessment JSON decode failed: %r", raw[:500])
            return None
        if not isinstance(parsed, dict):
            return None
        required = {
            "system_prompt", "communication_directive", "observation_directive",
            "interaction_directive", "assessment_summary",
        }
        if set(parsed.keys()) != required:
            logger.warning("self-assessment keys mismatch: %s", set(parsed.keys()))
            return None
        if parsed.get("system_prompt") is not None and not self._bounded_text(
            parsed["system_prompt"], 2000
        ):
            return None
        if parsed.get("assessment_summary") and not self._bounded_text(
            parsed["assessment_summary"], 1200
        ):
            return None
        return parsed

    async def interpret_proactive_answer(
        self,
        question: str,
        utterance: str,
        predicate: str | None,
    ) -> dict[str, object] | None:
        """Resolve a reply to a model-authored question without lexical rules."""
        raw = await self._structured_chat(
            "Determine how the new utterance relates to a proactive question you previously asked. "
            "Return JSON only with relation ('answer', 'unknown', or 'unrelated'), value (the concise "
            "claim value or null), and reply (a concise natural acknowledgement or null). Do not turn "
            "an unrelated utterance into an answer.\n"
            f"Question: {question}\nClaim predicate, if any: {predicate or 'none'}\n"
            f"New utterance: {utterance}"
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {"relation", "value", "reply"}:
            return None
        relation = parsed.get("relation")
        value = parsed.get("value")
        reply = parsed.get("reply")
        if relation not in {"answer", "unknown", "unrelated"}:
            return None
        if value is not None and not self._bounded_text(value, 1200):
            return None
        if reply is not None and not self._bounded_text(reply, 600):
            return None
        if relation == "answer" and (value is None or reply is None):
            return None
        if relation == "unknown" and reply is None:
            return None
        if relation == "unrelated" and (value is not None or reply is not None):
            return None
        return parsed

    @staticmethod
    def _bounded_prompt_json(value: object, maximum: int) -> str:
        """Serialize model context with a transparent, semantics-agnostic capacity bound."""
        encoded = json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))
        if len(encoded) <= maximum:
            return encoded
        return json.dumps(
            {
                "capacity_truncated": True,
                "original_characters": len(encoded),
                "serialized_prefix": encoded[:maximum],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    @staticmethod
    def _bounded_text(value: object, maximum: int) -> bool:
        return isinstance(value, str) and bool(value.strip()) and len(value) <= maximum

    @classmethod
    def _reference_list(cls, value: object, *, maximum: int = 32) -> bool:
        return (
            isinstance(value, list)
            and len(value) <= maximum
            and all(cls._bounded_text(item, 300) for item in value)
        )

    @classmethod
    def _narrative_items(
        cls,
        value: object,
        *,
        maximum: int,
        text_fields: dict[str, int],
        temporal: bool = False,
        enum_fields: dict[str, set[str]] | None = None,
        nullable_text_fields: dict[str, int] | None = None,
    ) -> bool:
        if not isinstance(value, list) or len(value) > maximum:
            return False
        expected = {
            *text_fields,
            *(enum_fields or {}),
            *(nullable_text_fields or {}),
            "confidence",
            "entity_ids",
            "evidence_ids",
            "context_ids",
        }
        if temporal:
            expected.update({"started_at", "ended_at"})
        for item in value:
            if not isinstance(item, dict) or set(item) != expected:
                return False
            if any(
                not cls._bounded_text(item.get(field), bound)
                for field, bound in text_fields.items()
            ):
                return False
            if any(
                item.get(field) not in allowed
                for field, allowed in (enum_fields or {}).items()
            ):
                return False
            if any(
                item.get(field) is not None
                and not cls._bounded_text(item.get(field), bound)
                for field, bound in (nullable_text_fields or {}).items()
            ):
                return False
            confidence = item.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                return False
            if any(
                not cls._reference_list(item.get(field))
                for field in ("entity_ids", "evidence_ids", "context_ids")
            ):
                return False
            if temporal and any(
                item.get(field) is not None
                and not cls._bounded_text(item.get(field), 80)
                for field in ("started_at", "ended_at")
            ):
                return False
        return True

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

    async def compose_identity_question(
        self,
        profile: dict[str, object],
        scene: str,
        history: list[dict[str, object]],
        interaction_strategy: dict[str, object],
    ) -> dict[str, object] | None:
        raw = await self._structured_chat(
            "Decide whether and how Egg should ask a presently visible, face-grounded but unnamed "
            "person what they prefer to be called. This is social dialogue, not identity inference. "
            "Use encounter history, the audible conversation, current grounded scene, and Egg's "
            "evolving interaction strategy. Do not infer a name, demographic, personality, mental "
            "state, or relationship. Ask only if it is contextually natural; otherwise decline.\n"
            f"Face profile: {self._bounded_prompt_json(profile, 1200)}\n"
            f"Scene: {scene[:1400]}\n"
            f"Conversation: {self._bounded_prompt_json(history, 2200)}\n"
            f"Interaction strategy: {self._bounded_prompt_json(interaction_strategy, 1400)}\n"
            "Return only JSON: {\"speak\":boolean,\"question\":string|null,"
            "\"reason\":string,\"confidence\":number}. A spoken question must be one concise, "
            "natural sentence and must not claim prior familiarity that the evidence lacks."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        speak = parsed.get("speak")
        question = parsed.get("question")
        reason = parsed.get("reason")
        confidence = parsed.get("confidence")
        if (
            not isinstance(speak, bool)
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 400
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
        ):
            return None
        if speak:
            if not isinstance(question, str):
                return None
            question = " ".join(question.split())
            if not question or len(question) > 240:
                return None
        elif question is not None:
            return None
        return {
            "speak": speak,
            "question": question,
            "reason": " ".join(reason.split()),
            "confidence": float(confidence),
        }

    async def compose_identity_acknowledgement(
        self,
        preferred_name: str,
        utterance: str,
        history: list[dict[str, object]],
        interaction_strategy: dict[str, object],
    ) -> str | None:
        raw = await self._structured_chat(
            "A presently visible person has just explicitly supplied their preferred name after "
            "Egg asked. Author Egg's next brief, natural response in the ongoing conversation. "
            "Acknowledge the information without inventing familiarity or using a canned greeting, "
            "and use the name only as naturally as this moment warrants.\n"
            f"Preferred name: {preferred_name!r}\n"
            f"Utterance: {utterance!r}\n"
            f"Ordered conversation: {self._bounded_prompt_json(history, 2200)}\n"
            f"Interaction strategy: {self._bounded_prompt_json(interaction_strategy, 1600)}\n"
            "Return only JSON with exactly one key: {\"reply\":string}. Keep it to one concise "
            "spoken sentence."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {"reply"}:
            return None
        reply = parsed.get("reply")
        if not self._bounded_text(reply, 300):
            return None
        return " ".join(str(reply).split())

    async def compose_curiosity_question(
        self,
        candidate: dict[str, object],
        visible_people: list[str],
        scene: str,
        history: list[dict[str, object]],
        interaction_strategy: dict[str, object],
    ) -> dict[str, object] | None:
        raw = await self._structured_chat(
            "Egg's dream model selected a source-backed unresolved thread while a person is present. "
            "Decide whether asking now is genuinely useful and socially natural. If so, author one "
            "concise spoken question grounded in the selected thread, current scene, ordered dialogue, "
            "and evolving interaction strategy. Do not mechanically repeat the candidate summary, bolt "
            "a name onto a template, invent familiarity, or ask merely to appear curious.\n"
            f"Selected thread: {self._bounded_prompt_json(candidate, 1800)}\n"
            f"Visible preferred names: {self._bounded_prompt_json(visible_people, 400)}\n"
            f"Scene: {scene[:1400]}\n"
            f"Conversation: {self._bounded_prompt_json(history, 2200)}\n"
            f"Interaction strategy: {self._bounded_prompt_json(interaction_strategy, 1400)}\n"
            "Return only JSON: {\"speak\":boolean,\"question\":string|null,"
            "\"reason\":string,\"confidence\":number}. If speak is false, question must be null."
        )
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return None
        if set(parsed) != {"speak", "question", "reason", "confidence"}:
            return None
        speak = parsed.get("speak")
        question = parsed.get("question")
        reason = parsed.get("reason")
        confidence = parsed.get("confidence")
        if (
            not isinstance(speak, bool)
            or not isinstance(reason, str)
            or not reason.strip()
            or len(reason) > 400
            or not isinstance(confidence, (int, float))
            or isinstance(confidence, bool)
            or not 0 <= float(confidence) <= 1
        ):
            return None
        if speak:
            if not isinstance(question, str):
                return None
            question = " ".join(question.split())
            if not question or len(question) > 300:
                return None
        elif question is not None:
            return None
        return {
            "speak": speak,
            "question": question,
            "reason": " ".join(reason.split()),
            "confidence": float(confidence),
        }

    async def reflect_social_interaction(
        self,
        interaction: dict[str, object],
        history: list[dict[str, object]],
        prior_strategy: dict[str, object],
        prior_profiles: list[dict[str, object]],
    ) -> dict[str, object] | None:
        """Create revisable, evidence-bound social and response feedback."""

        raw = await self._structured_chat(
            "Reflect on one completed embodied conversation turn. Describe apparent affect and "
            "communicative behavior only as uncertain, time-local interpretations supported by the "
            "utterance and interaction outcome. Never diagnose, infer protected traits, assign a fixed "
            "personality, or treat tone as fact. Compare Egg's response with what followed in the "
            "ordered history and propose a strategy revision only when the evidence supports a concrete "
            "improvement. The strategy is a revisable communication method, never a fact about a person. "
            "For each visibly grounded person whose evidence is distinguishable, update a longitudinal "
            "interaction profile: synthesize sentiment trajectory, observed communication patterns, "
            "explicitly evidenced interaction preferences, and uncertainties across prior and current "
            "evidence. A profile describes Egg's interaction evidence, not a person's essence; preserve "
            "contradictions and uncertainty, and emit no update when attribution is ambiguous. "
            "Do not claim response speed, success, trust, preference, or a relationship change unless the "
            "provided fields or a subsequent human turn directly support it; report insufficient evidence "
            "with low confidence instead.\n"
            f"Interaction: {self._bounded_prompt_json(interaction, 2400)}\n"
            f"Ordered conversation: {self._bounded_prompt_json(history, 3000)}\n"
            f"Prior strategy: {self._bounded_prompt_json(prior_strategy, 1800)}\n"
            f"Prior social profiles: {self._bounded_prompt_json(prior_profiles, 3000)}\n"
            "Return only JSON with exactly: momentary_affect, communicative_behavior, "
            "relationship_update, response_feedback, strategy_revision, profile_updates. "
            "momentary_affect has label "
            "string, valence number -1..1, arousal number 0..1, confidence number 0..1, evidence string. "
            "communicative_behavior, relationship_update, and response_feedback each have summary, "
            "confidence, evidence strings. strategy_revision is null or has directive, rationale, and "
            "confidence. profile_updates is an array; each item has exactly subject_id, summary, "
            "sentiment_trajectory, communication_patterns, interaction_preferences, uncertainties, "
            "confidence, evidence. The three plural fields are arrays of strings. subject_id must be "
            "one of interaction.visible_person_ids. Keep arrays to at most 8 items and every string "
            "under 400 characters.",
            max_tokens=900,
        )
        return self.parse_social_reflection(raw)

    @staticmethod
    def parse_social_reflection(content: object) -> dict[str, object] | None:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(parsed, dict) or set(parsed) != {
            "momentary_affect",
            "communicative_behavior",
            "relationship_update",
            "response_feedback",
            "strategy_revision",
            "profile_updates",
        }:
            return None
        affect = parsed.get("momentary_affect")
        if not isinstance(affect, dict) or set(affect) != {
            "label", "valence", "arousal", "confidence", "evidence"
        }:
            return None
        if not all(
            isinstance(affect.get(key), (int, float))
            and not isinstance(affect.get(key), bool)
            for key in ("valence", "arousal", "confidence")
        ):
            return None
        if not (
            -1 <= float(affect["valence"]) <= 1
            and 0 <= float(affect["arousal"]) <= 1
            and 0 <= float(affect["confidence"]) <= 1
        ):
            return None
        for field in ("label", "evidence"):
            if not OmniusClient._bounded_text(affect.get(field), 400):
                return None
        for key in (
            "communicative_behavior", "relationship_update", "response_feedback"
        ):
            item = parsed.get(key)
            if not isinstance(item, dict) or set(item) != {
                "summary", "confidence", "evidence"
            }:
                return None
            if (
                not OmniusClient._bounded_text(item.get("summary"), 400)
                or not OmniusClient._bounded_text(item.get("evidence"), 400)
                or not isinstance(item.get("confidence"), (int, float))
                or isinstance(item.get("confidence"), bool)
                or not 0 <= float(item["confidence"]) <= 1
            ):
                return None
        revision = parsed.get("strategy_revision")
        if revision is not None:
            if not isinstance(revision, dict) or set(revision) != {
                "directive", "rationale", "confidence"
            }:
                return None
            if (
                not OmniusClient._bounded_text(revision.get("directive"), 1000)
                or not OmniusClient._bounded_text(revision.get("rationale"), 400)
                or not isinstance(revision.get("confidence"), (int, float))
                or isinstance(revision.get("confidence"), bool)
                or not 0 <= float(revision["confidence"]) <= 1
            ):
                return None
        updates = parsed.get("profile_updates")
        if not isinstance(updates, list) or len(updates) > 8:
            return None
        for update in updates:
            if not isinstance(update, dict) or set(update) != {
                "subject_id",
                "summary",
                "sentiment_trajectory",
                "communication_patterns",
                "interaction_preferences",
                "uncertainties",
                "confidence",
                "evidence",
            }:
                return None
            if any(
                not OmniusClient._bounded_text(update.get(field), 400)
                for field in (
                    "subject_id", "summary", "sentiment_trajectory", "evidence"
                )
            ):
                return None
            confidence = update.get("confidence")
            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not 0 <= float(confidence) <= 1
            ):
                return None
            for field in (
                "communication_patterns", "interaction_preferences", "uncertainties"
            ):
                values = update.get(field)
                if (
                    not isinstance(values, list)
                    or len(values) > 8
                    or any(
                        not OmniusClient._bounded_text(value, 400) for value in values
                    )
                ):
                    return None
        return parsed

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
                        "Adjudicate only the opaque segmented pixels in this PNG; transparent pixels and background are "
                        "not evidence. First decide whether those pixels contain a coherent physical object rather than "
                        "a fragment, shadow, texture, body part, duplicate mask, or detector hallucination. If grounded, "
                        "identify it with a short ordinary noun phrase and describe its actually visible shape, color, "
                        "parts, markings, damage, orientation, and state without inventing hidden properties. Independently "
                        "state whether the detector's proposed category is supported. Do not return a scene, genre, "
                        "material, person occupation, or visual style as the object label. "
                        "Independently inspect the opaque pixels for actually visible printed, written, or displayed "
                        "characters. Do not infer text merely because this kind of object commonly has a label, and "
                        "do not guess or transcribe unreadable characters. Return JSON only: "
                        "{\"object_present\":boolean,\"label\":string|null,\"confidence\":number,"
                        "\"appearance_description\":string,\"detector_supported\":boolean,"
                        "\"detector_assessment\":string,\"visible_text\":boolean,"
                        "\"text_regions\":[string]}. text_regions is at most four short location descriptions "
                        "such as 'upper-left display' or 'shirt front'; it is empty when no text is visibly grounded. "
                        "When object_present is false, label must be null, confidence describes that rejection, and "
                        "appearance_description must say what the pixels actually show. "
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
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 180},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._background_gate:
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

    async def compare_masked_object_candidate(
        self,
        reference_png: bytes,
        current_png: bytes,
        reference: dict[str, object],
        detector_label: str,
        detector_confidence: float,
    ) -> dict[str, object] | None:
        """Adjudicate a CLIP proposal from two pixel-grounded object masks."""

        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "The first image is retained evidence for an existing object profile. The second image is a new "
                        "segmented detection. Transparent pixels are not evidence. Decide whether the second contains a "
                        "coherent physical object, and whether both images show the same physical object instance—not "
                        "merely two objects of the same category. Confirm same_instance only when visible shape, parts, "
                        "markings, wear, text, and other distinctive evidence agree and no visible conflict exists; an "
                        "embedding similarity is only a proposal. If viewpoint or occlusion prevents instance-level "
                        "adjudication, return false. Describe the current object's visible appearance and correct both "
                        "the detector and retained label when pixels require it. Return JSON only: "
                        "{\"object_present\":boolean,\"same_instance\":boolean,\"confidence\":number,"
                        "\"label\":string|null,\"appearance_description\":string,\"detector_supported\":boolean,"
                        "\"analysis\":string,\"visible_correspondences\":[string],\"visible_conflicts\":[string],"
                        "\"visible_text\":boolean,\"text_regions\":[string]}.\n"
                        f"Retained profile metadata (not pixel truth): {json.dumps(reference, ensure_ascii=False)[:1000]}\n"
                        f"Current detector proposal: {detector_label!r} at {detector_confidence:.2f}."
                    ),
                    "images": [
                        base64.b64encode(reference_png).decode("ascii"),
                        base64.b64encode(current_png).decode("ascii"),
                    ],
                }
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 240},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._background_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.vision_base_url).rstrip('/')}/api/chat",
                    json=payload,
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(
                            f"Ornith object comparison HTTP {response.status}: {detail}"
                        )
                    result = await response.json()
        try:
            content = result["message"]["content"]
        except (KeyError, TypeError) as error:
            raise RuntimeError("Ornith object comparison returned an invalid completion") from error
        return self.parse_object_comparison(content)

    async def classify_masked_object(
        self, image_png: bytes, detector_label: str, detector_confidence: float
    ) -> tuple[str, float] | None:
        """Compatibility classification view over the richer visual analysis."""
        analysis = await self.classify_masked_object_analysis(
            image_png, detector_label, detector_confidence
        )
        if analysis is None:
            return None
        if analysis.get("object_present") is not True or not analysis.get("label"):
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
        async with self._background_gate:
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

    async def answer_visual_question_analysis(
        self,
        frames: list[tuple[str, bytes, str]],
        utterance: str,
        scene: str,
    ) -> dict[str, object] | None:
        """Answer from ASR-boundary frames, preserving grounding metadata."""

        if not frames:
            return None
        visual_payload, tile_ledger = await asyncio.to_thread(
            self._visual_contact_sheet, frames
        )
        frame_ledger = [
            {
                "camera_id": camera_id,
                "captured_at": captured_at,
                "contact_sheet_tile": tile_ledger[index],
            }
            for index, (camera_id, _image, captured_at) in enumerate(frames)
        ]
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "You are the visual perception path for an embodied companion. "
                        "Answer the speaker's question using only visible pixels in the supplied labeled camera "
                        "contact sheet, whose tiles were all "
                        "frozen at the accepted utterance boundary. Reconcile cameras when they overlap and cite only "
                        "camera IDs whose pixels support the answer. "
                        "Be direct and conversational in one short sentence. If an item is partly occluded, "
                        "say what it most likely is and express uncertainty. If the pixels do not support an "
                        "answer, say what must be moved into view. Do not identify people or infer sensitive "
                        "traits. Do not let detector labels override pixels. Return JSON only: "
                        '{"answer":string,"grounded":boolean,"confidence":number,'
                        '"supporting_camera_ids":[string],"observations":[string],"uncertainty":string|null}.\n'
                        f"Speaker utterance: {utterance!r}\nFrame ledger: {json.dumps(frame_ledger)}\n"
                        f"Contemporaneous detector context (hypotheses, not truth): {scene[:1200]}"
                    ),
                    "images": [base64.b64encode(visual_payload).decode("ascii")],
                }
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 180},
            "keep_alive": "5m",
        }
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with self._conversational_gate:
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
            grounded = parsed.get("grounded")
            confidence = float(parsed.get("confidence", 0.0))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return None
        if not isinstance(answer, str) or not isinstance(grounded, bool):
            return None
        normalized = " ".join(answer.strip().split())
        if not normalized or len(normalized) > 320 or not 0 <= confidence <= 1:
            return None
        valid_camera_ids = {camera_id for camera_id, _image, _captured_at in frames}
        supporting = parsed.get("supporting_camera_ids")
        observations = parsed.get("observations")
        uncertainty = parsed.get("uncertainty")
        return {
            "answer": normalized,
            "grounded": grounded,
            "confidence": confidence,
            "supporting_camera_ids": [
                item for item in supporting[:8]
                if isinstance(item, str) and item in valid_camera_ids
            ] if isinstance(supporting, list) else [],
            "observations": [
                " ".join(item.split())[:240]
                for item in observations[:8]
                if isinstance(item, str) and item.strip()
            ] if isinstance(observations, list) else [],
            "uncertainty": (
                " ".join(uncertainty.split())[:300]
                if isinstance(uncertainty, str) and uncertainty.strip()
                else None
            ),
        }

    def _visual_contact_sheet(
        self, frames: list[tuple[str, bytes, str]]
    ) -> tuple[bytes, list[str]]:
        """Pack simultaneous views into one labeled, locally generated VLM image."""
        from PIL import Image, ImageDraw, ImageFont, ImageOps

        count = len(frames)
        columns = 1 if count == 1 else 2
        rows = math.ceil(count / columns)
        size = self.config.visual_contact_sheet_size
        tile_width = size // columns
        tile_height = size // rows
        sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), "black")
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.load_default()
        ledger: list[str] = []
        for index, (camera_id, encoded, _captured_at) in enumerate(frames):
            row, column = divmod(index, columns)
            image = Image.open(io.BytesIO(encoded)).convert("RGB")
            fitted = ImageOps.contain(
                image,
                (tile_width, tile_height),
                method=Image.Resampling.LANCZOS,
            )
            left = column * tile_width + (tile_width - fitted.width) // 2
            top = row * tile_height + (tile_height - fitted.height) // 2
            sheet.paste(fitted, (left, top))
            label = str(camera_id)[:80]
            label_box = draw.textbbox((0, 0), label, font=font)
            label_width = label_box[2] - label_box[0]
            label_height = label_box[3] - label_box[1]
            label_x = column * tile_width + 6
            label_y = row * tile_height + 6
            draw.rectangle(
                (
                    label_x - 3,
                    label_y - 3,
                    label_x + label_width + 3,
                    label_y + label_height + 3,
                ),
                fill="black",
            )
            draw.text((label_x, label_y), label, fill="white", font=font)
            ledger.append(f"row {row + 1}, column {column + 1}")
        output = io.BytesIO()
        sheet.save(output, format="JPEG", quality=82, optimize=True)
        return output.getvalue(), ledger

    async def answer_visual_question(
        self, image_jpeg: bytes, utterance: str, scene: str
    ) -> str | None:
        """Compatibility view for callers that provide one current frame."""

        analysis = await self.answer_visual_question_analysis(
            [("camera", image_jpeg, datetime.now(timezone.utc).isoformat())],
            utterance,
            scene,
        )
        return str(analysis["answer"]) if analysis is not None else None

    @staticmethod
    def parse_object_classification(content: object) -> tuple[str, float] | None:
        analysis = OmniusClient.parse_object_analysis(content)
        if analysis is None:
            return None
        if analysis.get("object_present", True) is not True or not analysis.get("label"):
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
        object_present = parsed.get("object_present", isinstance(label, str))
        if (
            not isinstance(object_present, bool)
            or not isinstance(confidence, (int, float))
            or (object_present and not isinstance(label, str))
            or (not object_present and label is not None)
        ):
            return None
        normalized = " ".join(label.strip().split()) if isinstance(label, str) else ""
        if (object_present and (not normalized or len(normalized) > 64)) or not 0 <= float(confidence) <= 1:
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
        result = {
            "label": normalized if object_present else None,
            "confidence": float(confidence),
            "visible_text": visible_text,
            "text_regions": normalized_regions,
        }
        if not any(
            key in parsed
            for key in (
                "object_present",
                "appearance_description",
                "detector_supported",
                "detector_assessment",
            )
        ):
            return result
        return {
            **result,
            "object_present": object_present,
            "appearance_description": " ".join(
                str(
                    parsed.get("appearance_description")
                    or normalized
                    or "segmented pixels did not ground a coherent object"
                ).split()
            )[:600],
            "detector_supported": bool(parsed.get("detector_supported", True)),
            "detector_assessment": " ".join(
                str(parsed.get("detector_assessment") or "").split()
            )[:400],
        }

    @staticmethod
    def parse_object_comparison(content: object) -> dict[str, object] | None:
        if not isinstance(content, str):
            return None
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return None
        object_present = parsed.get("object_present")
        same_instance = parsed.get("same_instance")
        confidence = parsed.get("confidence")
        label = parsed.get("label")
        description = parsed.get("appearance_description")
        analysis = parsed.get("analysis")
        if (
            not isinstance(object_present, bool)
            or not isinstance(same_instance, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= float(confidence) <= 1
            or (object_present and not isinstance(label, str))
            or (not object_present and label is not None)
            or not isinstance(description, str)
            or not isinstance(analysis, str)
        ):
            return None
        normalized_label = " ".join(label.split()) if isinstance(label, str) else None
        if object_present and (not normalized_label or len(normalized_label) > 64):
            return None
        result = {
            "object_present": object_present,
            "same_instance": same_instance if object_present else False,
            "confidence": float(confidence),
            "label": normalized_label,
            "appearance_description": " ".join(description.split())[:600],
            "detector_supported": bool(parsed.get("detector_supported", False)),
            "analysis": " ".join(analysis.split())[:600],
            "visible_correspondences": [
                " ".join(item.split())[:180]
                for item in parsed.get("visible_correspondences", [])[:8]
                if isinstance(item, str) and item.strip()
            ] if isinstance(parsed.get("visible_correspondences"), list) else [],
            "visible_conflicts": [
                " ".join(item.split())[:180]
                for item in parsed.get("visible_conflicts", [])[:8]
                if isinstance(item, str) and item.strip()
            ] if isinstance(parsed.get("visible_conflicts"), list) else [],
            "visible_text": bool(parsed.get("visible_text", False)),
            "text_regions": [
                " ".join(item.split())[:80]
                for item in parsed.get("text_regions", [])[:4]
                if isinstance(item, str) and item.strip()
            ] if isinstance(parsed.get("text_regions"), list) else [],
        }
        if not result["appearance_description"] or not result["analysis"]:
            return None
        return result

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

    async def _structured_chat(self, prompt: str, *, max_tokens: int = 128) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        bounded_tokens = max(64, min(int(max_tokens), 1024))
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
            "max_tokens": bounded_tokens,
            "realtime_options": {
                "max_history_messages": 4,
                "max_tokens": bounded_tokens,
            },
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

    async def _narrative_structured_chat(
        self, prompt: str, *, max_tokens: int
    ) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        bounded_max_tokens = max(256, min(int(max_tokens), 4096))
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
            "temperature": 0.2,
            "max_tokens": bounded_max_tokens,
            "tools": False,
            "think": False,
            "realtime": False,
            "realtime_options": {
                "max_history_messages": 2,
                "max_tokens": bounded_max_tokens,
            },
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
                        raise RuntimeError(
                            f"Omnius narrative chat HTTP {response.status}: {detail}"
                        )
                    result = await response.json()
        try:
            reply = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius narrative chat returned an invalid completion") from error
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Omnius narrative chat returned an empty completion")
        return reply.strip()

    def update_system_prompt(self, prompt: str) -> None:
        """Replace the dynamic system prompt used by _chat()."""
        if prompt and prompt.strip():
            self._system_prompt = prompt.strip()

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    async def _chat(
        self,
        prompt: str = "",
        *,
        remember: bool = True,
        include_memory: bool = True,
        messages: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
    ) -> str:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        if messages is not None:
            chat_messages = list(messages)
        else:
            chat_messages = [
                {"role": "system", "content": system_prompt or self._system_prompt},
                *(self._conversation if include_memory else []),
                {"role": "user", "content": prompt},
            ]
        payload = {
            "model": self.config.model,
            "messages": chat_messages,
            "stream": False,
            "max_tokens": 80,
            "temperature": 0.6,
            "tools": False,
            "think": self.config.reasoning_enabled,
            "realtime": True,
            "realtime_options": {"max_history_messages": 8, "max_tokens": 80},
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
            last_user = chat_messages[-1] if chat_messages else None
            if last_user and last_user.get("role") == "user":
                self._conversation.append(last_user)
            self._conversation.append({"role": "assistant", "content": response})
            self._conversation = self._conversation[-8:]
        return response
