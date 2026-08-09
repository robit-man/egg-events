from __future__ import annotations

import base64
import asyncio
import json
import zlib

import os

import aiohttp

from egg_companion.config import OmniusConfig


class OmniusClient:
    def __init__(self, config: OmniusConfig) -> None:
        self.config = config
        self._conversation: list[dict[str, str]] = []
        self._model_gate = asyncio.Lock()
        self.last_transcription_metadata: dict[str, object] = {}

    def _headers(self) -> dict[str, str]:
        if not self.config.bearer_token_env:
            return {}
        token = os.getenv(self.config.bearer_token_env)
        if not token:
            raise RuntimeError(f"required token environment variable is unset: {self.config.bearer_token_env}")
        return {"Authorization": f"Bearer {token}"}

    async def health(self) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3)) as session:
            async with session.get(f"{str(self.config.base_url).rstrip('/')}/health", headers=self._headers()) as response:
                response.raise_for_status()

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
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
            async with session.get(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/state", headers=self._headers()
            ) as response:
                response.raise_for_status()
                payload = await response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Omnius voice state is not an object")
        return payload

    async def voice_catalog(self) -> dict[str, object]:
        timeout = aiohttp.ClientTimeout(total=10)
        base_url = str(self.config.base_url).rstrip("/")
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{base_url}/v1/voice/models", headers=self._headers()) as response:
                response.raise_for_status()
                tts = await response.json()
            async with session.get(f"{base_url}/v1/voice/asr-models", headers=self._headers()) as response:
                response.raise_for_status()
                asr = await response.json()
            async with session.get(f"{base_url}/v1/voice/supertonic-settings", headers=self._headers()) as response:
                if response.status == 404:
                    supertonic = {"supported": False, "settings": {}, "options": {"voices": []}}
                else:
                    response.raise_for_status()
                    supertonic = await response.json()
        return {"tts": tts, "asr": asr, "supertonic": supertonic}

    async def ensure_asr_model(self, model_id: str) -> None:
        catalog = await self.voice_catalog()
        asr = catalog.get("asr")
        if isinstance(asr, dict) and asr.get("current") == model_id:
            return
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/voice/asr-models/switch",
                json={"modelId": model_id},
                headers=self._headers(),
            ) as response:
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Omnius ASR model switch HTTP {response.status}: {detail}")

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

    async def conversation_reply(self, utterance: str, scene: str) -> str:
        return await self._chat(
            "Local speech, already verified as human speech by VAD:\n"
            f"{utterance!r}\n\nEmbodied context:\n{scene}\n\n"
            "Decide from the conversational history and context whether this is directed to Egg. "
            "If it is not directed to Egg or does not merit an audible interruption, reply exactly [[SILENT]]."
        )

    async def reason_about_utterance(
        self, utterance: str, context: str
    ) -> dict[str, object] | None:
        raw = await self._structured_chat(
            "Classify a VAD-verified utterance for interaction routing, without answering it.\n"
            f"Utterance: {utterance!r}\nContext: {context[:1200]}\n"
            "Return only JSON: {\"directed\":boolean,\"act\":\"question\"|\"correction\"|"
            "\"person_naming\"|\"object_naming\"|\"command\"|\"acknowledgement\"|"
            "\"conversation\",\"confidence\":number}."
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
        return {
            "directed": parsed["directed"],
            "act": parsed["act"],
            "confidence": float(parsed["confidence"]),
        }

    async def interpret_correction(
        self, utterance: str, candidate: dict[str, object]
    ) -> dict[str, str] | None:
        return await self.interpret_observation_feedback(utterance, candidate)

    async def interpret_person_naming(self, utterance: str) -> str | None:
        return await self.interpret_person_introduction(utterance)

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

    async def interpret_person_introduction(self, utterance: str) -> str | None:
        raw = await self._structured_chat(
            "Determine whether the speaker explicitly provides their own preferred name. "
            f"Utterance: {utterance!r}\n"
            "Return only JSON: {\"name\": string|null}. Do not infer names, extract names of third parties, "
            "or treat descriptions such as 'I am tired' as a name."
        )
        return self.parse_person_name(raw)

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

    async def transcribe(self, wav_audio: bytes) -> str | None:
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        headers = {**self._headers(), "Content-Type": "audio/wav"}
        async with self._model_gate:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    f"{str(self.config.base_url).rstrip('/')}/v1/voice/transcribe",
                    data=wav_audio,
                    headers=headers,
                ) as response:
                    if response.status >= 400:
                        detail = (await response.text())[:500]
                        raise RuntimeError(f"Omnius ASR HTTP {response.status}: {detail}")
                    payload = await response.json()
        text = payload.get("text")
        rejection_reason = self.transcription_rejection_reason(payload)
        if not isinstance(text, str) or not text.strip():
            rejection_reason = "empty transcript"
        self.last_transcription_metadata = {
            "duration": payload.get("duration"),
            "language": payload.get("language"),
            "segments": payload.get("segments") if isinstance(payload.get("segments"), list) else [],
            "accepted": rejection_reason is None,
            "rejection_reason": rejection_reason,
        }
        if rejection_reason is not None:
            return None
        return text.strip()

    @staticmethod
    def transcription_is_grounded(payload: dict[str, object]) -> bool:
        return OmniusClient.transcription_rejection_reason(payload) is None

    @staticmethod
    def transcription_rejection_reason(payload: dict[str, object]) -> str | None:
        segments = payload.get("segments")
        if not isinstance(segments, list) or not segments:
            return None
        scored = [segment for segment in segments if isinstance(segment, dict)]
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
            text = payload.get("text")
            if isinstance(text, str) and OmniusClient._text_compression_ratio(text) >= 2.4:
                return "repetitive transcript compression"
        return None

    @staticmethod
    def _text_compression_ratio(text: str) -> float:
        data = text.strip().encode("utf-8")
        if not data:
            return 0.0
        compressor = zlib.compressobj(level=9, wbits=-15)
        compressed = compressor.compress(data) + compressor.flush()
        return len(data) / max(len(compressed), 1)

    async def ocr_advanced(self, image_path: str) -> dict[str, object] | None:
        """Read text from a local image path via Omnius's multi-PSM tesseract + optional
        vision-refinement OCR pipeline. Returns None when the tool is unavailable (501)
        or the response is malformed, so callers can treat it as purely corroborating
        evidence alongside the Ornith VLM classification rather than a hard dependency."""
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                f"{str(self.config.base_url).rstrip('/')}/v1/ocr/advanced",
                json={"imagePath": image_path},
                headers=self._headers(),
            ) as response:
                if response.status == 501:
                    return None
                if response.status >= 400:
                    detail = (await response.text())[:500]
                    raise RuntimeError(f"Omnius OCR HTTP {response.status}: {detail}")
                result = await response.json()
        if not isinstance(result, dict) or not result.get("success"):
            return None
        text = result.get("ocrText")
        if not isinstance(text, str) or not text.strip():
            return None
        return {
            "text": text.strip()[:500],
            "vision_used": bool(result.get("visionUsed")),
        }

    async def classify_masked_object(self, image_png: bytes, detector_label: str, detector_confidence: float) -> tuple[str, float] | None:
        """Classify a real transparent-mask crop with the configured local Ornith VLM."""
        image_data = base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self.config.vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Classify only the opaque segmented object in this PNG; ignore transparent pixels and background. "
                        "Return JSON only: {\"label\": string|null, \"confidence\": number}. "
                        f"Detector candidate: {detector_label!r} at {detector_confidence:.2f}."
                    ),
                    "images": [image_data],
                },
            ],
            "stream": False,
            "format": "json",
            "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 64},
            "keep_alive": 0,
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
        return self.parse_object_classification(content)

    @staticmethod
    def parse_object_classification(content: object) -> tuple[str, float] | None:
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
        return normalized, float(confidence)

    async def audit_object_label(self, profile: dict[str, object]) -> dict[str, object] | None:
        """Cheap, text-only confidence audit of an already-labelled object.

        Uses the cognition model rather than the vision model: it triages which
        profiles are worth a real image-grounded VLM re-classification instead of
        overwriting a label itself. Returns None on any malformed response so the
        caller always falls back to the VLM path rather than trusting a silent pass.
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

    async def _chat(self, prompt: str) -> str:
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
                *self._conversation,
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "temperature": 0.6,
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
                        raise RuntimeError(f"Omnius chat HTTP {response.status}: {detail}")
                    result = await response.json()
        try:
            reply = result["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Omnius returned an invalid chat completion") from error
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("Omnius returned an empty chat completion")
        response = reply.strip()
        self._conversation.extend(({"role": "user", "content": prompt}, {"role": "assistant", "content": response}))
        self._conversation = self._conversation[-12:]
        return response
