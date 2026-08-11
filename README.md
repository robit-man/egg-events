# Egg Companion

Real-hardware companion runtime for a Jetson AGX with camera array, ReSpeaker direction-of-arrival microphone, speaker, and Omnius REST services. It intentionally has no simulation path. Audit failures remain visible, while the runtime gracefully degrades and independently retries failed camera, vision, audio, memory, and Omnius components so healthy capabilities stay live. This Egg currently reports an **AGX Orin / JetPack R36.3**, rather than Xavier.

## What it does

- Captures directly from V4L2 or RTSP camera sources.
- Runs open-vocabulary YOLOE instance segmentation, pose estimation, and coarse actions (`standing`, `seated`, `waving`).
- Keeps raw MJPEG camera streams independent from asynchronous SVG mask/label overlays.
- Runs CLIP scene classification, anonymous person recall, masked-object recall, and sparse Ornith Vision correction.
- Persists source-grounded entities, episodes, evidence, claims, revisions, graph edges, and embeddings in local SQLite WAL storage.
- Uses prediction residuals, habituation, communicative action, and deterministic interruption policy rather than frame-count novelty.
- Captures the ReSpeaker XVF3000's processed AEC/beamformed ASR channel with adaptive WebRTC-VAD turn boundaries, native DSP VAD/DoA/AEC/AGC/RT60 telemetry, listen/think/speak LED states, and revisioned semantic barge-in with tail-only WAV resume.
- Reasons through Omnius `/v1/chat`, publishes only responses owned by the latest finalized heard-audio revision, and emits Supertonic `F4` WAV audio.
- Audits Jetson GPU power state, V4L2 cameras, ReSpeaker input/output/DOA, model checkpoints, CUDA, memory integrity, Ornith availability, and Omnius voice/cognition contracts.

## Safety and privacy boundary

The companion maintains an on-device profile gallery from validated face crops and masked objects. Identity and object evidence, embeddings, aliases, confidence, and provenance remain under `data/`; continuous video and rejected audio are not retained. Set `identity.enabled: false` or `object_learning.enabled: false` to disable collection. The loopback dashboard supports inspect, correction, metadata-only export, and cascade deletion, and never serializes raw embedding blobs.

## Install on the Egg

The single launcher performs the Jetson-specific bootstrap, including the CUDA PyTorch build, vision checkpoints, CUDA CTranslate2 ASR runtime, Ornith model, ReSpeaker DSP route, GPU runtime-PM guard, and bounded Ollama service configuration.

```bash
./egg bootstrap
```

`config/egg.yaml` discovers every V4L2 camera not already listed, rotates all corrected sources `90°` before inference, uses ReSpeaker USB `2886:0018`, Omnius `1.0.608+` on port `11435`, `omnius-qwen35-9b:latest` for cognition, `robit/ornith-vision:9b` for sparse masked-object teaching, and Supertonic voice `F4`.

Identity dreams use the pinned AdaFace IR18/WebFace4M checkpoint locally and
offline. Bootstrap it explicitly on a new installation; the randomized idle
scheduler then consolidates profiles and projects their complete evidence history
without dashboard interaction. The Dreams page is an audit/status view with a
manual trigger only as an optional override:

```bash
.venv/bin/python scripts/install_dream_identity_model.py
```

The model card requires users to follow the training dataset's license for their
deployment. Dream merges use quality-weighted AdaFace and SFace templates,
reciprocal neighborhood/score-separation evidence, compatible names, and repeated
or spatially explicit distinct-person constraints. Source profiles and evidence
remain intact behind reversible aliases. The People page opens each canonical
person into a dated encounter timeline across every coalesced source profile.

## Audit and run

```bash
./egg
```

`./egg` is the single entry point. It bootstraps when required, applies the ReSpeaker DSP route, serves the dark dashboard at `http://127.0.0.1:8788`, and launches the companion while audit checks remain available as non-blocking diagnostics. One usable camera is sufficient; unavailable cameras are warnings, not global blockers. Bootstrap also enables `egg-companion.service`, which invokes this same entry point after boot.

```bash
./egg audit
./egg memory-audit
./egg memory verify
./egg memory migrate
./egg test
./egg serve --port 8788
./egg trace --url http://127.0.0.1:8788 --seconds 30
./egg evaluate --trace tests/fixtures/traces/baseline.json
```

The bootstrap installs `egg-gpu-pm-guard.service` before display-manager and Ollama. The guard accepts the kernel's normal `auto` policy as well as `on`; the audit always probes CUDA directly and only treats explicit PM error/unsupported states as failures. Ollama is limited to one loaded model, one parallel request, and a `4096` token context so ASR, cognition, and vision cannot exhaust unified memory.

## Integration points

- Camera sources: `egg_companion/adapters/camera.py`
- YOLOE + SAM + CLIP + face embeddings: `egg_companion/adapters/vision.py`
- ReSpeaker DOA JSON-lines serial protocol: `egg_companion/adapters/audio.py`
- Omnius chat/ASR/TTS REST API: `egg_companion/adapters/omnius.py`
- Optional external event bridge: `egg_companion/adapters/system_service.py`

Omnius chat, ASR, voice warm-up, and TTS are native endpoints in `OmniusClient`; TTS WAV is delivered to the local speaker through `aplay`. A separate event bridge remains optional for external system integrations.

The local conversational floor and interruption invariants are documented in
`docs/VOICE_TURN_RUNTIME.md`. Voice state (`listening`, `audio_detected`,
`transcribing`, `processing`, `response_playing`, or `barge_pending`) and causal
playback identities are exposed in the dashboard telemetry snapshot.

The research rationale and implementation ledger are maintained in `docs/COGNITIVE_MEMORY_RESEARCH.md`, `docs/COGNITIVE_MEMORY_WORK_ORDERS.md`, and `docs/COGNITIVE_MEMORY_EXECUTION.md`.
