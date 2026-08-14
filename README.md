# Egg Companion

Real-hardware companion runtime for a Jetson AGX with camera array, ReSpeaker direction-of-arrival microphone, speaker, and Omnius REST services. It intentionally has no simulation path. Audit failures remain visible, while the runtime gracefully degrades and independently retries failed camera, vision, audio, memory, and Omnius components so healthy capabilities stay live. Omnius daemon, cognition, voice, catalog, and audio readiness are tracked independently and stale failures clear automatically after recovery; the cognition monitor uses Omnius' lightweight `/health/ready` probe rather than consuming a synthetic chat inference. This Egg currently reports an **AGX Orin / JetPack R36.3**, rather than Xavier.

## What it does

- Captures directly from V4L2 or RTSP camera sources.
- Runs open-vocabulary YOLOE instance segmentation, pose estimation, and coarse actions (`standing`, `seated`, `waving`).
- Keeps raw MJPEG camera streams independent from asynchronous SVG mask/label overlays.
- Runs CLIP scene classification, anonymous person recall, masked-object recall, and sparse Ornith Vision correction.
- Reuses one person entity when adjacent same-camera instance masks strongly overlap, then records a local Ornith visual-continuity and displacement audit from the two masked crops.
- Persists source-grounded entities, episodes, evidence, claims, revisions, graph edges, and embeddings in local SQLite WAL storage.
- Uses prediction residuals, habituation, communicative action, and deterministic interruption policy rather than frame-count novelty.
- Captures the ReSpeaker XVF3000's processed AEC/beamformed ASR channel with adaptive WebRTC-VAD turn boundaries, native DSP VAD/DoA/AEC/AGC/RT60 telemetry, listen/think/speak LED states, and revisioned semantic barge-in with tail-only WAV resume.
- Runs a JetPack-matched CUDA dual-Whisper service: `tiny.en` admits grounded speech, then `base.en` verifies and supplies the transcript; silence and known Whisper outro hallucinations are rejected before conversation ingress.
- Runs grounded Omnius YAMNet/AudioSet scene classification asynchronously, links sound events to simultaneously visible people/objects, and feeds recent high-confidence audio context into later turns without delaying ASR.
- Reasons through Omnius `/v1/chat`, publishes only responses owned by the latest finalized heard-audio revision, and emits Supertonic `F4` WAV audio.
- Audits Jetson GPU power state, V4L2 cameras, ReSpeaker input/output/DOA, model checkpoints, CUDA, memory integrity, Ornith availability, and Omnius voice/cognition contracts.

## Safety and privacy boundary

The companion maintains an on-device profile gallery from validated face crops and masked objects. Identity and object evidence, embeddings, aliases, confidence, and provenance remain under `data/`; continuous video and rejected audio are not retained. Set `identity.enabled: false` or `object_learning.enabled: false` to disable collection. The loopback dashboard supports inspect, correction, metadata-only export, and cascade deletion, and never serializes raw embedding blobs.

## Install on the Egg

The single launcher performs the Jetson-specific bootstrap, including the CUDA PyTorch build, vision checkpoints, pinned dual-Whisper Jetson container, Ornith model, ReSpeaker DSP route, GPU runtime-PM guard, and bounded Ollama service configuration.

```bash
./egg bootstrap
```

`config/egg.yaml` discovers every V4L2 camera not already listed, rotates all corrected sources `90°` before inference, uses ReSpeaker USB `2886:0018`, Omnius `1.0.608+` on port `11435`, `omnius-qwen35-9b:latest` for cognition, `robit/ornith-vision:9b` for sparse masked-object teaching, and Supertonic voice `F4`.

## Jetson/ARM64 voice stack: what made realtime ASR work

This installation is `aarch64`, Jetson Linux `R36.3` (JetPack 6 / CUDA 12.2), on an AGX Orin. The decisive constraint is that CUDA libraries on Jetson are not interchangeable with generic x86 or PyPI CUDA wheels. A package can import successfully while still being ABI-incompatible with the JetPack driver, or an innocent dependency resolution can silently replace NVIDIA's working Torch.

The bootstrap therefore follows these rules:

1. It creates the application venv with `--system-site-packages`, because JetPack supplies part of the native CUDA/media stack outside PyPI.
2. It directly verifies `torch.cuda.is_available()` and the real device name. A version string or package presence is not accepted as proof of GPU execution.
3. If CUDA Torch is absent, it extracts the matched Python 3.10 packages (`torch`, `torchgen`, `torchvision`, and their dist-info) from `dustynv/l4t-pytorch:r36.2.0`. The tested result here is Torch `2.2.0`, CUDA `12.2`, on Orin.
4. It installs the Egg and high-level vision packages with `pip --no-deps`. This is intentional: unconstrained installs of OpenCLIP, Ultralytics, NeMo, or Transformers can replace the Jetson Torch build with a generic wheel.
5. It runs Whisper in a separate NVIDIA-runtime container, `dustynv/whisper:r36.2.0`, with host networking, host IPC, a persistent model cache, and the dedicated REST port `11436`. Omnius remains on `11435` for chat/TTS, so a long language-model generation cannot hold the ASR ingress gate.
6. `tiny.en` is the low-latency admission pass; `base.en` is the verification/final-text pass. The service compares both outputs and uses no-speech probability, average log probability, compression/repetition checks, real RMS/VAD evidence, and explicit short-outro rejection. In particular, “thanks for watching” and its common Whisper variants cannot be admitted from silence.

The implementation lives in [scripts/bootstrap-jetson.sh](scripts/bootstrap-jetson.sh), [scripts/jetson_whisper_server.py](scripts/jetson_whisper_server.py), and [deploy/egg-whisper.service](deploy/egg-whisper.service). The dedicated service design was informed by the working [EGG dual-Whisper voice pipeline](https://github.com/robit-man/EGG/tree/main/voice) and the deployment lessons in [whisper-stt-jetson](https://github.com/muttleydosomething/whisper-stt-jetson); the local code remains the executable source of truth.

Do not let Omnius auto-install generic `nemo_toolkit[asr]` on this host. An optional NeMo import failure previously caused pip to replace the NVIDIA CUDA Torch build. [scripts/repair_omnius_asr_runtime.py](scripts/repair_omnius_asr_runtime.py) makes the managed runtime use its Transformers fallback, fixes the generated Python 3.10 f-string, repairs managed-script discovery, and preserves the requested ASR language through the HTTP/CLI boundary.

The ReSpeaker USB 4-Mic Array v2.0 exposes six channels. Channel `0` is the XVF3000's processed AEC/beamformed stream and is the correct ASR source; raw microphone channels are useful for diagnostics but materially worse for room speech. Egg applies the hardware DSP route, reads native VAD/DoA/AEC/AGC/RT60 state, amplifies the quiet processed stream before WebRTC VAD, and normalizes admitted WAV audio toward `asr_target_rms` with a bounded maximum gain. Current working values are in `config/egg.yaml`; changing desktop “default input” does not change the explicitly resolved physical array.

Useful ground-truth checks after any dependency or JetPack change are:

```bash
uname -m
head -n 1 /etc/nv_tegra_release
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
curl -fsS http://127.0.0.1:11436/health
./egg audit
```

If Torch loses CUDA after a package install, stop and restore the JetPack-matched build; do not debug ASR thresholds on a CPU/fallback runtime. If ASR is healthy but quiet, inspect the ReSpeaker processed-channel selection, DSP route, pre-VAD gain, admitted WAV RMS, and target-RMS normalization in that order.

## Audio comprehension and conversation provenance

Transcription and sound understanding are deliberately separate. ASR stays on the fast CUDA Whisper service. A queue of size one coalesces admitted audio windows for Omnius' persistent `/v1/audio/classify` service, whose YAMNet model emits scores across the 521-class AudioSet taxonomy. Older Omnius releases fall back to `audio_analyze/classify`. Only labels above `audio_comprehension.minimum_confidence` become `sound_event` graph entities. They are explicitly related with `heard_with` edges to visible people and objects, and recent grounded labels become bounded scene context for subsequent replies. Dedicated classifier readiness is exposed as the `omnius-audio` dashboard check rather than being conflated with daemon, voice, or cognition health.

Omnius also exposes an `audio_analyze/comprehend` role pipeline. On this installation its AV sidecar roles currently report `mock semantic scaffold`; those event labels are never persisted or shown as perception. Egg admits only the independent numeric YAMNet classifier and locally measured WAV facts until the sidecar reports live roles. Classification runs behind ASR, so a cold TensorFlow/model load cannot add latency to the current spoken response.

The Jetson hardware matrix, current Omnius failure analysis, persistent-worker design, REST schema, and acceptance suite for the Omnius implementer are in [docs/OMNIUS_JETSON_AUDIO_COMPREHENSION_HANDOFF.md](docs/OMNIUS_JETSON_AUDIO_COMPREHENSION_HANDOFF.md).

Every admitted utterance now supplies a durable context ID to its audio evidence, visual/web tool invocations, retrieval influences, user corrections, preferred-name bindings, learned-object labels, audio classifications, and agent action evidence. The Voice page renders those as live tags on the same historical turn—for example `fresh vision ✓`, `memory recall ×4`, `remembered name: Troy`, `label updated: amber mug`, or `Speech 67%`. Late asynchronous evidence updates the existing message in place and survives daemon restarts; it does not reset the page or create a second fake heard turn.

## Mask-aware OCR and nested visual content

OCR admission is visual rather than category-gated. A periodic single-pass sparse frame scan proposes actual text boxes, projects their coordinates back into camera space, and assigns each box to the smallest containing/overlapping instance mask. Only those grounded crops receive the more expensive multi-variant advanced pass. This naturally covers screens, books, signs, packaging, shirts, people, held media, and previously unseen kinds of objects without maintaining a reward-hack label list or OCRing every person crop. Novel stable object masks are also OCRed in parallel with Ornith analysis; the VLM reports whether and where text is visibly grounded, but its hint is provenance rather than the sole admission gate. Segmentation polygons perspective-rectify oblique masked crops. Low-confidence fragments are discarded rather than promoted as memory. The local multi-pass Tesseract path remains available while Omnius or Ollama is cold.

Each unrecognized text-bearing mask receives a stable camera-local observation ID based on temporal mask overlap; all monitors with the same detector label are no longer grouped into one category node. Accepted OCR creates `object → contains_text → content → contains_fragment` relationships, retains the rectified crop as clickable evidence, and stores OCR regions, confidence, engine, source mask polygon, and bounding box in provenance. Egg understands Omnius 1.0.629's canonical `{args:{image}} → result.data` advanced-OCR contract, but remote refinement remains opt-in until its managed Jetson OCR dependencies are ready; see [the live defect report](docs/OMNIUS_NEMOTRON_ACTIVATION_ISSUE.md).

## Evidence-grounded meta-graph and evolving cognitive documents

Quiet-period replay now projects a graph over the graph. Canonical entity pairs recurring across the configured number of distinct temporal encounter windows create stable `abstraction` nodes plus `supports_pattern` and `recurrently_associated_with` synapses. Transient appearance tracks and adjacent-frame repetitions cannot satisfy this rule. Every abstraction carries representative source episode IDs, temporal support IDs, raw observation count, support-period count, confidence, and `inferred_noncausal` epistemic status. Replaying the same evidence updates a deterministic edge instead of inflating its confirmations; abstractions excluded by newer support rules are superseded with their derived edges retracted. Source entities also receive bounded derived summaries.

Four revisioned `cognitive_document` nodes compound over this meta-graph: **World model**, **My story**, **Communication strategy**, and **Reflective working set**. `agent:egg → maintains/guides_communication → document`, `reflection → informs_working_set → document`, and `abstraction → informs_world_model → document` edges keep their provenance visible and clickable. “My story” uses first-person language but explicitly remains a source-grounded, revisable account rather than a claim of subjective experience. The working set is inspectable reflective state, not private chain-of-thought. A bounded slice feeds normal dialogue, fresh visual questions, and proactive visual communication; subsequent spoken, suppressed, corrected, interrupted, modality, and tool outcomes become new evidence for later revisions.

Identity dreams also drive chronological consolidation, even when a pass performs zero identity merges. A lightweight startup replay independently discovers every retained local-calendar day and backdates never-narrated history oldest-first in bounded passes, always refreshing the latest observed day; it repeats until the reported backlog reaches zero. Identity merges additionally rebuild every affected canonical/alias day. Frame-level evidence is coalesced into ordered, configurable time windows carrying people, objects, OCR content, sound events, admitted speech, agent replies, source modalities, episode IDs, and artifact IDs. Each day becomes a revisioned `daily_narrative` node with an evidence-grounded abstract synopsis and explicit `appears_in_day`, `observed_in_day`, `read_in_day`, `heard_in_day`, `precedes_day`, `replays_day`, and `contributes_to_story` synapses. Recurrent within-day co-occurrence is described only as non-causal association. The Dreams audit reports discovered history, chapters backdated, backlog remaining, and the resulting **My story** revision. The in-page Narrative workspace presents days newest-first as a vertical timeline; expanding a day reveals its latest-to-oldest encounter periods, nested episode summaries, people/objects/content/sound tags, and retained image/audio/OCR artifacts. Selecting an orange daily-story node in the graph exposes the same grounded chapter from its associative context.

The architecture draws on four complementary research patterns: observation/retrieval/reflection feedback from [Generative Agents](https://arxiv.org/abs/2304.03442); structural relational generalization from the [Tolman–Eichenbaum Machine](https://doi.org/10.1016/j.cell.2020.10.024); episodic knowledge organized under a working self from the [Self-Memory System](https://doi.org/10.1037/0033-295X.107.2.261); and partner-sensitive terminology from [Lexical Entrainment for Conversational Systems](https://aclanthology.org/2023.findings-emnlp.22/). These are design inspirations, not a claim that Egg implements a biological mind.

## Temporal person continuity

Short-term identity follows same-camera instance-mask geometry before detector numbering. When a person mask in a later frame exceeds either `track_mask_iou_threshold` or `track_mask_containment_threshold` within `track_mask_max_gap_seconds`, the observation reuses the existing track and, if enrolled, its canonical person profile—even when its bounding box has jumped far enough to receive a new detector label. A track can be assigned only once per frame, so two simultaneous people cannot be collapsed through the same prior mask. Transient track IDs are globally unique rather than restart-local counters, preventing a new runtime from attaching evidence to an unrelated historical observation.

Strong mask continuity queues two transparent masked crops for local `robit/ornith-vision:9b` comparison when mask geometry rescues an otherwise dislocated detection or a transient track becomes an enrolled person. Steady frames on an already-associated track do not schedule redundant VLM work. Ornith reports bounded JSON containing the visible continuity cues, confidence, and a displacement narrative; geometry remains the realtime authority so a slow or unavailable VLM cannot fragment tracking. The comparison contact sheet and analysis become evidence on the single canonical entity, transient-to-canonical aliases are projected automatically into memory, and the People page exposes the recent merge ledger. The queue and per-track cooldown are bounded so identity auditing cannot block camera inference or build an unbounded VLM backlog.

## Offline identity dreams

Identity dreams use pinned AdaFace IR18/WebFace4M and InsightFace MobileFaceNet
checkpoints locally and offline. Bootstrap them explicitly on a new installation; the randomized idle
scheduler then consolidates profiles and projects their complete evidence history
without dashboard interaction. Every completed pass then performs the dated
chronological replay and story/meta-graph revision described above, whether or
not a new alias was created. The Dreams page is an audit/status view with a
manual trigger only as an optional override:

```bash
.venv/bin/python scripts/install_dream_identity_model.py
```

The model card requires users to follow the training dataset's license for their
deployment. Dream merges use quality-weighted AdaFace, SFace, and MobileFaceNet
templates, two-of-three model consensus, compatible names, and repeated
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

Omnius owns chat, voice warm-up, and TTS; `egg-whisper.service` owns ASR on port `11436`, and `OmniusClient` composes both voice catalogs for the dashboard. TTS WAV is delivered to the local speaker through `aplay`. A separate event bridge remains optional for external system integrations.

The local conversational floor and interruption invariants are documented in
`docs/VOICE_TURN_RUNTIME.md`. Voice state (`listening`, `audio_detected`,
`transcribing`, `processing`, `response_playing`, or `barge_pending`) and causal
playback identities are exposed in the dashboard telemetry snapshot.

The research rationale and implementation ledger are maintained in `docs/COGNITIVE_MEMORY_RESEARCH.md`, `docs/COGNITIVE_MEMORY_WORK_ORDERS.md`, and `docs/COGNITIVE_MEMORY_EXECUTION.md`.
