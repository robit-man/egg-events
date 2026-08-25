# Egg Companion

Real-hardware companion runtime for a Jetson AGX with camera array, ReSpeaker direction-of-arrival microphone, speaker, and Omnius REST services. It intentionally has no simulation path. Audit failures remain visible, while the runtime gracefully degrades and independently retries failed camera, vision, audio, memory, and Omnius components so healthy capabilities stay live. Omnius daemon, cognition, voice, catalog, and audio readiness are tracked independently and stale failures clear automatically after recovery; the cognition monitor uses Omnius' lightweight `/health/ready` probe rather than consuming a synthetic chat inference. This Egg currently reports an **AGX Orin / JetPack R36.3**, rather than Xavier.

## What it does

- Captures directly from V4L2 or RTSP camera sources.
- Runs open-vocabulary YOLOE instance segmentation, pose estimation, and coarse actions (`standing`, `seated`, `waving`).
- Keeps raw MJPEG camera streams independent from asynchronous SVG mask/label overlays.
- Runs CLIP scene proposals and local Ornith pixel adjudication; embeddings never establish durable object identity by themselves.
- Reuses one person entity when adjacent same-camera instance masks strongly overlap, then records a local Ornith visual-continuity and displacement audit from the two masked crops.
- Persists source-grounded entities, episodes, evidence, claims, revisions, graph edges, and embeddings in local SQLite WAL storage.
- Uses prediction residuals, habituation, communicative action, and deterministic interruption policy rather than frame-count novelty.
- Falls off vision/OCR inference frequency toward an idle floor once the cameras and microphone go quiet, and snaps back to full rate the instant novelty, presence, or speech returns (`activity` config).
- Captures the ReSpeaker XVF3000's processed AEC/beamformed ASR channel with adaptive WebRTC-VAD turn boundaries, native DSP VAD/DoA/AEC/AGC/RT60 telemetry, listen/think/speak LED states, and revisioned semantic barge-in with tail-only WAV resume.
- Runs a JetPack-matched CUDA dual-Whisper service: `tiny.en` admits grounded speech, then `base.en` verifies and supplies the transcript; acoustically unsupported and weakly disagreeing decodes are rejected before conversation ingress.
- Runs grounded Omnius YAMNet/AudioSet scene classification asynchronously, links sound events to simultaneously visible people/objects, and feeds recent high-confidence audio context into later turns without delaying ASR.
- Reasons through Omnius `/v1/chat`, publishes only responses owned by the latest finalized heard-audio revision, and emits Supertonic `F4` WAV audio.
- Freezes multi-camera evidence at each utterance boundary, lets the realtime model route visual requests to Ornith, and learns a versioned interaction strategy from revisable sentiment/behavior and response feedback.
- Audits Jetson GPU power state, V4L2 cameras, ReSpeaker input/output/DOA, model checkpoints, CUDA, memory integrity, Ornith availability, and Omnius voice/cognition contracts.
- Maintains a typed, bitemporal operational world model (entities, properties, relations, conflicts, assertion history) queried directly into LLM context, with policy-gated `speak`/`focus_camera`/`inspect_entity` actions and real-time gaze detection.
- Fuses on-demand monocular depth from all four panoramic-array cameras into one shared voxel occupancy grid using Bayesian log-odds mapping, rendered as a real-color, orbitable 3D reconstruction on the dashboard.

## Novelty/activity-driven perception frequency

`egg_companion/core/activity.py`'s `ActivityGovernor` tracks one system-wide alertness scale from the same signals already flowing through the runtime: per-tick novelty and detection presence from `CognitiveArchitecture.perceive` (`_attend`), and ReSpeaker VAD speech detection (`_stream_waveform`). While the room holds novelty, a visible detection, or speech, alertness stays at `1.0` and vision analysis/pose/semantics (`vision.analysis_fps`/`pose_fps`/`semantic_fps`) and full-frame OCR (`ocr.full_frame_interval_seconds`) run at their configured rate. After `activity.decay_seconds` of a genuinely empty, silent scene, alertness decays exponentially toward `activity.idle_floor`, throttling that same inference down proportionally — the quiet-room analogue of reduced visual/auditory vigilance. Any new novelty, presence, or speech resets alertness to full immediately; there is no cooldown on recovery. Set `activity.enabled: false` to keep every rate at its static configured value. Current alertness is visible on the dashboard telemetry snapshot under `activity`.

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
6. `tiny.en` is the low-latency admission pass; `base.en` is the verification/final-text pass. The service compares both outputs and uses no-speech probability, average log probability, compression/repetition checks, real RMS/VAD evidence, and dual-decode agreement. It does not blacklist semantic phrases: silence rejection is derived from acoustic and model evidence, so legitimately spoken text is not censored.

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

Every admitted utterance now supplies a durable context ID to its audio evidence, visual/web tool invocations, retrieval influences, user corrections, preferred-name bindings, learned-object labels, audio classifications, and agent action evidence. The Voice page renders those as live tags on the same historical turn—for example `ASR-boundary vision ✓`, `memory recall ×4`, `remembered name: Troy`, `label updated: amber mug`, or `Speech 67%`. Late asynchronous evidence updates the existing message in place and survives daemon restarts; it does not reset the page or create a second fake heard turn.

## ASR-boundary visual conversation

At the acoustic end of every admitted utterance—before transcription enters its queue—Egg freezes every configured camera frame that is within the transport freshness bound. The snapshot stores each camera ID, its actual capture time, the utterance boundary, and the contemporaneous detector observation. No transcript phrase, detector category, person-box size, or hand-authored camera score selects a frame. The realtime dialogue model owns the `vision` tool decision. When selected, all frozen frames are sent together to local Ornith; its bounded result names supporting camera IDs, pixel-grounded observations, confidence, and uncertainty. The answer, exact JPEG evidence, frame ledger, model ID, and conversation context ID are retained together. A later camera frame can never silently replace the one associated with the question.

Human speech preempts in-flight background object review, social reflection, and duplicate adjudication before those jobs can occupy Ornith ahead of a conversational visual request. ASR remains on its independent CUDA service and never waits for this cancellation. General dialogue, visual answers, TTS, and barge-in continue to use revision ownership, so a newer heard turn invalidates an obsolete visual reply before playback.

## Pixel-adjudicated object memory

YOLO labels and CLIP similarities are now explicitly proposals, not durable object identities. Every stable segmented non-person detection admitted to the bounded adjudication queue is shown to Ornith as opaque pixels. The VLA first decides whether the mask contains a coherent physical object rather than a fragment, body part, texture, duplicate mask, or detector hallucination; it then returns the supported noun label, a detailed visible appearance description, detector agreement, confidence, and visible-text regions. Rejected proposals remain inspectable evidence but create no object entity.

A CLIP hit must pass a two-image Ornith comparison between retained profile evidence and the current mask. The model is asked for the *same physical instance*, not merely the same category, and must report visible correspondences and conflicts. Only a confirmed comparison advances `last_seen`, updates the running visual embedding and thumbnail, and publishes an `object_id` back into detections. An unconfirmed but coherent current object becomes a distinct profile even when its noun label matches another profile. Thus two mugs no longer collapse because both are called “mug.”

Quiet review always reopens due legacy profiles against retained pixels; a text-only history audit cannot clear a visual claim. Unsupported historical labels are retracted in the knowledge graph while their evidence and revisions remain. High-similarity historical profile pairs are likewise only merge proposals: Ornith must confirm distinctive instance-level agreement before embeddings, encounter evidence, episodes, and graph edges coalesce under a reversible `same_object_as` alias. Appearance descriptions become evidence-backed `has_appearance` claims. This fail-closed design is informed by detector hallucination findings in [visual part verification](https://arxiv.org/abs/2106.02523), VLM object-hallucination measurement in [POPE](https://arxiv.org/abs/2305.10355), open-set grounding in [Grounding DINO](https://arxiv.org/abs/2303.05499), and robust self-supervised visual features in [DINOv2](https://arxiv.org/abs/2304.07193). These works support proposal-plus-verification; none justifies treating a single model score as certainty.

## Adaptive social cognition

After a completed human turn, an interruptible background model pass combines its transcript, measured VAD/acoustic facts, available audio-semantic output, the agent response, publication outcome, ordered conversation, visible grounded entities, the current interaction strategy, and prior social profiles. It produces a timestamped momentary affect interpretation, observable communicative behavior, relationship-context update, response feedback, and—only when supported—a revision to Egg's communication method. For each unambiguously grounded visible person it can also revise a longitudinal `social_profile` containing a sentiment trajectory, observed communication patterns, explicitly evidenced interaction preferences, and unresolved uncertainty. Every version remains linked to its source turn. These profiles describe Egg's interaction evidence—not a person's essence—and are never converted into fixed personality, diagnostic, demographic, or protected-trait claims.

An accepted strategy revision updates the stable `interaction-strategy:current` graph node while its source turn and prior evidence remain versioned. That directive and any relevant person's revisable social profile are injected into later conversational context alongside the model-authored observation policy, so observed response outcomes change future timing, tone, directness, and question choice. Identity introductions, preferred-name acknowledgements, and proactive curiosity are authored at the moment of use from grounded encounter history, current dialogue, visible people, and this evolving strategy; there is no fixed “who are you?”, “nice to meet you”, or name-prefixed curiosity template. Vision, web-search, naming, and directed-speech semantics use the model's typed route rather than keyword fallbacks. Operational cooldowns, one-question floor ownership, visible-person requirements, provenance, and queue capacity remain code-level safety bounds, while the semantic choice to ask and the actual language belong to the model.

## Mask-aware OCR and nested visual content

OCR admission is visual rather than category-gated. A periodic single-pass sparse frame scan proposes actual text boxes, projects their coordinates back into camera space, and assigns each box to the smallest containing/overlapping instance mask. Only those grounded crops receive the more expensive multi-variant advanced pass. This naturally covers screens, books, signs, packaging, shirts, people, held media, and previously unseen kinds of objects without maintaining a reward-hack label list or OCRing every person crop. Novel stable object masks are also OCRed in parallel with Ornith analysis; the VLM reports whether and where text is visibly grounded, but its hint is provenance rather than the sole admission gate. Segmentation polygons perspective-rectify oblique masked crops. Low-confidence fragments are discarded rather than promoted as memory. The local multi-pass Tesseract path remains available while Omnius or Ollama is cold.

Each unrecognized text-bearing mask receives a stable camera-local observation ID based on temporal mask overlap; all monitors with the same detector label are no longer grouped into one category node. Accepted OCR creates `object → contains_text → content → contains_fragment` relationships, retains the rectified crop as clickable evidence, and stores OCR regions, confidence, engine, source mask polygon, and bounding box in provenance. Egg understands Omnius 1.0.629's canonical `{args:{image}} → result.data` advanced-OCR contract, but remote refinement remains opt-in until its managed Jetson OCR dependencies are ready; see [the live defect report](docs/OMNIUS_NEMOTRON_ACTIVATION_ISSUE.md).

## Typed operational world model and policy-gated actions

Alongside the associative knowledge graph, `egg_companion/world/` maintains a typed, bitemporal operational world model: append-only historical assertions (`world_assertions`, `relation_assertions`) are reconciled into a materialized current-state projection (`current_property_state`, `current_relation_state`) by `Reconciler`/`WorldStateStore`, with an explicit ontology (`OntologyRegistry`) governing valid property/relation types and a calibration/transform graph for cross-modal value normalization. `WorldQuery` surfaces conflicts, entity ranking signals, and scoped summaries; `CognitiveContext` injects a reconciled window of this typed state directly into LLM dialogue context rather than relying solely on freeform associative recall. Bulk/windowed queries use indexed and window-function SQL rather than per-entity round trips, keeping `/api/world` and the per-turn context build fast even with thousands of entities.

Every outward action `_speak`, `focus_camera`, and `inspect_entity` now goes through a shared propose → `PolicyValidator` check → execute → record pipeline (`_dispatch_gated_action` in `runtime.py`) instead of executing directly: an `ActionProposal` is validated against world-model policy (e.g. zone/location constraints) before it runs, and both the proposal and its outcome become world-model evidence. `focus_camera` and `inspect_entity` are real executors — the runtime can direct attention to a specific camera or crop and describe a live-tracked entity, gated the same way spoken output is. Real-time gaze detection (`egg_companion/core/gaze.py`) classifies a person's gaze direction from pose keypoints and feeds `gaze_state` into both the world model and the attention scorer's `gaze_bonus`, so being looked at contributes to what the companion notices next, alongside `camera_focus_bonus` for whatever `focus_camera` is currently pointed at.

## Fused multi-camera voxel occupancy mapping

The four cameras are a co-located panoramic array (video0 rightmost, sweeping counter-clockwise through video1–video3, ~60° between neighbors, configured in `occupancy.camera_yaw_degrees`), not independent unrelated viewpoints. `egg_companion/core/occupancy.py`'s `VoxelGrid` fuses every camera's monocular metric depth into one shared "egg frame" grid by rotating each camera's back-projected points by its known array yaw before voxelizing, rather than keeping four disconnected local reconstructions. Depth itself comes from Depth Anything 3 (`DA3METRIC-LARGE`), run on demand as a bounded-lifetime subprocess in a sibling project's venv (`egg_companion/adapters/depth.py`) — this Jetson doesn't have headroom to keep a ~4GB model resident alongside vision/ASR/cognition, so the tradeoff is a ~15–30s cold-load cost per cycle in exchange for guaranteed full memory reclaim on exit. One camera integrates at a time, staggered by `occupancy.update_interval_seconds`.

Per-voxel state is a log-odds occupancy estimate — the standard Bayesian occupancy-grid-mapping formulation used throughout robotics mapping (OctoMap, ROS `costmap_2d`) — not a binary flag: a direct depth hit nudges log-odds up (scaled by the depth model's per-pixel confidence), a ray traversal nudges it down, and evidence accumulates and clamps, so a single noisy observation barely moves an estimate while corroborated evidence dominates and sustained contrary evidence can still flip a voxel's classification (an object that moves away eventually reads as free again). Each occupied voxel also carries the real RGB color sampled from its source camera frame at the pixel that produced it, so the reconstruction reflects the actual observed scene rather than a synthetic confidence gradient. A compact natural-language summary (`summary_text()`) is written into the world model as `environment:egg`'s `occupancy_summary` property, available to LLM context the same way any other world-model fact is.

The `/vision` dashboard page renders this as an orbitable 3D scene (`services/vendor/occupancy_scene.js`) alongside each contributing camera's live feed drawn as a textured plane at its correct radial position, matching the same yaw convention used server-side. It shares the `/graph` page's WebGL renderer rather than opening a second context — some browser/GPU configurations only sustain one live WebGL context per page — and transparently falls back to a pure 2D-canvas compatibility renderer (same rotation/projection technique as the knowledge-graph page's own fallback) when no WebGL context is available at all, so the view still renders on GPU-disabled or remote-desktop browser sessions.

## Evidence-grounded meta-graph and evolving cognitive documents

Quiet-period replay supplies a graph-over-graph provenance ledger without assigning semantic importance from counts, lexical frequency, detector repetition, object categories, or hand-authored gap rules. The interruptible Omnius dream agent decides which associations matter after inspecting the chronological ledger and, when useful, invoking bounded memory, graph, evidence, or web-search tools. Its model-authored themes become stable `abstraction` nodes with `expresses_theme`, `informs_world_model`, and `informs_observation_policy` synapses; every link retains the source narrative and artifact references selected by the model. Revisions retract obsolete model themes and derived edges without deleting their source evidence. Legacy count-threshold recurring-pair abstractions are retired rather than allowed to compete with this path.

Four revisioned `cognitive_document` nodes compound over this meta-graph: **World model**, **My story**, **Communication strategy**, and **Reflective working set**. `agent:egg → maintains/guides_communication → document`, `reflection → informs_working_set → document`, and `abstraction → informs_world_model → document` edges keep their provenance visible and clickable. “My story” uses first-person language but explicitly remains a source-grounded, revisable account rather than a claim of subjective experience. The working set is inspectable reflective state, not private chain-of-thought. A bounded slice feeds normal dialogue, fresh visual questions, and proactive visual communication; subsequent spoken, suppressed, corrected, interrupted, modality, and tool outcomes become new evidence for later revisions.

Identity dreams also drive chronological consolidation, even when a pass performs zero identity merges. A lightweight startup replay independently discovers every retained local-calendar day and backdates never-narrated history oldest-first in bounded passes, always refreshing the latest observed day; it repeats until the reported backlog reaches zero. Identity merges additionally rebuild every affected canonical/alias day. Frame-level evidence is coalesced into ordered, configurable time windows carrying people, objects, OCR content, sound events, admitted speech, agent replies, source modalities, episode IDs, and artifact IDs. The deterministic reducer joins admitted ASR and agent outcomes by durable conversation context ID, but deliberately stops at an inspectable provenance ledger; it does not decide meaning with lexical filters, stop-word tables, frequency thresholds, or hand-authored topic rules.

Each day becomes a revisioned `daily_narrative` node with an evidence-grounded ledger and explicit `appears_in_day`, `observed_in_day`, `read_in_day`, `heard_in_day`, `precedes_day`, `replays_day`, and `contributes_to_story` synapses. During quiet periods an interruptible Omnius dream agent reads the ledger and the current versioned narrative constitution, chooses and invokes bounded memory-search, graph-inspection, evidence-inspection, or web-search tools, and then authors nested episodes, themes, uncertainties, story updates, and the next observation policy. It may propose a general revision to its own narrative constitution; a separate model pass reviews the proposal before activation, and accepted revisions retain their prior text, model ID, source chapter, tool audit, and timestamp. Live human speech cancels this background inference immediately.

Model-authored `narrative_theme` nodes join meaning across days through `expresses_theme` and `informs_observation_policy` edges. Obsolete model themes and their synapses are retracted without deleting source evidence. The current model-authored policy is injected directly into reply context and cognition telemetry; code does not recover keywords or infer relevance from label substrings. The model names exact graph entity IDs and chooses `observe`, `retrieve`, `ask`, `speak`, or `deprioritize`; novelty cannot independently cross a numeric threshold and manufacture outward behavior. Responses to a model-authored question are interpreted by the model rather than stop phrases or question-word tables. A global proactive cooldown, visible-person requirement, provenance constraints, tool permissions, schema validation, and capacity bounds remain fixed operational safety contracts. Heard assertions never become facts solely because they are repeated.

The same semantic state revises **World model**, **My story**, **Communication strategy**, and **Reflective working set**, so later observation and dialogue are conditioned by a bounded summary rather than a raw transcript dump. The Dreams audit reports discovered history, chapters backdated, backlog remaining, and the resulting **My story** revision. The in-page Narrative workspace presents days newest-first as a vertical timeline; cards expose the day’s themes and open-thread count, and expanding a day shows its conversation arc, memory updates, latest-to-oldest encounter periods, nested episode summaries, modality tags, and retained image/audio/OCR artifacts. Selecting an orange daily-story or related theme node in the graph exposes the same grounded chapter from its associative context.

The architecture draws on complementary research patterns: observation/retrieval/reflection feedback from [Generative Agents](https://arxiv.org/abs/2304.03442); bounded tiered conversational context from [MemGPT](https://arxiv.org/abs/2310.08560); structural relational generalization from the [Tolman–Eichenbaum Machine](https://doi.org/10.1016/j.cell.2020.10.024); episodic knowledge organized under a working self from the [Self-Memory System](https://doi.org/10.1037/0033-295X.107.2.261); partner-sensitive terminology from [Lexical Entrainment for Conversational Systems](https://aclanthology.org/2023.findings-emnlp.22/); narrative integration at event boundaries from [Baldassano et al.](https://doi.org/10.1016/j.neuron.2017.06.041); and the relation among familiarity, habituation, and novelty allocation demonstrated by [Cooke et al.](https://doi.org/10.1038/nn.3920). Those sources motivate an engineering contract—replay, reflection, selective retrieval, prediction error, and habituation—not a claim that Egg implements a biological mind.

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
