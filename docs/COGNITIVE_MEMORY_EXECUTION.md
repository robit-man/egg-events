# Cognitive Memory Execution Ledger

This is the implementation-level checklist for the work orders. Every checked item must have a test or a recorded hardware trace; no component is considered complete because a UI placeholder exists.

## Active Defect Closure

- [x] Treat Jetson GPU `control=auto` as healthy and run the CUDA probe independently.
- [x] Make readiness checks diagnostic-only so one or several failures cannot globally halt the runtime.
- [x] Accept one usable camera, downgrade unavailable peers to warnings, and supervise component retries independently.

- [x] Replace exact volatile detector-label signatures with stable entity/mask/scene signatures.
- [x] Require repeated confirmation before a detector-label change can split an episode.
- [x] Verify a static four-camera scene stays below the configured episode-rate ceiling.
- [x] Change Omnius cognition from the failing compatibility route to the installed `/v1/chat` contract.
- [x] Validate both direct-answer and silent-decision response parsing.
- [x] Serialize ASR, cognition, and Ornith requests through one runtime gate.
- [x] Configure Ornith requests to unload after sparse teaching calls.
- [x] Capture and expose VLM HTTP response bodies on failures without leaking image data.
- [x] Verify ReSpeaker short-read recovery resets the reader for the next waveform chunk.

## Ornith Corrective Object Learning

- [x] Enumerate every existing masked object profile at startup.
- [x] Preserve each original label as an immutable historical claim.
- [x] Submit the alpha-masked crop, never the enclosing camera rectangle, to Ornith.
- [x] Require strict `{label, confidence}` JSON and reject prose, empty labels, and out-of-range confidence.
- [x] Store Ornith model ID, detector label, detector confidence, timestamp, and mask checksum as provenance.
- [x] Permanently revise a wrong active label without deleting its prior claim.
- [x] Persist the corrected label into the legacy profile and graph entity in one logical operation.
- [x] Recompute or retain the CLIP embedding against the exact saved transparent mask.
- [x] Match every eligible live mask against the local CLIP library before VLM escalation.
- [x] Attach recalled object entity ID, label, similarity, evidence count, and provenance to live detections.
- [x] Overlay the recalled/corrected label instead of the unstable base-model class.
- [x] Feed recalled objects into scene context and associative retrieval.
- [x] Escalate to Ornith only when no local embedding clears threshold or evidence conflicts.
- [x] Deduplicate queued masks by perceptual fingerprint, stability, and cooldown.
- [x] Bound queue size, retries, backoff, image dimensions, and concurrent model residency.
- [x] Expose `pending`, `recalled`, `vlm_verified`, `user_corrected`, and `failed` states in telemetry.
- [x] Add a test proving a recalled mask bypasses Ornith.
- [x] Add a test proving a confident Ornith correction survives restart.
- [ ] Hardware-test one existing mask review and one future CLIP-only recall.

## End-to-End Cognitive Data Path

- [x] Assign stable evidence IDs before any graph write.
- [x] Store only VAD-accepted speech events with RMS, VAD duration, DOA, and ASR model provenance.
- [x] Store vision events only after continuity/stability gating.
- [x] Keep face-confirmed persons, appearance tracks, object categories, and object instances distinct.
- [x] Link every entity sighting to evidence and its containing episode.
- [x] Convert user naming and corrections into append-only claims and revisions before replying.
- [x] Import legacy people and objects idempotently with source profile IDs.
- [x] Retrieve candidates by entity, lexical cue, recent episode, and modality embedding.
- [x] Rerank candidates by confidence, evidence quality, recency, graph path, and correction state.
- [x] Build a bounded context containing only supported claims and explicit uncertainty.
- [x] Pass that context into Omnius cognition for every directed utterance.
- [x] Decide audible interaction with deterministic policy before TTS.
- [x] Record both spoken actions and suppression reasons.
- [x] Run idle consolidation one bounded job at a time.
- [x] Retain evidence/checksums while expiring media according to policy.
- [x] Provide inspect, alias, correction, export, and delete APIs.
- [x] Ensure no API serializes raw embedding blobs.
- [x] Add deterministic trace metrics for camera, overlay, waveform, ASR, object-learning, memory, and failures.
- [x] Run unit, integrity, trace, audit, camera, waveform, ASR, cognition, VLM-teaching, and restart-recall gates.

## WO-001: Contracts and Controls

- [x] Create the `egg_companion.memory` package boundary.
- [x] Define immutable evidence, event, episode, retrieval, and attention contracts.
- [x] Add bounded memory, event segmentation, cognitive attention, and privacy configuration.
- [x] Add contract serialization tests and invalid configuration tests.

## WO-002: Evidence Graph Store

- [x] Create idempotent SQLite schema migrations with WAL and foreign keys.
- [x] Add append-only evidence, entities, episodes, claims, edges, embeddings, revisions, and jobs.
- [x] Enforce local relative media keys and transaction rollback behavior.
- [x] Add entity/evidence linking, revision, graph queries, and cascade deletion APIs.
- [x] Import legacy identity and object profiles idempotently with provenance.
- [x] Verify SQLite reopen persistence on the Jetson filesystem.

## WO-003: Event Segmentation

- [x] Build bounded per-camera, speech, reasoning, memory, and object-candidate queues.
- [x] Segment vision, valid speech, correction, and identity-change events into episodes.
- [x] Suppress repeated static-frame durable writes.
- [x] Persist only VAD-accepted and ASR-grounded speech evidence.
- [x] Verify static-scene episode count remains bounded on hardware.

## WO-004: Entity Resolution

- [x] Resolve face-confirmed people conservatively into graph entities.
- [x] Preserve anonymous appearance tracks separately from confirmed people.
- [x] Attach mask-aware object evidence and user aliases as versioned claims.
- [x] Record confidence components and correction provenance.
- [x] Route only stable unknown masked instances to Ornith Vision 9B.
- [x] Save accepted Ornith labels as `vlm_verified` CLIP object evidence, then recall locally before any future VLM call.
- [x] Re-escalate only when local CLIP recall is below threshold, evidence conflicts, or a user corrects the label.

## WO-005: Associative Recall

- [x] Generate candidates from recent episodes, entity IDs, transcript terms, and embeddings.
- [x] Rerank with temporal, graph, quality, and correction signals.
- [x] Build a source-grounded, bounded LLM context block.
- [x] Verify irrelevant and revised memories are excluded or explicitly marked.

## WO-006–010: Cognitive Operation and Governance

- [x] Replace frame novelty with explainable world-state prediction residuals and habituation.
- [x] Separate interaction permission from attention capture priority.
- [x] Add consolidation/replay jobs and retention enforcement.
- [x] Expose inspect, correction, export, and deletion controls in the dashboard.
- [x] Run replay traces, camera stress, audio stress, and recovery tests on hardware.

## Verification Evidence

- [x] `57` CPU-safe tests pass after readiness degradation, evaluation, retention, dialogue, telemetry, buffer, and dashboard API changes.
- [x] A live alpha-masked camera crop was classified by `robit/ornith-vision:9b` as `television` at `0.95` confidence.
- [x] The four discovered cameras previously sustained approximately `7–8` raw FPS with `90°` corrected orientation.
- [x] The ReSpeaker six-channel silence probe measured channel `0` RMS `0.007928` with no sustained VAD activation.
- [x] SQLite integrity is `ok`, foreign keys have no violations, embeddings have no orphans, and legacy sources have no duplicates.
- [x] Jetson GPU runtime-PM deadlock was isolated to `gk20a_busy → rpm_resume` with `runtime_status=suspending`.
- [x] `egg-gpu-pm-guard.service` is installed and enabled before display-manager and Ollama startup.
- [x] Ollama is constrained to one loaded model, one parallel request, and `4096` context tokens.
- [x] Reboot once to activate the GPU PM guard and clear uninterruptible NVIDIA driver waiters.
- [x] Post-reboot hardware audit passes with `control=auto`, direct CUDA 12.2, four camera frames, ReSpeaker I/O/DOA, Omnius voice, and `/v1/chat` cognition.
- [x] A 10-second live trace passes four independent raw/mask streams, waveform, memory, and zero runtime/hardware failures.
- [x] A 300-second four-camera trace passes with `2.8` visual episode starts/minute below the `4.0` ceiling, `9000` waveform updates, and no runtime or hardware failures.
- [x] Two live legacy migrations preserve `13` identities with no duplicate sources, orphan embeddings, missing media, or checksum mismatches.
- [x] The metadata-only baseline evaluation passes event-boundary, identity, object, ASR, retrieval, interaction, correction, latency, and growth gates.
- [x] After the installed-service restart, re-run the complete `57`-test suite, memory/restart recall, grounded ASR and deterministic TTS-echo rejection, a live four-camera/audio trace, and the final service-token audit with all `32` checks passing.
