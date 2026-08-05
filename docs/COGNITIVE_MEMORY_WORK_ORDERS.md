# Egg Cognitive Memory Work Orders

**Derived from:** [`COGNITIVE_MEMORY_RESEARCH.md`](COGNITIVE_MEMORY_RESEARCH.md)
**Status:** implemented; real-device acceptance evidence is tracked in `COGNITIVE_MEMORY_EXECUTION.md`
**Scope:** convert the cognitive-memory research into incremental, testable changes in the current Egg codebase.

## Execution Rules

- Preserve the single `./egg` bootstrap/launch path and local-only operation.
- Keep video, VAD, AEC, DOA, detection, and identity inference outside the LLM critical path.
- Make every persistent fact source-grounded: confidence alone is insufficient without evidence, time, and origin.
- Migrate existing identity/object stores; do not discard current local data during a memory-schema upgrade.
- Treat face-confirmed profiles and appearance-only tracks as different evidence classes.
- Ship each work order with unit tests first, then a real-device verification step where applicable.
- Do not use open-ended reward objectives. Attention and speech decisions must be explainable score/policy outcomes.

## Current Foundation Map

| Existing responsibility | Current location | Target role |
| --- | --- | --- |
| Runtime orchestration | `egg_companion/runtime.py` | injects observations, audio events, graph writes, retrieval, and speech policy |
| Data models | `egg_companion/models.py` | stable transport contracts for observations, events, evidence, and memory results |
| Vision and embeddings | `egg_companion/adapters/vision.py` | emits detections, face embeddings, CLIP embeddings, masks, behavior, and quality metadata |
| ReSpeaker capture / VAD / DOA | `egg_companion/adapters/audio.py` | emits grounded speech events with RMS, VAD, DOA, and AEC state |
| Omnius ASR/TTS/LLM | `egg_companion/adapters/omnius.py` | performs only ASR, TTS, and deliberate source-grounded reasoning |
| Attention tracking | `egg_companion/core/attention.py` | becomes the low-latency novelty and target-selection layer |
| Scene frame tracking | `egg_companion/services/scene.py` | becomes a short-term visual continuity adapter, not durable memory |
| Identity persistence | `egg_companion/services/identity.py` | migrates into graph-backed entity/profile evidence |
| Object learning | `egg_companion/services/object_library.py` | migrates into graph-backed object entities and segmentation evidence |
| Dashboard telemetry | `egg_companion/services/telemetry.py` | exposes memory lifecycle, evidence, attention scores, and runtime metrics |
| Dashboard HTTP | `egg_companion/services/dashboard.py` | adds inspect/correct/export/delete memory APIs and views |
| Runtime settings | `egg_companion/config.py`, `config/egg.yaml` | adds bounded, validated memory/attention/privacy configuration |

## Dependency Order

```text
WO-001 contracts/config
       |
       +--> WO-002 evidence store ----> WO-003 event segmentation
       |            |                           |
       |            +--> WO-004 entity migration +--> WO-005 associative retrieval
       |                                             |
       +--> WO-006 attention policy ----------------+--> WO-007 context + interaction
                                                            |
WO-008 consolidation <-------------------------------------+
       |
       +--> WO-009 dashboard/governance
       |
       +--> WO-010 trace-based evaluation and hardware validation
```

Implement in order. A later work order may use a feature flag, but must not bypass an unmet earlier data contract.

---

## WO-001 — Define Cognitive Contracts and Configuration

**Layer:** shared contracts and configuration
**Prerequisites:** none
**Deliverable:** new typed data contracts and explicit feature controls before changing runtime behavior.

### Implement

- Add `egg_companion/memory/__init__.py` as the new memory package boundary.
- Extend `egg_companion/models.py` with frozen transport types:
  - `EvidenceRef`: ID, modality, source URI or local media key, camera/audio source, timestamp, quality, and metadata.
  - `PerceptualEvent`: typed observation input (`vision`, `speech`, `object`, `identity`, `user_correction`, `attention`).
  - `EpisodeDraft`: event-boundary candidate with time range, evidence refs, entities, surprise components, and summary fields.
  - `MemoryHit`: entity/episode/claim result with score, confidence, provenance, and explanation terms.
  - `AttentionDecision`: capture priority, outward-speech permission, score components, reason, and cooldown.
- Add configuration models in `egg_companion/config.py`:
  - `MemoryConfig`: `enabled`, `storage_dir`, raw-media retention, episode min/max duration, retrieval count, and graph traversal bounds.
  - `EventSegmentationConfig`: thresholds for entity/action/speech/DOA/prediction/correction boundaries.
  - `CognitiveAttentionConfig`: novelty weights, interruption threshold, proactive speech rate limit, and uncertainty-question budget.
  - `PrivacyConfig`: persistent identity enablement, profile retention, evidence retention, export/delete enablement.
- Add corresponding defaults to `config/egg.example.yaml`; add explicit local values to `config/egg.yaml` only where behavior must differ from defaults.
- Keep `IdentityConfig` and `ObjectLearningConfig` operational during migration; add a `migration_mode` enum (`legacy`, `dual_write`, `graph`) to transition safely.

### Do Not

- Do not add a graph database dependency. SQLite is the foundation.
- Do not add a new always-on ML model in this order.
- Do not change existing camera rotation or ReSpeaker configuration.

### Tests

- Add `tests/test_memory_config.py` for defaults, invalid ranges, and YAML validation.
- Add `tests/test_memory_models.py` for immutable contracts and serializable evidence metadata.

### Acceptance Gate

`./egg test` passes with the memory feature disabled and current runtime behavior unchanged.

---

## WO-002 — Build the Local Evidence and Temporal Graph Store

**Layer:** durable episodic foundation
**Prerequisites:** WO-001
**Deliverable:** a SQLite-backed, append-first property graph in `data/cognitive-memory/`.

### Implement

- Add `egg_companion/memory/store.py` as the public storage API. No other module should issue raw SQL directly.
- Add `egg_companion/memory/schema.py` for idempotent migrations and schema versioning using `PRAGMA user_version`.
- Create tables:
  - `entities(entity_id, entity_type, display_name, state, created_at, updated_at, merged_into, metadata_json)`.
  - `episodes(episode_id, started_at, ended_at, state, novelty, summary, created_at)`.
  - `claims(claim_id, subject_id, predicate, object_id_or_text, confidence, state, valid_from, valid_to, created_at, revised_at)`.
  - `edges(edge_id, source_id, relation, target_id, confidence, valid_from, valid_to, confirmation_count, state, metadata_json)`.
  - `evidence(evidence_id, modality, captured_at, source_type, source_id, media_key, quality, payload_json, embedding_key)`.
  - `episode_evidence(episode_id, evidence_id, role)` and `entity_evidence(entity_id, evidence_id, role)`.
  - `revisions(revision_id, target_type, target_id, decision, replacement_value, actor, created_at, evidence_id)`.
  - `embeddings(embedding_id, owner_type, owner_id, modality, model_id, dimensions, vector_blob, quality, created_at)`.
  - `jobs(job_id, kind, state, payload_json, started_at, completed_at, error)` for consolidation/retry observability.
- Add indexes for timestamps, entity type/state, claim subject/predicate, evidence source/time, and embedding owner/modality.
- Use SQLite WAL mode, foreign keys, parameterized queries, a single process-safe connection strategy, and transaction boundaries per event.
- Implement store API methods: `append_evidence`, `open_episode`, `append_episode_evidence`, `upsert_entity`, `link_entities`, `assert_claim`, `revise_claim`, `close_episode`, `recent_episodes`, `entity_detail`, and `delete_entity_cascade`.
- Preserve raw media outside SQLite in a bounded `data/cognitive-memory/media/` hierarchy. Store only local relative keys and checksums in SQLite.

### Migration Integration

- Add `egg_companion/memory/migrate_legacy.py` to import existing `IdentityLibrary` SQLite profiles and file-backed `ObjectLibrary` profiles as entities with migration provenance.
- Import face/CLIP/object embeddings with their source model names and mark imported observations as `legacy_import`, not as new live events.
- Run migration idempotently and record the source profile ID in metadata.

### Tests

- Add `tests/test_memory_store.py`: transaction rollback, temporal edges, revision append-only behavior, evidence linkage, and SQLite reopen persistence.
- Add `tests/test_memory_migration.py`: import fixture identity/object stores twice and verify no duplicate graph entities.

### Acceptance Gate

An empty database initializes on Jetson; an imported legacy profile is retrievable with its original local profile ID and no live runtime uses it yet.

---

## WO-003 — Convert Continuous Perception into Bounded Episodes

**Layer:** perceptual buffer and event segmentation
**Prerequisites:** WO-001, WO-002
**Deliverable:** meaningful events instead of frame-by-frame durable writes.

### Implement

- Add `egg_companion/memory/buffer.py`:
  - bounded JPEG frame references per camera;
  - bounded audio segment references with VAD/RMS/DOA metadata;
  - no unbounded NumPy frame retention;
  - per-source TTL and memory caps from `MemoryConfig`.
- Add `egg_companion/memory/segmentation.py` with `EventSegmenter.ingest(event) -> list[EpisodeDraft]`.
- Score an event boundary from independently visible components:
  - entity birth/death or identity hypothesis change;
  - object/action/relationship change;
  - valid speech start/end and materially changed DOA;
  - scene semantic change;
  - user correction, explicit object naming, or task transition;
  - later: prediction residual supplied by WO-006.
- Keep an active episode per local context (initially camera plus conversational context); merge evidence only inside configurable time windows.
- Force-close episodes at `max_duration`, on inactivity, or on explicit user interaction completion.
- In `egg_companion/runtime.py`:
  - create `PerceptualEvent` after `_analyze` and after valid ASR only;
  - feed it to the buffer and segmenter in a non-blocking task/queue;
  - continue `RuntimeTelemetry.record_observation` immediately for dashboard freshness;
  - never store rejected silent/echo VAD windows as speech evidence.

### Modify Existing Components

- Keep `SceneInventory` in `egg_companion/services/scene.py` as short-term IoU continuity only; add a clear docstring that it is not durable memory.
- Add an optional stable `track_id` attribute to `Detection` or carry a track mapping in the event payload. Do not infer identity from a detection label alone.

### Tests

- Add `tests/test_event_segmentation.py` using synthetic timestamped `PerceptualEvent` sequences.
- Assert 100 identical frames create one active episode, not 100 episodes.
- Assert a valid utterance, a person entering, and a correction each produce a boundary or evidence update.
- Assert a low-RMS/VAD-rejected chunk writes no transcript evidence.

### Acceptance Gate

With a static camera scene for five minutes, episode count remains bounded; dashboard telemetry remains responsive.

---

## WO-004 — Make People and Objects Conservative Graph Entities

**Layer:** entity resolution and cross-modal binding
**Prerequisites:** WO-002, WO-003
**Deliverable:** durable entities with evidence classes, uncertainty, and non-destructive corrections.

### Implement

- Add `egg_companion/memory/entities.py` with `EntityResolver` and explicit outcomes: `new`, `recalled`, `hypothesis`, `rejected`, `merged_by_user`.
- Refactor `egg_companion/services/identity.py` into a compatibility adapter around graph entities:
  - face embeddings create or recall `Person` entities only when SFace quality and threshold requirements pass;
  - non-face CLIP observations become `appearance_track` evidence, not confirmed people;
  - each match creates a sighting evidence edge with camera/time/confidence;
  - uncertain matches create `same_as` hypotheses with independent evidence counts;
  - names become alias claims, with actor `user`, not a mutation that erases anonymous identity provenance.
- Refactor `egg_companion/services/object_library.py` into a graph-backed object/profile adapter:
  - preserve SAM mask media and CLIP embedding as evidence;
  - user labels become versioned `has_name` claims;
  - repeated matching produces sighting edges instead of merely updating one file profile;
  - distinguish an object category (for example, `mug`) from a physical object instance when enough continuity exists.
- Add `egg_companion/memory/fusion.py` for quality-aware scoring:
  - face score is primary for person recall;
  - CLIP is supporting visual evidence;
  - time/camera continuity and user-confirmed aliases are independent signals;
  - return individual component scores to telemetry and dashboard.
- Update `_analyze` in `egg_companion/runtime.py` to attach entity IDs, resolver outcome, and evidence quality to detections without blocking camera rendering.

### Do Not

- Do not classify protected/sensitive attributes.
- Do not name a person from an unprompted guess.
- Do not merge profiles solely because clothing, posture, or a low-quality face crop looks similar.

### Tests

- Extend `tests/test_identity.py` with face-quality rejection and cross-restart graph recall.
- Add `tests/test_entity_resolver.py`: low-confidence appearance track does not create confirmed `Person`; two independent face profiles remain separate; explicit user naming produces an alias claim.
- Add `tests/test_object_memory.py`: mask-aware labeled object recall retains the segmentation evidence.

### Acceptance Gate

The dashboard can distinguish `face-confirmed`, `anonymous appearance`, and `uncertain same-person hypothesis`, each with confidence components and source evidence.

---

## WO-005 — Add Associative Multimodal Retrieval

**Layer:** pattern completion and deliberate recall
**Prerequisites:** WO-002, WO-003, WO-004
**Deliverable:** current perception and speech retrieve a compact, evidence-backed context subgraph.

### Implement

- Add `egg_companion/memory/retrieval.py` with two explicit stages:
  1. candidate generation from entity IDs, recent episodes, lexical transcript terms, and per-modality nearest-neighbor embedding search;
  2. graph expansion and quality-aware reranking using temporal proximity, edge confidence, evidence quality, recency, and correction state.
- Implement nearest-neighbor search initially by bounded in-process NumPy cosine search over the `embeddings` table. Add retrieval limits and a modality filter. Do not add an external vector database.
- Implement a deterministic, bounded personalized graph walk over `edges`, with visited-node limits, edge-type allowlists, and score explanations.
- Return `MemoryHit` objects with `why` data such as `face_similarity`, `same_camera_continuity`, `user_named_alias`, `recent_episode`, and `corrected_claim`.
- Add `egg_companion/memory/context.py` to turn a retrieval result into a maximum-size LLM context block:
  - current validated user utterance;
  - current episode summary;
  - a small list of relevant memories and supporting evidence;
  - explicit uncertainty and contradictions;
  - no raw embeddings, no unverified sensitive inferences, and no unrelated history.
- In `runtime.py`, replace `_scene_context()` with a method that combines current scene telemetry plus `MemoryContextBuilder` output for `_omnius.conversation_reply` and calibration prompts.

### Tests

- Add `tests/test_memory_retrieval.py`: partial entity cue retrieves related episode; expired/retracted claim is downranked; unrelated entity is excluded; every hit has provenance.
- Add `tests/test_memory_context.py`: size bound, no missing evidence references, contradictions remain labeled as uncertain.

### Acceptance Gate

A spoken question about a recently named object returns only its relevant graph evidence and never fabricates a memory without a source.

---

## WO-006 — Replace Frame Novelty With Explainable Cognitive Attention

**Layer:** attention, prediction error, and interruption control
**Prerequisites:** WO-001, WO-003, WO-004
**Deliverable:** low-latency attention decisions that are measurable and non-chatty.

### Implement

- Extend `egg_companion/core/attention.py` rather than creating LLM attention:
  - maintain a short-lived per-camera world state keyed by stable entity/track IDs;
  - calculate component scores for new entity, entity disappearance, action change, object relation change, speech/DOA change, semantic novelty, and repetition penalty;
  - emit `AttentionDecision` with all component values.
- Add `egg_companion/core/prediction.py` for a deliberately simple short-horizon predictor:
  - expected presence/count/location buckets for recently tracked entities;
  - prediction residual when a stable scene changes unexpectedly;
  - no opaque learned reward model in this first version.
- Refactor `AttentionManager.select` to return track/capture priority only. Move audible interruption permission into a new `egg_companion/cognition/interaction_policy.py`.
- Create `egg_companion/cognition/__init__.py` and `egg_companion/cognition/interaction_policy.py`:
  - require explicit user-directed speech, task/safety rule, or configured high-value unresolved calibration before speaking;
  - enforce per-reason cooldowns and a rolling proactive-speech budget;
  - prohibit repeated scene narration and generic greetings;
  - log suppression reason (for example `repetition`, `not_addressed`, `cooldown`, `low_information_gain`).
- In `runtime.py`, replace `_should_greet` / direct proactive `companion_reply` behavior with `InteractionPolicy.should_speak(decision, event, memory_context)`.

### Tests

- Extend `tests/test_attention.py` for repeated-frame penalty, action-change novelty, and explanation fields.
- Add `tests/test_interaction_policy.py`: static scene produces no repeated speech; direct user question passes; low-confidence object correction obeys question budget.

### Acceptance Gate

The dashboard shows attention components and the runtime can explain every spoken or suppressed proactive action without consulting the LLM.

---

## WO-007 — Ground Omnius Cognition in Memory and Natural Dialogue

**Layer:** deliberate language reasoning and calibration
**Prerequisites:** WO-005, WO-006
**Deliverable:** Omnius receives bounded evidence, not canned scene text or unlimited chat history.

### Implement

- Refactor `egg_companion/adapters/omnius.py`:
  - replace `_conversation` as the sole long-term state with a short conversation window plus `MemoryContext` supplied by the runtime;
  - add structured methods: `reason_about_utterance`, `interpret_correction`, `interpret_person_naming`, and `interpret_object_naming`;
  - require strict JSON only for classifier-like tasks, validate schema locally, and store the original utterance/evidence separately;
  - keep `conversation_reply` for natural language, but instruct it that it may only state graph-supported facts and must express uncertainty where supplied.
- Add `egg_companion/cognition/dialogue.py`:
  - classify whether a valid utterance is directed to Egg without a hard-coded wake word;
  - identify interaction acts: question, correction, naming, command, acknowledgement, conversation;
  - use recent DOA, timing after TTS, current episode, and language reasoning as evidence—not transcript text alone;
  - create user-correction events in the graph before generating a reply.
- Update `runtime._listen`:
  - retain existing VAD/RMS gate before any Omnius ASR call;
  - derive current episode and memory context after transcription;
  - process corrections/naming before normal conversation;
  - invoke TTS only after `InteractionPolicy` permits it;
  - append reply evidence to the active episode for later audit.
- Update the Omnius system prompt to forbid unsupported identity assertions, unsupported personal claims, repetitive acknowledgements, and ungrounded visual narration. It should not claim first-person perceptual certainty beyond supplied evidence.

### Tests

- Add `tests/test_dialogue.py` with mocked Omnius JSON responses: user names person, corrects object, asks ordinary question, and room conversation not directed to Egg.
- Add `tests/test_omnius_payloads.py`: context bound, schema validation, no raw media or embeddings in requests.

### Acceptance Gate

An explicit correction becomes a graph revision before Egg replies; an undirected room conversation does not trigger a canned response.

---

## WO-008 — Add Idle Consolidation, Replay, and Retention

**Layer:** multi-timescale memory
**Prerequisites:** WO-002 through WO-007
**Deliverable:** repeated experience becomes compact semantic memory without deleting evidence.

### Implement

- Add `egg_companion/memory/consolidation.py` with idempotent job types:
  - `deduplicate_sightings` for near-identical evidence in one episode/window;
  - `strengthen_repeated_edges` by increasing confirmation counts rather than overwriting facts;
  - `derive_semantic_candidates` from repeated, high-confidence episode patterns;
  - `expire_raw_media` under `PrivacyConfig`, retaining evidence metadata/checksum according to policy;
  - `replay_recent_episodes` to form candidate temporal schemas only when source evidence exists.
- Use the existing runtime event loop to schedule bounded idle work after a configurable quiet interval. Run expensive jobs one at a time and yield between batches so camera/audio latency remains protected.
- Persist each job in the `jobs` table and expose retries/errors; never silently discard failed consolidation.
- Semantic claims derived by replay must be `candidate` until a confirmation threshold or an explicit user statement is met.
- Add `egg_companion/memory/retention.py` for deterministic deletion/export planning. Deleting an entity must remove or tombstone dependent embeddings, media, evidence links, and aliases according to policy.

### Tests

- Add `tests/test_consolidation.py`: duplicate sightings compact correctly; conflicting claims stay distinct; semantic candidates retain source episode IDs; deletion reaches dependent media keys.
- Add `tests/test_retention.py`: expired media is deleted only after configured age and legal graph references are retained/tombstoned correctly.

### Acceptance Gate

After repeated sightings, the graph becomes smaller/more connected without losing the ability to show raw supporting episodes or user corrections.

---

## WO-009 — Expose Memory, Attention, and Governance in the Dashboard

**Layer:** observability and user control
**Prerequisites:** WO-002 through WO-008, implemented incrementally as data becomes available
**Deliverable:** local dashboard makes the system inspectable and correctable.

### Implement

- Extend `egg_companion/services/telemetry.py` with thread-safe snapshots for:
  - active episode ID and start/end state;
  - latest event-boundary reason and score components;
  - current attention decision and speech suppression reason;
  - retrieval hits with confidence, modality scores, and evidence count;
  - graph job/consolidation state and memory-store size;
  - per-camera corrected frame dimensions/aspect ratio already present in camera telemetry.
- Extend `egg_companion/services/dashboard.py` APIs:
  - `GET /api/memory/episodes`, `GET /api/memory/entities/{entity_id}`, `GET /api/memory/claims`;
  - `POST /api/memory/entities/{entity_id}/alias` for explicit naming;
  - `POST /api/memory/revisions` for correction/merge rejection;
  - `GET /api/memory/export/{entity_id}` and `DELETE /api/memory/entities/{entity_id}` gated by `PrivacyConfig`;
  - `GET /api/cognition/state` for attention, retrieval, and consolidation health.
- Update the existing dark dashboard page in `egg_companion/services/dashboard.py` to show:
  - live active episode and event boundary reason;
  - People separated into face-confirmed profiles, anonymous appearance tracks, and unresolved hypotheses;
  - object cards with segmentation mask thumbnails, labels, confidence, and evidence count;
  - a graph/evidence inspector for selected entity/claim;
  - audio waveform, RMS, VAD, DOA, ASR transcript, TTS model/voice, and current cognition model;
  - spoken/suppressed action ledger so repetitive behavior is visible;
  - retention/export/delete controls with confirmation UX.
- Never expose raw embeddings in HTTP responses. Use local evidence thumbnails/audio references only under configured retention/privacy rules.

### Tests

- Add `tests/test_dashboard_memory_api.py` using an `aiohttp` test client for state, revision, export authorization, and deletion behavior.
- Add serialization tests to ensure API output contains evidence IDs/confidence but no embedding blobs.

### Acceptance Gate

A user can inspect why Egg recognized or did not recognize something, correct it, see the revision persisted, and delete the profile locally.

---

## WO-010 — Build a Trace-Based Evaluation and Real-Hardware Verification Harness

**Layer:** quality gates and regression prevention
**Prerequisites:** each preceding order contributes its own unit tests
**Deliverable:** reproducible local evaluation before claims about cognitive behavior.

### Implement

- Add `tests/fixtures/traces/` with consented, metadata-only synthetic/recorded event traces. Do not commit private face images or audio unless explicitly approved and access-controlled.
- Add `egg_companion/evaluation/`:
  - `event_boundaries.py` for precision/recall against labeled episode boundaries;
  - `identity_metrics.py` for false merge, false split, and recall measures;
  - `retrieval_metrics.py` for source-grounded memory QA and latency;
  - `interaction_metrics.py` for interruption rate, repeated-speech rate, correction retention, and undirected-speech suppression;
  - `hardware_smoke.py` for non-destructive real-device checks.
- Add CLI subcommands in `egg_companion/cli.py`:
  - `egg evaluate --trace <path>` for deterministic offline scoring;
  - `egg memory migrate` for idempotent legacy import;
  - `egg memory verify` for database integrity, media checksums, and retention dry run.
- Extend `egg_companion/services/audit.py` with non-invasive checks for memory DB writable/WAL-ready, expected model checkpoints, configured retention directories, and Omnius reasoning endpoint health.
- Update `scripts/bootstrap-jetson.sh` and the root `egg` script only if a newly introduced dependency cannot use the existing Python standard library/installed dependencies. Prefer no new dependency for the SQLite graph phase.

### Required Metrics

| Area | Minimum metric |
| --- | --- |
| Event segmentation | boundary precision/recall; static-scene event rate |
| Identity | false merge rate, false split rate, confirmed-profile recall |
| Object learning | segmentation provenance retained; user-label recall accuracy |
| ASR grounding | rejected silent/echo windows; accepted speech windows with VAD/RMS metadata |
| Retrieval | provenance coverage, answer-support rate, p50/p95 latency |
| Interaction | unsolicited speech rate, duplicate reply rate, correction retention |
| Runtime | camera FPS, ASR delay, GPU/RAM, DB/media growth |

### Acceptance Gate

`./egg test`, `./egg audit`, and `./egg evaluate --trace ...` report separate pass/fail results. No claim of identity, recall, or natural interaction quality is made without a measured trace result.

---

## Implementation Sequence and Review Checkpoints

| Checkpoint | Work orders | Review question |
| --- | --- | --- |
| Foundation | WO-001, WO-002 | Can every stored fact point to evidence and survive restart? |
| Perception | WO-003, WO-004 | Does continuous sensing form stable episodes/entities without duplicate-frame inflation or unsafe identity merges? |
| Recall | WO-005 | Can Egg retrieve only relevant, provenance-backed context from partial cues? |
| Behavior | WO-006, WO-007 | Does Egg remain quiet unless engagement or high-value calibration justifies speaking? |
| Long-term memory | WO-008 | Does replay strengthen knowledge without overwriting corrections or hiding uncertainty? |
| Trust and quality | WO-009, WO-010 | Can a user inspect, correct, delete, and measure all important cognitive behavior? |

## Initial File Change Inventory

### New Packages and Modules

```text
egg_companion/memory/__init__.py
egg_companion/memory/schema.py
egg_companion/memory/store.py
egg_companion/memory/migrate_legacy.py
egg_companion/memory/buffer.py
egg_companion/memory/segmentation.py
egg_companion/memory/entities.py
egg_companion/memory/fusion.py
egg_companion/memory/retrieval.py
egg_companion/memory/context.py
egg_companion/memory/consolidation.py
egg_companion/memory/retention.py
egg_companion/cognition/__init__.py
egg_companion/cognition/interaction_policy.py
egg_companion/cognition/dialogue.py
egg_companion/core/prediction.py
egg_companion/evaluation/__init__.py
egg_companion/evaluation/event_boundaries.py
egg_companion/evaluation/identity_metrics.py
egg_companion/evaluation/retrieval_metrics.py
egg_companion/evaluation/interaction_metrics.py
egg_companion/evaluation/hardware_smoke.py
```

### Existing Modules to Modify

```text
egg_companion/models.py
egg_companion/config.py
config/egg.yaml
config/egg.example.yaml
egg_companion/runtime.py
egg_companion/core/attention.py
egg_companion/services/identity.py
egg_companion/services/object_library.py
egg_companion/services/scene.py
egg_companion/services/telemetry.py
egg_companion/services/dashboard.py
egg_companion/services/audit.py
egg_companion/adapters/omnius.py
egg_companion/cli.py
scripts/bootstrap-jetson.sh       # only if dependency audit proves necessary
egg                              # only if subcommands need dispatch changes
```

## Definition of Done for the Whole Program

The cognitive-memory implementation is complete only when Egg can, on real configured hardware:

1. observe a continuous multimodal scene without creating a new durable item per frame;
2. retain and recall evidence-backed people, objects, episodes, and user corrections across restart;
3. keep unconfirmed appearance similarity separate from face-confirmed person recall;
4. use VAD-gated speech, DOA, and scene context to determine whether a response is appropriate without a hard-coded wake word;
5. provide Omnius only concise, retrieved, confidence-labeled context for deliberate reasoning;
6. consolidate repeated events without losing evidence, correction history, or delete/export controls;
7. expose all confidence, provenance, attention, ASR, and retention state in the local dashboard; and
8. pass unit, trace, audit, and real-hardware smoke checks with reported metrics.
