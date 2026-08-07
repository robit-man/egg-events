# Egg Cognitive Memory Research Ledger

**Status:** research baseline
**Last reviewed:** 2026-08-06
**Purpose:** preserve the research rationale, target architecture, and implementation decisions for making Egg a grounded, multimodal companion with durable, evidence-based memory.

## Operating Definition

"Human-brain-like" is not a claim of consciousness or simulated emotion. For Egg, it means implementing falsifiable cognitive capabilities:

- Segment continuous perception into meaningful events rather than treating every frame as a new observation.
- Bind vision, faces, objects, speech, direction-of-arrival, actions, time, and place into evidence-backed associations.
- Separate similar people or objects conservatively, then complete a recalled memory from partial present cues.
- Allocate attention using novelty, prediction error, relevance, and uncertainty rather than an opaque reward objective.
- Consolidate repeated experience into compact semantic knowledge while preserving raw evidence and corrections.
- Speak sparingly, naturally, and only when an explicit request, safety condition, or useful information opportunity justifies interruption.

This is a cognitive-systems design target, not an attempt to imitate a brain's biological implementation.

## Research Findings

### Episodic Memory Must Be Event-Based

A video frame is an observation, not a memory. Egg should create a multimodal episode when there is a meaningful boundary, such as a new person or object, a change in action or relationship, speech or direction-of-arrival change, a large prediction error, a user correction, or a task transition. An episode should normally span roughly 2–30 seconds and refer to the underlying media buffers rather than copy all raw media into the database.

This direction is supported by recent embodied-memory work that combines spatial, temporal, episodic, and semantic memory in a dynamic knowledge graph, and by recent dialogue memory work that uses event segmentation for durable episodic storage. [RoboMemory (arXiv 2025)](https://arxiv.org/abs/2508.01415) and [ES-Mem (arXiv 2026)](https://arxiv.org/abs/2601.07582) are research preprints and require local validation before their mechanisms are adopted wholesale.

### Graph Memory Is Better Than a Flat Transcript or Vector Store

A useful companion needs explicit relationships and provenance. Store entities and episodes in a temporal property graph, backed locally by SQLite at first:

| Node | Examples |
| --- | --- |
| `Person` | known named person, anonymous stable person profile |
| `Object` | segmented held object, recurring object instance, user-named item |
| `Place` | camera view now; calibrated physical location later |
| `Episode` | bounded multimodal event |
| `Utterance` | transcript, speaker direction, VAD confidence |
| `Action` | holding, entering, speaking, placing, approaching |
| `Claim` | "this object is called mug", "Alex prefers tea" |
| `Hypothesis` | uncertain identity, object label, relationship, or causal link |

| Edge | Meaning |
| --- | --- |
| `seen_in` / `heard_in` | perceptual evidence for an episode |
| `speaks_about` | utterance refers to an entity or claim |
| `holds`, `near`, `looks_at` | spatial or behavioral relation |
| `before`, `during`, `caused` | temporal or causal relation |
| `same_as` | evidence-scored entity-resolution hypothesis |
| `corrected_by`, `contradicts` | explicit revision and disagreement |

Every mutable fact or edge must include a confidence, source references, validity interval, confirmation count, last-confirmed time, decay policy, and user-correction state. This makes recall inspectable and allows the system to say why it believes something.

Graph-based recall should use present cues to activate a small evidence subgraph, then deliberately rerank it before presenting it to the language model. This two-stage association-and-verification pattern is closely aligned with [Associa (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.901/) and graph-retrieval ideas from [HippoRAG (NeurIPS 2024)](https://papers.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf).

### Preserve Multiple Modalities Until Retrieval

Do not collapse all perception into one embedding too early. Egg should keep a bundle of independently inspectable feature channels:

- SFace or an equivalent face-recognition embedding for face-based person continuity.
- CLIP/SigLIP embedding for broad visual appearance and semantic similarity.
- Segmentation-mask-aware embedding for held objects rather than square crops of a face, hand, or background.
- Speech embedding, transcript, VAD quality, and ReSpeaker direction-of-arrival evidence.
- Action, pose, detection, and camera/view features.
- An optional VLM scene/event summary, always tied to its media evidence.

At recall time, combine channels using quality-aware late fusion. For example, a high-quality face embedding can outweigh weak appearance similarity; when no face is visible, an appearance track can remain provisional rather than becoming a false persistent identity. Unified audio/video/text representations are an interesting longer-term capability, but should not replace modality-specific evidence on Egg yet. See [WAVE (arXiv 2025)](https://arxiv.org/abs/2509.21990), which is an experimental research result rather than an operational dependency.

### Pattern Separation Before Pattern Completion

The system must avoid destructive merges:

1. **Pattern separation:** create separate anonymous profiles for uncertain people and object instances. Keep `same_as` as an evidence-scored hypothesis instead of immediately merging database rows.
2. **Pattern completion:** when Egg sees partial current evidence, use it to retrieve compatible graph neighborhoods, then verify against face, time continuity, appearance, location, and user history.
3. **Confirmation:** promote a merge or a name attachment only after repeated independent evidence or an explicit user statement.

This is particularly important for the current identity library. A body-only or clothing-only CLIP match is a temporary appearance observation, not a uniquely identified person. Persistent person recall should rely primarily on face/re-identification evidence with visible confidence and provenance. The conceptual inspiration comes from hippocampal associative-memory research, including [episodic and associative memory from spatial scaffolds (Nature 2025)](https://www.nature.com/articles/s41586-024-08392-y).

### Attention Should Be Surprise- and Relevance-Gated

Egg should not narrate scenes continuously. Use a deterministic, inspectable attention score:

```text
attention = novelty
          + prediction_error
          + user_relevance
          + unresolved_uncertainty
          + task_or_safety_priority
          - repetition
          - interruption_cost
```

Candidate novelty signals:

- A new stable entity, a significant identity hypothesis change, or an unfamiliar object.
- An object moved, disappeared, was handed over, or changed relationship to a person.
- A new speech segment, speaker direction change, or speech/vision mismatch.
- An event that violates a short-horizon world prediction.
- A direct correction, question, instruction, or remembered preference.

Use low-level attention for capture and memory writing. Permit outward speech only for explicit conversational engagement, safety/task requirements, or a high-value, low-frequency calibration question. This avoids reward hacking through attention-seeking chatter. Recent work supports separating novelty and memory representations while retaining their interaction: [Naturalistic novelty and memory representations (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-55833-x) and [predictive coding in the hippocampus (Nature 2026)](https://www.nature.com/articles/s41586-025-09958-0).

### Consolidation and Replay Create Semantic Memory

Memory should have distinct timescales:

| Store | Content | Retention / mutation policy |
| --- | --- | --- |
| Perceptual buffer | short media/audio windows | bounded TTL; source for a newly detected event |
| Working context | current conversation and active task | compact, frequently refreshed |
| Episodic graph | bounded events with evidence | append-first, versioned corrections |
| Semantic graph | stable claims and routines | derived by consolidation; never overwrite raw evidence |
| Procedural memory | user-approved interaction preferences and skills | explicit provenance and easy reset |

During idle periods, consolidation should deduplicate near-identical sightings, strengthen repeatedly supported temporal/causal edges, cluster recurring contexts, derive candidate semantic claims, and retain contradictory evidence. Replay should compose recurring transition patterns into reusable schemas rather than merely compress a transcript. Relevant inspiration: [memory replay and compositional structure (Nature Neuroscience 2025)](https://www.nature.com/articles/s41593-025-01908-3).

## Target Runtime Architecture

```text
Cameras / ReSpeaker / Omnius
            |
            v
  perception_buffer (bounded raw rings)
            |
            +--> vision: detection, segmentation, face, CLIP, pose
            +--> audio: AEC, RMS, VAD, ASR, DOA
            |
            v
       event_segmenter
            |
            v
 entity_resolver + object_resolver
            |
            v
   episodic_graph + vector indexes
            |                    |
            |                    +--> associative_retriever
            v                             |
      idle_consolidator                    v
            |                      context_composer
            +----------------------------> LLM / interaction policy
                                              |
                                              v
                                      Omnius TTS / dashboard
```

### Required Modules

| Module | Responsibility | Initial acceptance criterion |
| --- | --- | --- |
| `memory/schema.py` | SQLite schema and migrations for nodes, edges, evidence, embeddings, revisions | database round-trip preserves provenance and intervals |
| `memory/episodes.py` | event-boundary scoring and episode lifecycle | repeated static frames do not create repeated events |
| `memory/entities.py` | conservative person/object resolution and hypothesis management | identity merge requires configurable multi-signal threshold |
| `memory/retrieval.py` | graph expansion, vector retrieval, quality-aware reranking | every recalled item includes evidence and confidence |
| `memory/consolidation.py` | idle dedupe, replay, semantic-claim derivation | raw episodes remain recoverable after summary creation |
| `cognition/attention.py` | inspectable novelty, relevance, interruption scoring | dashboard exposes score components and rate limits |
| `cognition/context.py` | compact LLM context with source-grounded memory | LLM receives only relevant, cited observations |
| `cognition/calibration.py` | natural clarification and user corrections | asks only when uncertainty is useful and speech is appropriate |

## Current Egg Integration Notes

- The existing SQLite identity library is a suitable starting store, but should become a connector into the graph's `Person`, `Observation`, and `same_as` hypothesis nodes.
- Store face-derived profiles separately from appearance-only tracks. Appearance evidence can assist recall, but must not claim a distinct human identity without sufficiently strong evidence.
- Object memory must use the segmentation mask as part of its feature provenance; a square crop of a detected `person` is not a reliable object sample.
- ASR must be admitted into memory only after actual VAD speech, non-silent RMS, quality checks, and echo/TTS suppression. A transcript from silence is neither a conversation event nor evidence.
- The LLM should receive an event summary and retrieved evidence when the user speaks or when deliberate action is justified. It should not be in the per-frame inference path.

## Implementation Roadmap

- [x] **Phase 1 — Durable episode schema:** create local graph tables, evidence records, temporal validity, revision records, and migrations.
- [x] **Phase 2 — Event segmentation:** bind current camera, ASR, DOA, face, object, and action observations into bounded episodes.
- [x] **Phase 3 — Associative retrieval:** seed retrieval from current perception; run graph activation plus multimodal reranking; expose sources in the dashboard.
- [x] **Phase 4 — Conservative identity/object resolution:** turn existing profiles into graph entities and retain uncertain aliases without premature merge.
- [x] **Phase 5 — Context-grounded interaction:** build an LLM context assembler that supplies only current task, relevant episodes, semantic claims, confidence, and source references.
- [x] **Phase 6 — Idle consolidation:** deduplicate, replay, derive semantic candidates, and keep explicit user corrections authoritative.
- [x] **Phase 7 — Evaluation harness:** collect local consented traces and score retrieval, identity merge/split errors, event-boundary accuracy, false interruptions, latency, and correction retention.

## Evaluation Criteria

The system is improving only when measured behavior improves:

- **Identity:** report precision, false merges, false splits, recall latency, and evidence quality separately.
- **Episode segmentation:** label a small local set of activity traces and calculate boundary precision/recall; static scenes should not inflate event counts.
- **Memory QA:** ask queries about people, objects, interactions, and time; require the returned answer to link to supporting episodes.
- **Social behavior:** measure false proactive utterances, unwanted interruption rate, response relevance, and user correction rate.
- **ASR grounding:** count rejected silent/echo windows and verify that stored utterances have VAD, RMS, and source metadata.
- **Resource use:** track Jetson GPU/RAM, database growth, retrieval latency, and camera/ASR inference latency under continuous operation.

## Safety and Data Governance

People profiles, face embeddings, speech, and behavioral history are sensitive. Keep the system local by default and provide dashboard controls for:

- viewing the evidence and confidence behind each person/object/claim;
- attaching, changing, or removing a human-readable name;
- exporting or deleting a profile and all dependent evidence;
- setting retention windows for raw media, episodes, and semantic claims;
- marking a relationship or memory as incorrect;
- disabling persistent identity memory while retaining short-lived interaction context.

User corrections should create a revision record rather than silently rewrite history. A name is an alias supplied by a person, not evidence that the biometric identity matcher is infallible.

## Literature and Discovery Log

| Source | Relevance | Implementation implication | Status |
| --- | --- | --- | --- |
| [RoboMemory (arXiv 2025)](https://arxiv.org/abs/2508.01415) | Dynamic graph linking episodic, spatial, temporal, and semantic robot memory | use typed multimemory graph and explicit temporal edges | research reviewed; validate locally |
| [Associa (ACL Findings 2025)](https://aclanthology.org/2025.findings-acl.901/) | intuitive association followed by deliberate evidence recall | two-stage graph activation then reranking | research reviewed |
| [HippoRAG (NeurIPS 2024)](https://papers.neurips.cc/paper_files/paper/2024/file/6ddc001d07ca4f319af96a3024f6dbd1-Paper-Conference.pdf) | graph retrieval inspired by associative memory | seeded graph traversal for recall | research reviewed |
| [WAVE (arXiv 2025)](https://arxiv.org/abs/2509.21990) | unified audio, video, and text representation | explore cross-modal alignment only after reliable late fusion | research reviewed; experimental |
| [ES-Mem (arXiv 2026)](https://arxiv.org/abs/2601.07582) | event segmentation for long-horizon memory | episode boundaries instead of transcript chunks | research reviewed; experimental |
| [Spatial scaffolds and associative memory (Nature 2025)](https://www.nature.com/articles/s41586-024-08392-y) | hippocampal association organized by spatial structure | retain camera/place/time context in each episode | research reviewed |
| [Naturalistic novelty and memory (Nature Communications 2025)](https://www.nature.com/articles/s41467-025-55833-x) | novelty and memory representations interact but differ | attention must keep novelty and familiarity components visible | research reviewed |
| [Predictive coding in hippocampus (Nature 2026)](https://www.nature.com/articles/s41586-025-09958-0) | prediction error as a memory/learning signal | include prediction residual in event/attention scoring | research reviewed |
| [Memory replay and compositional structure (Nature Neuroscience 2025)](https://www.nature.com/articles/s41593-025-01908-3) | replay can compose prior structures | perform idle replay into schemas, not just summaries | research reviewed |
| [eMEM (arXiv 2026)](https://arxiv.org/abs/2606.03374) | SQL plus graph/vector/spatial data structures | start SQLite; consider spatial index only after camera calibration | research reviewed; experimental |
| [POLAR (arXiv 2026)](https://arxiv.org/abs/2605.26256) | personalized embodied multimodal graph memory | retain multimodal evidence and personalized links | research reviewed; experimental |
| [C-CLIP (ICLR 2025)](https://mlanthology.org/iclr/2025/liu2025iclr-cclip/) | continual learning considerations for CLIP-like models | avoid unvalidated online finetuning; retain replay/evaluation data first | research reviewed |
| [World Models: A Comprehensive Survey of Architectures, Methodologies, Reasoning Paradigms, and Applications (arXiv 2606.00133, 2026)](https://arxiv.org/pdf/2606.00133) | three-axis taxonomy for embodied world models: functionality (decision-coupled vs. general-purpose), temporal modeling, spatial representation | classifies `WorldStatePredictor` as a lightweight decision-coupled world model; confirms the existing design point rather than requiring a rewrite | research reviewed; validated existing design |
| [A Comprehensive Survey on World Models for Embodied AI (arXiv 2510.16732, 2025)](https://arxiv.org/pdf/2510.16732) | broad taxonomy of embodied world models and evaluation protocols | corroborates the decision-coupled framing above | research reviewed |
| [World Model for Robot Learning: A Comprehensive Survey (arXiv 2605.00080, 2026)](https://arxiv.org/html/2605.00080v1) | survey of world models specifically for robot learning and control | no immediate Egg change; useful reference if manipulation/planning is added later | research reviewed |
| [V-JEPA 2 / V-JEPA-2-AC (arXiv 2506.09985, 2025)](https://arxiv.org/abs/2506.09985) | self-supervised video world model enabling zero-shot robot manipulation via MPC, trained on 1M hours of video + 62 hours of robot data | evaluated and **deferred**: would compete with the Jetson's already-budgeted unified memory (single Ollama model, YOLOE+SAM+CLIP+SFace+ASR+TTS resident); Egg has no manipulator, so MPC-style action planning is not currently applicable | research reviewed; deferred, not adopted |
| [Neural Brain: A Neuroscience-inspired Framework for Embodied Agents (arXiv 2505.07634, 2025)](https://arxiv.org/pdf/2505.07634) | frames an embodied agent as sensing + a tightly coupled perception-cognition-action loop + adaptive short/long-term memory | direct model for `cognition/architecture.py`'s `CognitiveArchitecture`, which explicitly composes attention (perception), prediction-error evaluation (cognition), and evidence association into one perceive/associate loop instead of leaving it implicit across `runtime.py` | research reviewed; adopted as design basis |
| [CraniMem: Cranial Inspired Gated and Bounded Memory for Agentic Systems (arXiv 2603.15642, 2026)](https://arxiv.org/pdf/2603.15642) | bounded, gated memory to prevent unbounded agent context growth | Egg's `MemoryConfig` already bounds perceptual buffers (`buffer_frames_per_camera`, `buffer_ttl_seconds`, `buffer_max_bytes`) and context assembly (`context_max_characters`); no code change needed, already aligned | research reviewed; already aligned |
| [Generate, but Verify: Reducing Hallucination in VLMs with Retrospective Resampling (arXiv 2504.13169, 2025)](https://arxiv.org/pdf/2504.13169) | plain LLM/VLM self-verification has a high no-op/unreliable rate and should not silently overwrite outputs | the new object confidence-audit pass (`OmniusClient.audit_object_label`) only *routes* a profile to the existing image-grounded Ornith VLM correction path; it never rewrites a label itself | research reviewed; shaped audit routing design |

Two consequences of this pass are recorded explicitly so they don't read as oversights later: (1) Egg's existing `WorldStatePredictor` is intentionally a small decision-coupled prediction model rather than a generative world model — this matches, not lags, current best practice for a resource-constrained edge device with no manipulator. (2) A generative video world model (V-JEPA 2 class) was evaluated and consciously deferred, not silently skipped, because it would contend with the Jetson AGX Orin's already tightly budgeted unified memory (see README: one Ollama model, one parallel request, 4096-token context) and Egg currently has no action space for it to plan over.

## Query Log

Use these queries for periodic literature refreshes and implementation-specific follow-up:

```text
"episodic event segmentation multimodal embodied agent graph memory 2025 2026"
"hippocampal pattern separation pattern completion multimodal memory architecture"
"cross modal associative memory audio vision text embeddings long horizon agent"
"embodied agent dynamic knowledge graph episodic semantic procedural memory"
"novelty prediction error attention episodic memory naturalistic scenes"
"memory consolidation replay graph retrieval LLM agent temporal provenance"
"person re-identification uncertainty calibration graph memory embodied robot"
"segmentation mask object memory cross modal grounding user naming robot"
"active information seeking interruption cost social robot memory calibration"
```

When adding a discovery, include the publication date, source type (peer reviewed, preprint, or implementation report), a one-sentence finding, the proposed Egg impact, and a validation test. Do not treat a preprint as production evidence without a local benchmark.
