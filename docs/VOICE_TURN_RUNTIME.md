# Voice turn runtime

Egg's voice path adapts the transport-independent structure used by Voryn's
live-call runtime to a local ReSpeaker AEC microphone and `aplay` speaker.
Twilio media framing, carrier marks, call-tree state, and telephony recovery are
intentionally not part of this implementation.

## Adopted structure

- `ConversationTurnController` is the single authority for heard-audio revision,
  floor owner, active playback identity, pending barge identity, and the ordered
  audible transcript.
- VAD onset provisionally stops only the playback attempt that currently owns
  the floor. It does not make a semantic interruption decision.
- Final ASR is bound to the utterance and barge identities created at onset.
- A strict secondary model contract decides whether overlapping speech should
  cancel output. Timing and recent-TTS proximity are diagnostic context, not
  semantic gates.
- A rejected/empty false barge resumes from the estimated unplayed WAV tail; it
  never restarts the logical response from frame zero.
- Every finalized heard-audio turn increments the causal revision. Reasoning and TTS
  check that revision immediately before playback publication, so stale work
  cannot speak.
- Distinct completed utterances use bounded FIFO queues. Overload rejects the
  newest item explicitly in logs/telemetry instead of silently replacing an
  older heard-audio turn.

## Ingress sequence

1. The persistent ReSpeaker reader supplies the DSP/AEC ASR channel even while
   Egg is speaking when `audio.barge_in_enabled` is true.
   The six-channel XVF3000 route uses channel 0 for processed ASR, retains the
   four raw microphones and playback-reference channel at the device boundary,
   polls native VAD/DoA/AEC/AGC/RT60 state, and mirrors floor ownership on the
   12-pixel ring.
2. WebRTC VAD confirms onset with pre-roll and emits an identity-bearing start
   boundary. If playback is active, Egg terminates that `aplay` process and
   retains its source WAV plus estimated resume cursor.
3. The trailing-silence target grows from `vad_hangover_ms` toward
   `vad_hangover_max_ms` as real voiced duration and pause continuations grow.
   `segment_seconds` remains the hard memory/latency cap.
4. Completed audio carries its original, pre-normalization RMS/VAD snapshot into
   the bounded ASR FIFO. Digitally silent WAVs are rejected before the Omnius
   request; source-gate failures and unscored, known silence hallucinations are
   rejected before they can become transcript, memory, or dialogue evidence.
5. ASR failure or an acoustically rejected overlap releases the provisional
   barge and resumes the tail.
6. Admitted nonempty ASR creates the next authoritative heard-audio revision and
   supersedes older in-flight reasoning.

## Egress and interruption sequence

1. A response is synthesized speculatively for an expected heard-audio revision.
2. After synthesis and after acquiring the one speaker lease, the runtime checks
   the revision again before creating a playback identity.
3. On ordinary completion, one logical agent turn is committed to the audible
   ledger.
4. On overlap, playback enters `barge_pending`. The semantic classifier receives
   the exact heard transcript, active agent text, complete ordered ledger, and embodied
   context.
5. Genuine interruption or unavailable control inference uses audio-first
   behavior and permanently discards the paused tail. Echo, background speech,
   and non-substantive backchannels resume only the retained tail.
6. Late completion, cancellation, and duplicate barge callbacks are accepted
   only when their playback/barge identities still own the lifecycle.

## Voryn reference points

- `.aiwg/architecture/adr-001-causal-turn-commit-protocol.md`
- `lib/rawPlaybackLifecycle.js`
- `lib/captions/vad.js`
- `lib/rawAsrTurnScheduler.js`
- `lib/enterpriseAgentConversation.js`
- `lib/twilioRawAudioWs.js`

The matching Egg implementation is in
`egg_companion/cognition/conversation.py`, `egg_companion/adapters/audio.py`,
`egg_companion/adapters/speaker.py`, and `egg_companion/runtime.py`.
