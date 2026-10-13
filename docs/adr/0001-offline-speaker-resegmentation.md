# ADR 0001: Offline speaker re-segmentation contract

- **Status:** Proposed
- **Date:** 2026-10-13
- **Feature:** `vad-diarization`, task T8 (implemented by T9, consumed by T10)
- **Deciders:** project owner (product decisions), diarization maintainers

## Context

### What exists today (HEAD `f366c6a`)

Diarization is an online clustering pass per speech-aligned chunk, followed by a merge-only
offline refinement when the session stops.

| Concern | Current behaviour | Evidence |
|---|---|---|
| Embedding log | `Option<Vec<(Vec<f32>, SpeakerId)>>`: vector plus online id. No time, no chunk, no span | `crates/echo-diarize/src/online_diarizer.rs:35` |
| Log writes | One entry per embedded span (segmenter path) or per chunk (whole-chunk path) | `crates/echo-diarize/src/online_diarizer.rs:187-190`, `:212-215` |
| Log lifetime | Cleared by `reset`, dropped with the diarizer when the pipeline task ends | `crates/echo-diarize/src/online_diarizer.rs:239-244` |
| Domain port | `Diarizer::refine_offline(&self) -> Vec<(SpeakerId, SpeakerId)>`, default no-op | `crates/echo-domain/src/ports/diarizer.rs:144-159` |
| Spans | `SpeakerSpan { speaker_id, slot, start_sample, end_sample }`, chunk-relative, 16 kHz | `crates/echo-domain/src/ports/diarizer.rs:39-53` |
| Segmenter hint | `SpeakerSegment::local_speaker` is window-local (pyannote 5 s windows) | `crates/echo-domain/src/ports/segmenter.rs:41-45` |
| Attribution | Each ASR segment takes the span with the largest overlap, then shifts to session time | `crates/echo-app/src/use_cases/streaming/mod.rs:865-880`, `:987` |
| Stop sequence | Final flush, then `refine_offline`, then `SpeakersMerged` (only if non-empty), then `Stopped` | `crates/echo-app/src/use_cases/streaming/mod.rs:462-511` |
| Pause | Audio is discarded while paused and is not part of session time; the VAD is reset on resume; the diarizer is not | `crates/echo-app/src/use_cases/streaming/mod.rs:428-452` |
| Event | `TranscriptEvent::SpeakersMerged { merges: Vec<SpeakerMerge> }` | `crates/echo-domain/src/entities/streaming.rs:186-201`, `:253-258` |
| Recorder | Applies each merge with `merge_speakers`, best effort, one transaction per pair | `crates/echo-app/src/services/meeting_recorder.rs:263-296` |
| Storage port | `MeetingStore::merge_speakers(absorbed, surviving)` only | `crates/echo-domain/src/ports/storage.rs:106-124` |
| SQLite | Rewrites `segments.speaker_id`, deletes the absorbed row; the absorbed `label` is lost | `crates/echo-storage/src/sqlite.rs:202-251` |
| Schema | `speakers(id, meeting_id, slot, label NULL = never renamed, UNIQUE(meeting_id, slot))`; one nullable `segments.speaker_id` | `crates/echo-storage/migrations/0002_speaker_slot.sql:17-23`, `0001_initial.sql:32-40` |
| Event fan-out | The drain loop awaits `recorder.record(evt)` before sending the event to the frontend | `src-tauri/src/commands/streaming.rs:262-283` |
| Frontend | `speakersMerged` is ignored; the meeting is refetched after `stopped`. Live lines are per chunk (`speakerSlot`, capped at 500) | `src/hooks/useRecordingSession.ts:249-254`, `:186-198`, `:53` |
| Rename | Renames write `speakers.label` in storage only; the live diarizer is not told | `crates/echo-app/src/use_cases/rename_speaker.rs:66`, `src-tauri/src/commands/meetings.rs:108` |
| Wiring | The app always enables the log and the merge-only refine | `src-tauri/src/commands/mod.rs:484-487` |

There is no per-segment speaker edit in the product today. A speaker row with a non-null
`label` is the only user-authored diarization data.

### Why this must change

- The calibrated multilingual CAM++ still over-counts speakers: 4.15 estimated against
  2.80 true on the Spanish test set (`docs/spikes/2026-10-embedder-spike.md:103`). The
  T6 baseline showed the merge-only refine as a no-op at the old threshold
  (`docs/spikes/2026-10-der-baseline.md`).
- Merging cannot fix a span that was attached to the wrong speaker. It can only remove
  speakers. VBx (VB-HMM over PLDA-projected embeddings, AHC initialisation; task T10) produces a
  new label for every embedding. Some segments move between speakers, and speakers can be
  created or retired.
- VBx needs the embedding sequence in time order, with durations and discontinuities. The
  current log has none of these.
- User decision (2026-10-12): VBx works on the timestamped embedding sequence kept in memory for
  the session. No audio is retained.

## Decision

### D1. In-memory timestamped embedding log (echo-diarize)

The embedding log stays private to the diarizer adapter (`OnlineDiarizer`). It is never part of
a domain type that crosses a crate boundary toward storage, IPC or MCP.

```rust
/// One embedded span. Lives only in OnlineDiarizer memory.
struct EmbeddingLogEntry {
    spans: SmallVec<[SpanRef; 1]>, // (chunk_index: u32, span_index: u16); >1 only after compaction
    start_ms: u32,                 // session time = chunk offset_ms + span start
    end_ms: u32,                   // session time, exclusive
    epoch: u16,                    // 0 at start, +1 after each resume
    local_speaker: Option<u8>,     // segmenter hint, window-local; None on the whole-chunk path
    online_speaker: SpeakerId,     // id assigned by the online cluster at the time
    embedding: Box<[f32]>,         // as produced by the embedder (dim 192/256/512)
}
```

- **Segment ids are not in the log.** ASR runs after `assign_spans`, so the diarizer never sees
  segment ids. The log is keyed by `SpanRef`. A separate **attribution ledger** in `ChunkEmitter`
  (echo-app) records, for each emitted segment, `(segment_id, chunk_index, start_sample,
  end_sample)` in chunk-local 16 kHz samples, together with each chunk's spans. This is the data
  `dominant_span` already uses. The ledger holds no embeddings (about 40 B per segment).
- **Timestamps.** The pipeline passes a `ChunkContext { chunk_index, offset_ms, epoch }` to the
  diarizer through a new port method (D2). Session time excludes paused audio, as it does today.
- **Memory bound.** At 192 dims, an entry costs about 0.85 KB (768 B vector plus about 80 B of
  metadata and allocation overhead). A 3 h meeting with about 80 % speech has about 8 600 s of
  speech. With spans of about 2.5 s (segmenter on) that is about 3 500 entries (about 3 MB).
  The worst case, 1 s spans, gives about 8 600 entries (about 7.3 MB at 192 dims, about 18.5 MB
  at 512 dims).
  - Hard cap: **16 384 entries** (about 14 MB at 192 dims, about 35 MB at 512 dims).
  - When the cap is reached, the log **compacts** in place. Pairs of temporally adjacent entries
    with the same `online_speaker` and `epoch` are coalesced into one entry. Its embedding is the
    duration-weighted mean, renormalised. Its time range is the union, and its `spans` list is
    the concatenation. This halves the count while still covering every span. No entry is
    silently dropped.
- **Lifetime.** The log is created with the diarizer for one session and cleared by `reset`. It
  is dropped when `run_pipeline` returns (the diarizer is owned by the pipeline task). It is
  **never** written to disk or the database, never serialised, never put in an event, never
  sent over IPC or MCP, and never logged.
- **Privacy rationale.** Speaker embeddings are biometric-adjacent: they can re-identify a voice
  across recordings. The product promise is local, private transcription with no audio retention
  (`docs/ARCHITECTURE.md` §9). Keeping the vectors for one session in RAM is the minimum VBx
  needs.

### D2. Domain port: a reassignment plan, merge as a special case

All changes to `crates/echo-domain/src/ports/diarizer.rs` are additive. Default methods keep
fakes, the bench decorator (`crates/echo-proto/src/bench_der.rs:537-539`) and older adapters
compiling.

```rust
pub struct ChunkContext { pub chunk_index: u32, pub offset_ms: u32, pub epoch: u16 }

/// Default: forwards to assign_spans (context ignored).
async fn assign_spans_at(&mut self, samples: &[Sample], ctx: ChunkContext)
    -> Result<Vec<SpeakerSpan>, DomainError>;

/// Default: wraps refine_offline() as RefineOutcome::Merges (or Unchanged if empty).
fn refine_plan(&self, budget: RefineBudget) -> Result<RefineOutcome, DomainError>;

pub struct RefineBudget { pub deadline: std::time::Instant }

pub enum RefineOutcome {
    Unchanged,
    /// Legacy merge-only result: (absorbed, surviving).
    Merges(Vec<(SpeakerId, SpeakerId)>),
    /// Span-level relabelling.
    Relabel(SpanRelabelPlan),
}

pub struct SpanRelabelPlan {
    /// Final speaker for every logged span: (chunk_index, span_index) -> id.
    pub spans: Vec<(SpanRef, SpeakerId)>,
    /// Speakers the plan introduces (fresh SpeakerId, next free slot, label None).
    pub created: Vec<Speaker>,
    /// Online speakers left with no span, each with the id that took most of its duration.
    pub retired: Vec<SpeakerRetirement>, // { speaker, majority_recipient: Option<SpeakerId> }
}
```

- `refine_offline` stays. It is the merge-only path, and its doc points to `refine_plan`. A
  `Merges` outcome is converted by the pipeline exactly as today.
- **Stable ids.** VBx clusters are mapped back to online `SpeakerId`s by maximum-overlap
  (duration-weighted Hungarian) matching. Matched clusters keep the online id, so they also keep
  the slot, the colour and any user label. Only unmatched clusters receive a fresh id.
- **Cooperative timeout.** VBx checks `budget.deadline` between iterations and returns
  `DomainError` when the deadline passes. It never returns a partial plan.
- Inside echo-diarize, `OnlineDiarizer` delegates the re-clustering to a crate-private
  `OfflineReclusterer` trait (`fn recluster(&self, log, speakers, budget) -> Result<Labels>`).
  The current AHC refine becomes one implementation, and T10's VBx is the other.

### D3. Pipeline: turning a span plan into a segment plan

At stop, after the final flush (the existing ordering), `run_pipeline`:

1. Moves the diarizer into `tokio::task::spawn_blocking` and calls `refine_plan` with
   `deadline = now + refine_timeout`. A `JoinError` (panic) counts as a failure.
2. On `Relabel`, it relabels the ledger's spans and re-runs `dominant_span` and
   `chunk_dominant_speaker` for every ledger segment and chunk. Segments without spans stay
   `None`: there is no embedding to judge them by.
3. It emits **`TranscriptEvent::SpeakersReassigned`** with only the changed rows. It emits
   nothing when no segment changes. It still emits `SpeakersMerged` for a `Merges` outcome.
4. Emits `Stopped` in every case.

On error or timeout it logs `warn!` with the session id, the elapsed time and the error kind.
It never logs vectors. No reassignment is emitted, so the **online result stands**.

**Ordering.** ASR is synchronous per chunk inside the same task, so no ASR segment can arrive
after the plan. This is an invariant T9 must test: the plan covers every segment emitted before
it. If ASR ever becomes asynchronous, the pipeline must join outstanding ASR work before
`refine_plan`.

### D4. Reassignment event and IPC payload (echo-domain, specta-exported)

```rust
SpeakersReassigned {
    session_id: StreamingSessionId,
    plan_id: Uuid,                       // UUIDv7, correlates logs and makes replays detectable
    created: Vec<SpeakerRef>,            // { id, slot }
    retired: Vec<SpeakerRetirement>,     // { speaker, majority_recipient: Option<SpeakerId> }
    segments: Vec<SegmentSpeaker>,       // { segment_id, speaker_id: Option<SpeakerId> }
    chunks: Vec<ChunkSpeaker>,           // { chunk_index, speaker_id, speaker_slot } (live pane)
}
```

- The payload holds ids, slots and indices only. It never holds embeddings, scores or text.
- It is emitted after the last `Chunk` and before `Stopped`, like `SpeakersMerged`. The drain loop
  applies it to storage before the frontend receives it (`src-tauri/src/commands/streaming.rs:268-278`).
- **Live pane:** `useRecordingSession` patches `speakerSlot` on lines whose `chunkIndex` is in
  `chunks`. Lines already evicted by the 500-line cap are ignored.
- **Stored meeting view:** no change. The meeting is refetched after `stopped`. If the in-flight
  meeting is open in `MeetingDetail`, it refetches when it receives `speakersReassigned`.
- **Idempotence:** every field is a set-to-value and never a delta. Applying the event twice
  gives the same state. The UI may ignore a `plan_id` it has already applied.
- **Compatibility:** `SpeakersMerged` and `SpeakerMerge` are kept unchanged for the merge-only
  path. Regenerating the TypeScript bindings adds the new variant. The existing `switch` has no
  default branch, so an older UI simply ignores the event.

### D5. `MeetingStore::reassign_speakers` port

```rust
async fn reassign_speakers(
    &self,
    meeting_id: MeetingId,
    plan: &SpeakerReassignment, // created, retired, segments (same shapes as the event)
) -> Result<ReassignOutcome, DomainError>; // { segments_updated, speakers_created, speakers_deleted, labels_moved }
```

- **One transaction, all or nothing.** In order: verify the meeting exists; insert `created`
  rows (`INSERT ... ON CONFLICT(id) DO NOTHING`, so replay is safe); update every listed
  `segments.speaker_id` scoped by `meeting_id`; apply the label rule; delete each retired speaker
  that has no segments left and no label. The whole transaction rolls back (`Err`) if a listed
  segment or a referenced speaker does not belong to the meeting. Slots of new speakers come
  from the plan. They are never reused from retired rows, and `UNIQUE(meeting_id, slot)` stays
  intact.
- **User labels (rule: never overwrite a non-null label).**
  - A matched speaker keeps its id, and therefore its label.
  - A retired speaker with a label passes its label to `majority_recipient` only if that row's
    label is `NULL`. Otherwise the labelled row is **kept** with zero segments, so no
    user-authored name is ever destroyed. Owner decision 2026-10-13 (Q1): adopt this default.
  - Labels are read inside the transaction. A rename that races the stop is serialised by
    SQLite: it lands either before the transaction (and the rule then sees it) or after it (and
    applies to a row that still exists, or returns `false` if the row was deleted).
- The recorder handles `SpeakersReassigned` with a single `reassign_speakers` call. On `Err` it
  logs a warning and continues to `Stopped`. The rollback leaves the online result persisted
  intact (fail-safe).
- **No migration.** The existing schema expresses everything needed. Per-segment user edits do
  not exist yet. If they are added later, a `segments.speaker_locked` column (sqlx migration
  `0006`) would make the store skip locked rows. That is out of scope here (YAGNI).
- **FTS5 cost.** The `segments_au` trigger (`crates/echo-storage/migrations/0003_fts5.sql:60`)
  fires on any `UPDATE`, so every reassigned segment's text is re-indexed. `merge_speakers`
  already pays this cost today, and it is acceptable for thousands of rows. Narrowing the
  trigger to `AFTER UPDATE OF text` would be a separate, optional migration.
- `merge_speakers` stays for the legacy path. Since T9 it applies the same label rule, so it
  no longer drops the absorbed label.

### D6. Feature flag and how T10 plugs in

- `OfflineRefineMode { Off, MergeOnly, Resegment }` lives in `echo_diarize::presets`, one per
  embedder family. It is selected in `build_diarizer` (`src-tauri/src/commands/mod.rs:484-487`).
  Debug builds can override it with `ECHO_DIAR_REFINE=off|merge|reseg`, following the existing
  `ECHO_DIAR_SIM_THRESHOLD` pattern. `echo-proto bench der` gains `--refine-mode`.
- **Default stays `MergeOnly`** until T10 shows a DER improvement on the held-out test sets with
  `bench der`. Flipping the default is a one-line preset change in its own commit.
- `Resegment` requires a PLDA asset that matches the embedder. If the asset is missing, the
  adapter falls back to `MergeOnly` and logs one `info!` line.
- T9 delivers the contract with the AHC reclusterer behind `Resegment`. This proves the plan
  end to end without VBx. T10 adds `VbxReclusterer` (PLDA load, AHC init, VB-HMM) as the
  `Resegment` implementation.

### D7. Pause/resume, segmenter on/off, very short meetings

- **Pause/resume:** `epoch` increments on `Resumed`. VBx must not apply an HMM speaker-transition
  penalty across an epoch boundary (it treats each epoch as a separate run). Paused audio has no
  entries.
- **Segmenter on:** entries are sub-chunk spans. `local_speaker` is a hint only. It is
  window-local, so it is never used as an identity across windows.
- **Segmenter off:** one entry per chunk (4-20 s). VBx can only move whole chunks. A chunk with
  two speakers cannot be split, because no audio is retained.
- **Very short meetings:** `refine_plan` returns `Unchanged` when the log has fewer than
  `min_entries` (initially 4) or less than `min_speech_ms` (initially 10 000) of logged speech.
  T10 calibrates both values. A meeting with zero entries never emits a reassignment event.
- **Diarization off or failed chunks:** there is no diarizer, so there is no log and nothing to
  do. Chunks whose `assign_spans` failed have no spans and keep `speaker_id = None`.

### D8. Non-functional requirements (FURPS+)

| Area | Requirement |
|---|---|
| Functionality | Reassignment covers every segment that has spans. The legacy merge path behaves exactly as today when the mode is `MergeOnly`. |
| Usability | Stop never blocks the UI. `Stopped` follows within the refine timeout. Speaker colours and labels of matched speakers do not change. |
| Reliability | Any refine or storage failure leaves the online result persisted (fail-safe). The storage write is atomic. The event is idempotent. |
| Performance | Target: refine for a 3 h session (≤ 8 600 entries) completes in **p95 ≤ 5 s** on an Apple M1 Pro. Hard `refine_timeout` is **20 s**. AHC initialisation runs on at most 4 096 representatives (condensed f32 matrix of about 33.5 MB, transient). Above that, it runs on a strided subsample, and the remaining entries go to the nearest centroid. T10 must measure these figures. |
| Supportability | Logs record counts, timings, `plan_id` and the outcome kind, never vectors. `bench der --refine-mode` reproduces any mode offline. |
| + Privacy | The log is RAM-only, per session, at most about 35 MB, and dropped at stop. No embedding is ever persisted or crosses IPC or MCP. |

### Component sketch

```mermaid
flowchart LR
  subgraph echo-app [echo-app: run_pipeline]
    CE[ChunkEmitter<br/>attribution ledger<br/>segment ids + spans, no vectors]
  end
  subgraph echo-diarize
    OD[OnlineDiarizer<br/>EmbeddingLog in RAM]
    RC[OfflineReclusterer<br/>AHC (T9) / VBx+PLDA (T10)]
  end
  CE -- assign_spans_at(ctx) --> OD
  OD -- refine_plan(budget) --> RC
  RC -- SpanRelabelPlan --> CE
  CE -- SpeakersReassigned (ids only) --> REC[MeetingRecorder]
  REC -- reassign_speakers (1 tx) --> DB[(SQLite)]
  REC --> UI[Frontend: live pane patch / meeting refetch]
```

## Consequences

**Positive**

- VBx can correct mis-attributed spans and the speaker count, not only merge speakers. This is
  the lever T7 identified against the over-count.
- No audio is retained. The only new retained state is a bounded, per-session vector log in RAM.
- Storage gains an atomic, idempotent bulk operation that also protects user labels. The
  legacy merge path follows the same label rule since T9.
- The port changes are additive. Fakes, the bench decorator and the merge-only path keep working.

**Negative / costs**

- Stop latency grows by the refine time, bounded by the timeout. A long meeting may wait a few
  seconds before `Stopped`.
- The granularity is fixed at capture time: VBx cannot split a span or a segmenter-off chunk.
  The BUTSpeechFIT reference uses dense 1.5 s windows. This design starts from variable-length
  spans and may need extra window embeddings inside long spans (Q3).
- Two refine paths (merge and reassign) coexist until the flag is retired.
- A labelled speaker can survive with zero segments under the conservative label rule. The UI
  must tolerate such a speaker.

## Alternatives

| Alternative | Pros | Cons | Verdict |
|---|---|---|---|
| **A1. Retain audio for the session (RAM or temp file) and re-embed at stop** | Dense windows; reprocessing is free of capture-time choices; the closest to the VBx reference | Violates the no-audio-retention promise; RAM use of about 1.1 GB per 3 h at 16 kHz f32 (or disk exposure); a larger breach surface | Rejected (user decision 2026-10-12) |
| **A2. Persist embeddings in SQLite to allow later re-clustering** | Re-runs after model upgrades; cross-meeting speaker identity | Stores biometric-adjacent data at rest; needs encryption and a retention policy; scope creep | Rejected; revisit only with an explicit enrolment feature and consent |
| **A3. Express VBx output as merges only (keep the current port)** | No new event, port or storage operation | Cannot move segments between speakers; loses most of the VBx benefit | Rejected |
| **A4. Change `refine_offline`'s return type in place instead of adding `refine_plan`** | One method, no duplication | Breaks every implementer at once; no gradual migration | Rejected in favour of an additive default |
| **A5. Keep the log in echo-app (pipeline) instead of the diarizer** | One place for all session state | Puts embeddings into the application layer and the event path, so leaks become more likely; couples echo-app to vector maths | Rejected; the pipeline keeps only the vector-free ledger |
| **A6. Rewrite segments with `DELETE` + re-`INSERT`** | Simple SQL | Changes the rowids the FTS5 external-content index is keyed on, fires the delete and insert triggers for every row, and is not idempotent | Rejected |
| **A7. Zeroise embedding buffers on drop (`zeroize`)** | Defence in depth against memory scraping | A new dependency; an attacker who can read process memory already sees the live audio | Not adopted now; cheap to add later |

## Threat and privacy note

- **Data exposure (OWASP A01/A02-style).** Embeddings never leave process memory: they are not
  persisted, serialised, sent over IPC or MCP, or included in crash payloads. T9 adds a test that
  the serialised `SpeakersReassigned` contains no float arrays.
- **Logging (OWASP A09).** `EmbeddingLogEntry` must not derive `Debug` over the vector. It gets a
  manual `Debug` that prints only the dimension. `tracing` fields carry counts and ids only.
- **Integrity.** The store validates that every id in the plan belongs to `meeting_id` and
  rejects the whole plan otherwise. This prevents a malformed or replayed event from touching
  another meeting.
- **Availability.** A refine bug that loops or panics is bounded by the deadline, `spawn_blocking`
  and the fail-safe, so a meeting is never lost at stop.
- **LLM/MCP surfaces.** These surfaces see only the resulting `speaker_id` and labels in the
  database. Nothing new is exposed to them.

## Open questions (product decisions for the owner)

1. **Q1. Labelled speaker retired by VBx.** Resolved 2026-10-13: never overwrite a label; the
   labelled row is kept with zero segments, and its label moves only to a recipient whose label
   is `NULL`.
2. **Q2. Show the user a diff?** Resolved 2026-10-13: show a one-line notice after stop
   ("speaker attribution refined: N segments moved, M speakers merged") without undo. Nothing
   extra is persisted; the notice is built from the `SpeakersReassigned` payload.
3. **Q3. Extra window embeddings.** Resolved 2026-10-13: allowed. T10 may compute extra
   embeddings (for example 1.5 s windows inside spans longer than 3 s) during the session, at
   roughly twice the embed cost. Condition: T10 must also bound the merge-only refine to at most
   4 096 representatives, like the AHC/VBx path, so peak memory stays at about 35 MB for the log
   plus about 35 MB transient at stop, regardless of meeting length or window density.
4. **Q4. Default flip criterion.** What DER gain on the held-out sets justifies making
   `Resegment` the default? For example, a weighted DER at least 0.5 points lower and an
   over-count no worse than today.

## Implementation notes (T9)

T9 implements the contract with the AHC reclusterer behind `OfflineRefineMode::Resegment`.
The default stays `MergeOnly` for every embedder family. Where the code differs from the
sketches above:

- **Log entry.** `spans` is a `Vec<SpanRef>` instead of a `SmallVec`, to avoid a new
  dependency. The log lives in `echo_diarize::embedding_log`; `EmbeddingLogEntry` has a manual
  `Debug` that prints the dimension only, and no `Serialize`.
- **Compaction fallback.** When the full log has no time-adjacent pair with the same online
  speaker and epoch (speakers alternate on every span), each entry is coalesced with the next
  entry of the same speaker and epoch instead, so the log still shrinks and no span is dropped.
- **AHC.** `resegment::AhcReclusterer` runs average-linkage AHC over the log entries (one
  cluster per entry, nearest-neighbour chain with Lance-Williams updates, one condensed f32
  matrix), cut at the family's refine `merge_threshold`. Above 4 096 entries it runs on a strided
  subsample and the other entries join the nearest centroid. It checks the deadline between
  merges.
- **Segment-level retirements.** The pipeline retires every speaker that the plan retired or
  that lost its last segment after `dominant_span` is re-run, with the majority recipient taken
  from where its segments went (the plan's recipient is the fallback). A recipient the store
  does not know is dropped to `None`. Only created speakers that receive a segment or take over
  a retired speaker are sent, so the store never rejects a plan for a row it never persisted.
- **`merge_speakers`.** Fixed in T9 with the same label rule (`retire_speaker` in
  `echo-storage`), so the merge-only path no longer destroys a user label.
- **Timeout.** The pipeline waits the 20 s deadline plus a 1 s grace for the blocking task. A
  refine that ignores its deadline is detached; its result is discarded.
- **UI.** The live pane patches chunk slots for `speakersReassigned` and, now, `speakersMerged`,
  and shows the Q2 notice. `MeetingDetail` does not refetch on `speakersReassigned`; the
  refetch after `stopped` already shows the persisted result.
- **Known limit (unchanged).** The merge-only refine still builds a full `n × n` similarity
  matrix over the log. At the 16 384-entry cap that is about 1 GB; T10 should bound it the way
  the AHC reclusterer is bounded.

## Implementation notes (T9b review fixes)

Decisions taken while fixing the T9 review warnings:

- **Non-finite embeddings.** An embedding with a NaN or infinite value is still assigned online
  but never logged. The AHC also maps any non-finite distance to the largest cosine distance
  (2.0), so such an entry stays a singleton, and it returns `DiarizationFailed` instead of
  panicking if it ever finds no merge candidate. The strided nearest-centroid pass checks the
  deadline every 256 entries.
- **Hard cap, strictly.** The log never holds more than `EMBEDDING_LOG_CAP` entries. Compaction
  runs three O(n) passes: coalesce time-adjacent entries of the same speaker and epoch; if the log
  is still above three quarters of the cap, pair each entry with the next unpaired entry of the
  same speaker and epoch in one pass (a hash map of pending entries); if nothing could be
  coalesced (every entry has its own speaker and epoch), drop the oldest entry, whose spans then
  keep their online speaker. The first two passes usually free a large fraction of the cap, so
  the cost is amortised over many pushes. This replaces the quadratic search of the T9 fallback.
- **Speech duration.** A coalesced entry keeps `speech_ms`, the sum of its spans' speech, apart
  from its time range, which can cover other speakers' turns. Hungarian weights, the
  `min_speech_ms` floor, the majority recipient and the coalescing weights all use `speech_ms`.
- **Refine mode is the single source of truth.** `OnlineDiarizer::with_refine_mode` always wins,
  whatever the builder order; `with_offline_refinement` only selects `MergeOnly` when no mode was
  set. The re-segmentation AHC cut is its own preset value (`EmbedderFamily::resegment_threshold`,
  `with_resegment_threshold`), initially equal to the merge-only threshold, so the two passes can
  be calibrated apart. `bench der --refine-threshold` sets both.
- **Deadlines.** `RefineBudget::check` returns the documented `DiarizationFailed` error. The
  default `Diarizer::refine_plan` and the merge-only path check it before and after the
  uninterruptible `refine_offline`. The pipeline computes the deadline and the join wait with
  overflow-safe arithmetic, so even `Duration::MAX` ends in `Stopped`. The DER bench decorator
  turns a plan returned after the deadline into an error, so it never scores a plan the app would
  have discarded.
- **Payload consistency.** `retired` is sorted by speaker id. A created speaker that owns no
  segment is not sent, and chunk rows naming it are dropped (those live lines keep their online
  colour), so the event never names a speaker the store was not told to create. A plan that moves
  no segment still emits nothing, even if a chunk's dominant speaker changed: chunk rows only
  recolour live lines.
- **Store integrity.** `reassign_speakers` returns `Invariant` for a retired speaker or a created
  id that belongs to another meeting. An id unknown everywhere is a replay and is skipped.
- **Recorder retry.** A `Storage` error from `reassign_speakers` (for example SQLite busy) is
  retried once after 100 ms; an `Invariant` or `NotFound` error is not. Logs carry ids and counts,
  never vectors.
- **Empty labelled speakers.** Under the Q1 label rule, a labelled speaker can remain with zero
  segments. The speakers panel already lists every stored speaker with its talk-time share, so it
  shows such a speaker at 0 % where it can still be renamed; no UI change was needed.
- **Live pane merges.** `applySpeakerMerges` follows merge chains to the final survivor and
  leaves speakers in a merge cycle unchanged.
