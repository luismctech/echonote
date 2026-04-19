# ADR-0007: Diarization via ONNX embeddings (3D-Speaker ERes2Net) with online clustering

- **Status:** accepted
- **Date:** 2026-04-18
- **Deciders:** Tech Lead, ML Engineer
- **Technical story:** Sprint 1 days 5–6 — pick a speaker-embedding
  model, an inference runtime, and a clustering algorithm so the
  diarizer can land before Sprint 1 closes.

## Context and problem statement

Sprint 1 promises that EchoNote separates "who said what" during a
meeting (`docs/DEVELOPMENT_PLAN.md` §3.2). With the audio capture
(`AudioCapture`), VAD (`Vad`) and ASR (`Transcriber`) ports already in
place, the missing piece is a `Diarizer` that, fed mono 16 kHz PCM,
emits a stable speaker id per chunk and converges on the right number
of speakers without offline post-processing.

The decision window is now: the `Diarizer` port shipped on day 5
(commit `015e62b`); the day-6 task is to wire a real adapter behind it.
Picking the model and runtime up-front avoids re-doing the adapter when
we add UI affordances on day 7.

## Decision drivers

- **Quality target.** `docs/DEVELOPMENT_PLAN.md` §4.3 asks for
  diarization purity ≥ 0.85 on a 2-speaker call. That puts a floor on
  embedding quality (ERes2Net or comparable) — an MFCC + GMM baseline
  would not clear it.
- **Latency budget.** Embedding has to run inside the streaming
  pipeline alongside Whisper, which already dominates CPU. We want
  embedding latency < 50 ms per ~3 s chunk on Apple Silicon (≈ 0.02
  RTF) so it stays in the noise.
- **Local-first / privacy.** No network calls, no cloud APIs.
  Everything runs on the user's machine.
- **Distribution.** No extra system dependencies. Pure Rust where
  possible; if a binding is needed it must build cleanly into the
  signed Tauri bundle on macOS, Windows and Linux without a sidecar
  process.
- **Streaming-friendly.** The diarizer must label chunks as they
  arrive — offline agglomerative clustering on the entire recording is
  ruled out.
- **Bounded memory.** Long meetings (2 h+) cannot leak O(n²) state on
  the chunk count.
- **Permissive licence on both code and weights** so we can statically
  link and redistribute without copyleft strings.

## Considered options

### Embedding model

1. **3D-Speaker ERes2Net (English VoxCeleb), exported to ONNX.**
   192-dim embedding, ~26 MB, opset 13, no recurrent ops.
2. **WeSpeaker ResNet-34 / CAM++.** Comparable architecture and size,
   also available as ONNX from the same mirror.
3. **NVIDIA TitaNet (NeMo).** State-of-the-art quality but ~100 MB and
   the open-source export depends on NeMo's `pre/post`-processing ops
   that tract does not implement.
4. **pyannote 3.1 segmentation + embedding.** Higher quality on
   diarization-error-rate benchmarks, but the segmentation model needs
   a Python runtime in practice; mixing PyTorch into the bundle is not
   on the table (see ADR-0003).

### Inference runtime

A. **`tract-onnx` (pure Rust, used for Silero VAD).**
B. **`ort` (Rust bindings to onnxruntime, ships a C++ shared library).**
C. **`candle` (HuggingFace's pure-Rust ML).** No native ONNX import.

### Clustering

I. **Online threshold-based clustering.** O(k) state, single pass,
   labels emitted as chunks arrive.
II. **Offline agglomerative clustering** at the end of the meeting.
III. **Online VBx / spectral.** Higher accuracy in the literature, but
     the implementation cost and the offline post-processing it
     normally requires push it past Sprint 1's budget.

## Decision outcome

**Chosen options:**

- **Model.** 3D-Speaker ERes2Net (English VoxCeleb), ONNX, ~26 MB,
  192-dim embedding. Mirrored on Hugging Face by `csukuangfj` (the
  sherpa-onnx maintainer); upstream lives on ModelScope. Pre-processing
  is Kaldi-style 80-bin log-mel filterbank with per-bin CMN,
  implemented via the `mel_spec` crate which matches the model's
  `feature_normalize_type = global-mean` metadata.
- **Runtime.** `tract-onnx`, same crate already powering the Silero
  VAD adapter. The model uses only basic ops (Conv, Gemm, Add, Relu,
  Sigmoid, …) — no LSTM, no GatherND, no Resize — so tract loads and
  optimises it without falling back to a generic interpreter.
- **Time dimension.** Pinned to 300 fbank frames (~3 s) so tract can
  fully optimise the graph. Shorter chunks are right-padded with
  zeros, longer ones are centre-cropped. Calibration on the supplied
  fixtures showed 200 frames roughly halves the same-speaker cosine
  similarity vs. 300, while going beyond 300 adds latency without
  measurable gain.
- **Clustering.** Online threshold-based with a configurable
  similarity floor and a `max_speakers` cap (`OnlineCluster`,
  `crates/echo-diarize/src/cluster.rs`). Centroids are running L2-
  normalised means.

## Consequences

### Positive

- Single static crate (`tract-onnx`) for both VAD and embedding —
  no extra binary dependency, no `ort` C++ runtime to bundle and sign.
- Embedding cost on M-series CPU: ~10 ms per 3 s chunk in our day-6
  measurements — well inside the 50 ms budget.
- Clear separation: `SpeakerEmbedder` trait isolates the model from
  the clustering algorithm, so swapping ERes2Net for WeSpeaker (or a
  larger ERes2Net variant) is a one-file change.
- Online clustering keeps memory at O(k) where k is the speaker count,
  not O(n) on chunk count. Two-hour meetings cost the same as
  ten-minute ones.
- `OnlineDiarizer` already passes a synthetic two-speaker E2E test
  (`eres2net::tests::online_diarizer_clusters_two_speakers_correctly`)
  with five alternating windows assigned to exactly two clusters.

### Negative

- The English VoxCeleb checkpoint is exactly that — English. Spanish
  meetings (our target launch market) will get noisier embeddings
  until we add the multilingual `3dspeaker_speech_eres2net_sv_zh_en_*`
  checkpoint. Documented as a Sprint 2 follow-up; the adapter is
  already model-path-driven so swapping is a one-line config change.
- Threshold-based online clustering can over- or under-segment when
  the per-meeting acoustic conditions drift far from the calibration
  fixtures. Offline re-clustering pass remains an option if the
  Sprint 1 evaluation flags it (issue would be filed against the
  `OnlineCluster` strategy, not the embedder).
- Pinning T to 300 means the diarizer cannot meaningfully embed
  bursts shorter than ~0.5 s. Acceptable: short utterances seldom
  carry stable speaker identity anyway, and the `Diarizer::assign`
  contract already returns `None` for sub-min chunks.

### Neutral

- The embedding crate now pulls `mel_spec` (pure Rust, ~6 deps) into
  the workspace. No build-system impact.
- The download script grew an `embed` subcommand, parallel to the
  existing `vad` one; the model lives at
  `./models/embedder/eres2net_en_voxceleb.onnx`.

## Pros and cons of the options

### Embedding model

#### 1. 3D-Speaker ERes2Net EN VoxCeleb ONNX (chosen)

- **Pros.** ~26 MB, 192-dim, opset 13 with only basic ops, mirrored
  on Hugging Face under permissive terms, ships with companion test
  fixtures (`1-two-speakers-en.wav`) we can reuse, well-documented
  pre-processing (`feature_normalize_type = global-mean`,
  `sample_rate = 16000` in the ONNX metadata).
- **Cons.** EN-only checkpoint; the same architecture trained on
  Mandarin / multilingual is twice the size and not yet calibrated
  here.

#### 2. WeSpeaker ResNet-34 / CAM++

- **Pros.** Same mirror, also ~26 MB, well-supported by sherpa-onnx;
  serves as drop-in fallback if ERes2Net underperforms.
- **Cons.** Slightly lower DER on standard benchmarks per upstream
  reports; chose ERes2Net for the marginal quality edge while keeping
  WeSpeaker as the pre-approved escape hatch.

#### 3. NVIDIA TitaNet (NeMo)

- **Pros.** Highest single-model quality on VoxCeleb test-O.
- **Cons.** ONNX export drags NeMo's pre/post ops; tract refuses to
  optimise; would force us onto `ort`. Not worth the ~2 % DER edge
  given the build-complexity cost.

#### 4. pyannote 3.1

- **Pros.** State-of-the-art when used end-to-end.
- **Cons.** Practical deployment expects a Python runtime (or a
  bespoke Rust port we would have to write and maintain). Conflicts
  with ADR-0003's "no Python on the user's machine" stance.

### Runtime

#### A. tract-onnx (chosen)

- **Pros.** Pure Rust, statically linked, already in the workspace
  for Silero VAD, supports every op the model uses.
- **Cons.** Slower than onnxruntime on heavy graphs; not an issue at
  ~26 MB with 516 nodes.

#### B. ort

- **Pros.** Fastest available CPU/Metal inference; mature.
- **Cons.** Requires bundling onnxruntime's C++ shared library on
  every target, complicates notarisation. Only worth the cost if
  tract's perf becomes a bottleneck — not the case at day-6 sizing.

#### C. candle

- **Pros.** Native HF tooling, pure Rust.
- **Cons.** No first-class ONNX import; would require re-exporting
  the model into a candle-friendly format.

### Clustering

#### I. Online threshold-based (chosen)

- **Pros.** O(k) state, single pass, labels emitted as chunks arrive,
  trivial to extend with `rename` for the upcoming UI affordances.
- **Cons.** Sensitive to threshold tuning; mis-calibrated meetings
  can split or merge speakers.

#### II. Offline agglomerative

- **Pros.** Higher accuracy in the literature; standard in
  whisper.cpp/diarize-toolkit.
- **Cons.** Cannot run inline, breaks the streaming UX promise. Could
  be added later as a "polish" pass after the meeting ends.

#### III. Online VBx / spectral

- **Pros.** Best-in-class accuracy.
- **Cons.** Implementation effort ≫ Sprint 1 budget; pre-existing
  Rust ports do not exist.

## References

- `docs/ARCHITECTURE.md` §3.2.7 — diarization layering.
- `docs/DEVELOPMENT_PLAN.md` §4.3 — purity target.
- `crates/echo-domain/src/ports/diarizer.rs` — port definition.
- `crates/echo-diarize/src/eres2net.rs` — adapter implementation.
- `crates/echo-diarize/src/cluster.rs` — online clustering.
- ERes2Net paper — Chen et al., "An Enhanced Res2Net with Local and
  Global Feature Fusion for Speaker Verification", Interspeech 2023.
- Model mirror — https://huggingface.co/csukuangfj/speaker-embedding-models
- Upstream — https://www.modelscope.cn/models/iic/speech_eres2net_sv_en_voxceleb_16k
