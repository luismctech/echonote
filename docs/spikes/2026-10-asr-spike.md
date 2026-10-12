# ASR spike: Parakeet TDT v3 and Nemotron 3.5 Streaming vs Whisper large-v3-turbo (Spanish)

Date: 2026-10-11/12. Status: spike result, not a decision (no ADR). Nothing in the repo
source or dependencies was changed; scripts and raw results live outside the repo.

## Question

Would replacing or complementing the current ASR (whisper.cpp, large-v3-turbo ggml, Metal,
greedy, no context, language hint `es`, fed in 5 s non-overlapping chunks in live mode) with
NVIDIA Parakeet-TDT-0.6B-v3 (offline) or Nemotron-3.5-ASR-Streaming-0.6B (streaming), both
through sherpa-onnx int8, improve Spanish accuracy or latency enough to justify an integration?

## Setup

| Item | Value |
|---|---|
| Machine | Apple M1 Pro, 8 cores, AC power, idle; systems run strictly sequentially |
| Threads | 4 for every sherpa-onnx system (equals the app default `num_cpus / 2`); whisper.cpp uses the app default (4) plus the Metal GPU |
| sherpa-onnx | Python wheel 1.13.8 (same version as the Rust crate), CPU provider unless stated |
| Models | Parakeet v3 int8 (643 MB), Nemotron 3.5 560 ms int8 (666 MB), Nemotron 3.5 160 ms int8 (654 MB), Whisper turbo int8 on sherpa (1.0 GB), app baseline `ggml-large-v3-turbo.bin` (1.5 GB, installed model) |
| App baseline | The repo's real `WhisperCppTranscriber` (whisper-rs, Metal) with the app's `FullParams` (greedy, no context, temperature 0, `es`), via a throwaway crate that depends on `echo-asr`/`echo-domain` by path |
| Dataset | FLEURS `es_419` test split (908 utterances, public, read speech) |
| Subset | 400 utterances: parquet rows shuffled with Python `random.Random(42)`, first 400 taken; 16 kHz mono 16-bit WAV; 4,996.8 s total (83.3 min), 5.3-29.5 s, mean 12.5 s. Some sentences appear 2-3 times (different speakers), as in FLEURS |
| Reference | `raw_transcription` (cased, punctuated, digits as written) |

### Normalisation (applied identically to reference and every hypothesis)

NFC, lowercase, every Unicode punctuation/symbol character replaced by a space, whitespace
collapsed. Two reporting variants: accents kept, and accents folded (NFD, drop combining marks).
A third variant (folded + digits to Spanish words with `num2words`, installed in the throwaway
venv only) neutralises number formatting. WER and CER via `jiwer` over the corpus (not mean of
per-utterance values).

Number caveat: FLEURS references use digits ("802.11n", "2.4Ghz", "30%"). Nemotron emits spoken
forms ("ochocientos dos punto once"), which counts as errors against digit references; the
digits-to-words column shows the gap is mostly formatting. The digit conversion is itself
approximate (decimals, units, "%"), so treat that column as indicative.

## Results (400 utterances, 4,996.8 s)

WER/CER in %, lower is better. RTF = decode time / audio duration (model load excluded;
p50/p95 over utterances; total over the corpus). RSS = peak resident set of the whole process
(`/usr/bin/time -l`). CPU-s/audio-s = user CPU time per second of audio (whole run, load
included), a proxy for energy cost; the app's Metal path mostly burns GPU, not shown.

| System | Mode | WER kept | WER folded | WER folded + digits2words | CER kept | CER folded | RTF p50 | RTF p95 | RTF total | Peak RSS | Load | CPU-s/audio-s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **App baseline** whisper.cpp large-v3-turbo (Metal) | whole utterance | **2.73** | **2.50** | **2.34** | 0.97 | 0.92 | 0.108 | 0.167 | 0.104 | 1.92 GB | 0.9 s | 0.004 |
| App baseline (same code path) | **5 s chunks (app live mode)** | 8.54 | 8.31 | 8.15 | 6.01 | 5.94 | 0.278 | 0.336 | 0.277 | 1.90 GB | 0.9 s | 0.005 |
| Parakeet v3 int8, CPU | whole utterance | 3.80 | 3.63 | 3.40 | 1.52 | 1.47 | **0.030** | 0.032 | **0.030** | 1.50 GB | 1.0 s | 0.13 |
| Parakeet v3 int8, CoreML | whole utterance | 3.70 | 3.54 | 3.36 | 1.45 | 1.41 | 0.062 | 0.081 | 0.065 | 8.08 GB | 5.9 s | 0.25 |
| Parakeet v3 int8, CPU | 5 s chunks | 11.05 | 10.84 | 10.43 | 6.89 | 6.81 | 0.033 | 0.034 | 0.033 | 1.24 GB | 1.0 s | 0.15 |
| Nemotron 3.5 560 ms int8 | whole utterance fed at once | 6.33 | 6.17 | 4.29 | 3.57 | 3.50 | 0.097 | 0.103 | 0.098 | 1.27 GB | 0.8 s | 0.52 |
| Nemotron 3.5 160 ms int8 | whole utterance fed at once | 6.98 | 6.81 | 4.92 | 3.81 | 3.75 | 0.245 | 0.261 | 0.242 | 1.29 GB | 0.8 s | 1.31 |
| Whisper turbo int8 on sherpa, CPU | whole utterance | 3.92 | 3.71 | 3.53 | 2.12 | 2.07 | 0.245 | 0.277 | 0.244 | 2.99 GB | 1.1 s | 0.96 |

Cross-check: the repo's own `echo-proto bench wer` (whole utterances, its internal WER
normalisation, no warm-up) reports 2.89 % WER, RTF p50 0.11 / p95 0.17 on the same 400 pairs,
consistent with the 2.73 % above. `bench wer` transcribes the whole file (no chunking), so the
5 s mode needed the throwaway crate.

Model load time for whisper.cpp is 0.9 s warm; the first cold load after boot took about 9.7 s
(Metal shader compile / page-in) in a smoke test.

### Streaming latency (Nemotron, real-time feed, 20 utterances, 248 s)

Audio is fed in 100 ms pieces paced to wall clock. First-partial = time from audio start to the
first non-empty partial. Speech onset (first 20 ms frame with RMS > 0.01) has p50 1.19 s, because
FLEURS clips begin with silence, so the onset-relative value is the fairer user-perceived figure.
Final latency = time from the end of audio to the final text (includes 0.8 s of tail padding
decode). Peak RSS 1.27 GB for both runs. WER on this 20-utterance subset (kept / folded /
folded + digits2words): 560 ms 7.68 / 7.50 / 5.11; 160 ms 8.21 / 8.04 / 6.17 (small sample,
indicative only).

| Variant | First partial from audio start p50 / p95 | First partial after speech onset p50 / p95 | Final after end of audio p50 / p95 |
|---|---|---|---|
| Nemotron 560 ms | 2.48 s / 2.52 s | 0.93 s / 2.25 s | 0.07 s / 0.13 s |
| Nemotron 160 ms | 2.05 s / 2.68 s | 0.85 s / 1.93 s | 0.21 s / 0.24 s |

The 160 ms variant is not meaningfully faster to first text (the model emits its first token
about 1 s after speech onset either way) while costing 2.5x the CPU and +0.7 WER points.
For comparison, the app today cannot show any text before a 5 s chunk is full (plus decode
time of about 1.4 s for a 5 s chunk at RTF 0.28).

### Worst examples (full lists in `worst.txt`, raw results listed below)

Hardest utterances for every system are numeric/technical ones ("802.11n ... 2.4Ghz ... 5.0Ghz").

- Parakeet: `802 11N`, `2.4 GHz`, `5.0 gz`; proper-noun errors ("Scotur ... Intra ... encabo, Daroka" for "Scotturb ... Sintra ... Cabo da Roca"); merges "29¾ pulgadas" into "23 29 3 c...".
- Nemotron 560 ms: "El ochocientos dos once N Standard ..." (spoken-form numbers, English word "Standard"), and the clause "Esto era posible su compatibilidad" for "Esto hará posible su compatibilidad" (an actual recognition error).
- App whole: similar proper-noun errors ("Parancabo da Roca"), drops "p. m." details, and "10 y 11 pm".
- App 5 s chunks: duplicated or invented text at chunk boundaries ("tanto como el 802.11n estándar. tanto en ...", "los brindantes se han convertido en un gran británicos") and trailing hallucinations ("Gracias.", "¿Qué?").

## Qualitative notes

- **Punctuation and casing.** Parakeet, Whisper (both stacks) and the app all produce cased,
  punctuated text with digits (400/400 Parakeet outputs end in `.?!`). Nemotron produces cased text
  but only 91/400 (560 ms) and 34/400 (160 ms) outputs end in sentence punctuation and 284/272 contain
  any punctuation at all: weaker for readable meeting transcripts and for the summary LLM.
- **Numbers.** Parakeet and Whisper emit digits like the references. Nemotron emits only spoken
  forms (0/400 outputs contain a digit versus 67 references with digits). Needs inverse text
  normalisation if adopted.
- **Hallucinations.** Whole-utterance runs are clean (1/400 flagged phrase, "gracias" check, for every system).
  In 5 s chunk mode the app baseline emitted an unspoken "Gracias"/subtitle-style phrase in 90/400
  utterances; Parakeet in the same chunking 4/400. This is inflated here because FLEURS tails often end
  in silence and the harness has no RMS/VAD gate, which the live app has; the real rate on meetings is
  not measured.
- **Chunking dominates everything.** For Whisper, going from whole utterances to the app's 5 s chunks
  triples WER (2.73 -> 8.54). Parakeet degrades even more (3.80 -> 11.05) because cuts land mid-word.
  The segmentation policy matters more than the model choice. Neither model should be judged in the
  chunked regime as if it were the final design: a VAD-aligned segmenter (utterance-sized pieces) would
  restore near-whole-utterance accuracy. That was not tested.
- **Quantisation cost.** Same Whisper turbo weights: ggml fp16 on Metal 2.73 % versus sherpa int8 on
  CPU 3.92 % WER and 4.7x slower RTF (0.104 vs 0.244). int8 is not free; Parakeet/Nemotron are only
  available here as int8.
- **CoreML provider** (Parakeet): marginally better WER (3.70 vs 3.80, within noise), 2x slower than CPU,
  5.9 s load and 8 GB peak RSS. Not worth it; use CPU.

## Licensing

- **Parakeet-TDT-0.6B-v3**: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/). Obligation:
  attribution (credit NVIDIA/model name, link to the license, indicate if modified; the int8 ONNX
  conversion counts as a modification to indicate). Commercial use and redistribution allowed.
- **Nemotron-3.5-ASR-Streaming-0.6B**: OpenMDW-1.1 (https://openmdw.ai/license/1-1/; model card
  `license: other`). The earlier audit that called it CC-BY was wrong. Obligations from the license text:
  (1) free permission to use, modify and distribute the model materials without restriction, including
  copyright, patent, database and trade secret rights; (2) if you distribute any part, include a copy of
  the agreement and keep all copyright/origin notices; (3) patent or copyright litigation claiming the
  materials infringe terminates your rights (unless a defensive counterclaim); (4) no restrictions or
  obligations on outputs; (5) provided as is, no warranty; you are solely responsible for rights
  clearance and due diligence. Permissive and bundle-friendly; the compliance work is shipping the
  license text and notices.
- sherpa-onnx is Apache-2.0 (code); Whisper weights are MIT (not re-verified in this spike).

## Integration cost (estimate)

- **Runtime.** Official `sherpa-onnx` Rust crate 1.13.8 (plus ONNX Runtime native libs; whisper-rs 0.16
  cannot load Parakeet as it needs whisper.cpp >= 1.9). Adds a second native inference stack next to
  whisper.cpp; check bundle size, code signing/notarisation of the extra dylibs and CI build time
  (not measured).
- **Offline engine (Parakeet).** A new adapter behind the existing `Transcriber` port is the small
  part: `transcribe(&[f32], &TranscribeOptions) -> Transcript` maps directly. It has no language hint
  (auto multilingual, 25 languages), no `initial_prompt`, and no per-segment timestamps in the path I used;
  these are product regressions versus whisper.cpp to check (the segment model in `echo-domain`).
- **Streaming engine (Nemotron).** `Transcriber` is stateless per 5 s chunk; streaming needs a new
  stateful port (open stream, feed PCM, partial/final events, flush) and changes to the streaming use case
  and UI (partials). Larger design change than a model swap.
- **Catalog/download.** Models ship as `tar.bz2` archives with 3-4 files (encoder/decoder/joiner/tokens),
  not a single `.bin`; the model catalog needs archive extraction and multi-file integrity checks.
- **Size.** Parakeet int8 643 MB versus 1.5 GB for ggml large-v3-turbo; peak RSS 1.5 GB vs 1.9 GB.
  Nemotron 654-666 MB.
- **Speed/energy.** Parakeet CPU RTF 0.030 is 3.5x faster than the Metal Whisper path (0.104) in whole-
  utterance mode and needs no GPU, but uses 0.13 CPU-s per audio-s (Whisper/Metal 0.004 CPU-s + GPU).
  Relevant for fan/battery on long meetings; not measured with a power meter.

## Recommendation

1. **No-go on replacing whisper.cpp large-v3-turbo with either model for accuracy.** On identical
   audio and whole utterances the current model is clearly best: 2.73 % WER vs Parakeet 3.80 % and Nemotron
   6.33 % (4.29 % after neutralising number format), with the best punctuation. A ~1.1-point WER gap (about
   130 words on this set; not bootstrap-tested) is the cost of Parakeet's speed.
2. **Nemotron: no-go for now.** Worse accuracy, no digits, sparse punctuation, ~1 s to first text after
   speech onset, 5x Parakeet's CPU cost, and a stateful streaming port. Revisit only if live partial
   text becomes a product priority, then with the 560 ms variant (160 ms gives no first-token benefit here).
3. **Parakeet: conditional, optional engine, not a replacement.** Worth a follow-up only as an opt-in
   fast/low-RAM/no-GPU backend and only after utterance-aware (VAD) segmentation exists, since in the
   app's 5 s regime it is worse than Whisper (11.05 % vs 8.54 %). Gate any ADR on a second spike on real
   meeting audio.
4. **Highest-value finding: the 5 s chunking, not the model.** The same whisper.cpp model drops from
   8.54 % to 2.73 % WER when fed whole utterances, and chunk-boundary hallucinations disappear. A
   VAD-aligned segmenter (or longer windows with overlap/context) is the cheapest accuracy lever and
   benefits every engine. Recommended next spike: VAD segmentation + whisper.cpp (and Parakeet) on
   meeting-like audio.

## Limits (be careful extrapolating)

- Read speech (FLEURS es_419 is clean, one speaker, studio-like); no overlap, noise, crosstalk,
  code-switching or meeting domain vocabulary. Rankings may change on real meetings.
- Latin American Spanish only; Spanish-from-Spain, other languages and accents were not tested.
- int8 models only (the only sherpa-onnx exports); fp16/fp32 Parakeet/Nemotron could be better.
- One machine (M1 Pro), one run per system (no run-to-run variance); 4 threads for sherpa.
  Whisper.cpp ran on GPU, sherpa on CPU, so RTF compares engines as the app would run them, not
  equal hardware. CoreML only for Parakeet.
- 400-of-908 random subset; WER differences below about 0.3 points should not be treated as real.
- The Nemotron model card lists FLEURS among training datasets; the test split is nominally held
  out but contamination cannot be excluded.
- Real-time latency measured on 20 utterances only; timer granularity 100 ms feed.
- App chunk mode was reproduced by slicing at 5 s without the app's RMS/VAD gate and a minimum tail
  of 0.5 s, so hallucination counts are an upper bound for the real app.
- The digits-to-words variant is approximate.

## Reproducibility

Scripts (not committed): `prep.py` (dataset), `run_sherpa.py` (Parakeet/Nemotron/Whisper-sherpa,
offline and real-time modes, `CHUNK5=1` for chunked), `score.py` (normalisation + jiwer), a throwaway
crate `chunkbench` (depends on `echo-asr` and `echo-domain` by path; modes `whole` and `chunk5`),
and `run_all.sh` (sequential driver wrapping each run in `/usr/bin/time -l`). Raw per-utterance JSONL,
`scores.json`, `worst.txt` and time logs were kept in the session scratchpad
(`asr-spike/results/`), not in the repo.
