# Benchmark: speech-aligned chunking vs fixed 5 s chunks on long Spanish streams

Date: 2026-10-12. Status: benchmark result for task T5 of `vad-diarization` (branch
`feat/vad-diarization` at `d0a59f1`). No repository source was changed. The harness,
streams and raw results live outside the repo, in the session scratchpad.

## Question

Does the speech-aligned chunking from T4 (cut at VAD pauses between `min_chunk_ms` and
`max_chunk_ms`) beat the fixed 5 s chunks on long Spanish streams? Which `(min, max)` should
be the default? Does Silero v6.2 change the result compared with v5?

## Setup

| Item | Value |
|---|---|
| Pipeline | The real `StreamingPipeline::run_pipeline` (rubato resampler, `SileroVad::for_meetings`, `NaivePunctuator` as in the Tauri app, no diarizer), driven by a throwaway crate that depends on the repo crates by path |
| Capture | File-backed `AudioCapture`: 16 kHz mono, 20 ms frames, fed as fast as the pipeline pulls them (not real time). Nothing is dropped because a single ASR call is in flight at a time |
| ASR | `ggml-large-v3-turbo.bin` (repo `models/asr`), whisper.cpp with Metal, language `es`. Apple M1 Pro, runs strictly sequential |
| VAD | v5: repo `models/vad/silero_vad.onnx`. v6.2: the inlined `silero_vad.onnx` from the T1/T2 spike (SHA-256 `76d1667a…08b8`) |
| Streams | 12 synthetic streams, 38.3 min total (170-219 s each). Each stream is 15 FLEURS `es_419` utterances (from the 400-utterance ASR-spike subset), picked with `random.Random(20261012)`, no repeats across streams |
| Gaps | Every utterance is energy-trimmed (-40 dB re p95, 100 ms margin kept, 10 ms fades) and preceded by a gap of Gaussian noise at an RMS of 0.0005-0.003. Streams s00-s09 use 60 % short gaps (150-400 ms) and 40 % long gaps (0.6-2.0 s). Streams s10-s11 use only very short gaps (50-200 ms), to stress the forced-cut path. A 1 s noise tail closes each stream |
| Reference | The utterances' `raw_transcription`, concatenated per stream |
| Scoring | Corpus WER/CER over the 12 streams (jiwer), with the ASR-spike normalisation (NFC, lowercase, punctuation stripped). Two variants: accents kept and accents folded |
| Oracle | Each trimmed utterance is transcribed whole (perfect segmentation), as a floor |

### Metric definitions

- **Hallucination**: a chunk output that is non-empty although the chunk overlaps no
  ground-truth speech, or that contains a lexicon word (`gracias`, `subtítulos`, `amara`,
  `suscríbete`) not present in the reference, or that contains a phrase repeated back to back.
  All the outputs flagged here were "Gracias.".
- **Ground-truth speech**: 20 ms frames within 25 dB of each utterance's 95th-percentile
  level. FLEURS keeps a room floor of about -50 dBFS around the speech.
- **Cuts in voiced audio**: chunk boundaries that fall within ±40 ms of a ground-truth speech
  frame, a proxy for split words. **Forced**: VAD-aligned cuts that are not at a
  voiced-to-silence transition, meaning `max_chunk_ms` was reached.
- **End-of-speech latency**: the stream position when the chunk was cut, minus the end of
  the last speech in the chunk, plus that chunk's ASR time. This is the wait between someone
  finishing speaking and their text appearing. **First-word latency**: the same, measured from
  the first speech in the chunk, which is the worst wait for any word.

## Results

Silero v6.2, all configs:

| Config | WER/CER kept % | WER/CER folded % | WER s10-s11 | Ins. | Chunks | Len p50/p95/max s | Cuts in voiced / total | Forced | Halluc. | EoS lat. p50/p95 s | First-word lat. p50/p95 s | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle (whole utterance) | 2.24 / 0.82 | 2.03 / 0.78 | 2.42 | 7 | 180 | - | - | - | - | - | - | - |
| Fixed 5 s (today) | 7.92 / 5.08 | 7.71 / 5.01 | 8.59 | 142 | 461 | 5.0 / 5.0 / 5.0 | 332 / 452 | - | 3 | 1.27 / 3.55 | 6.22 / 6.47 | 0.26 |
| Aligned 2000/10000 | 3.40 / 2.07 | 3.14 / 2.01 | 4.23 | 55 | 296 | 7.7 / 10.0 / 10.0 | 51 / 296 | 87 | 22 | 2.09 / 2.84 | 9.23 / 11.55 | 0.18 |
| Aligned 3000/15000 (T4 default) | 2.33 / 1.00 | 2.08 / 0.95 | 2.42 | 13 | 219 | 9.2 / 14.7 / 15.0 | 19 / 219 | 19 | 4 | 2.06 / 2.66 | 10.33 / 16.27 | 0.13 |
| Aligned 4000/15000 (extra) | 2.31 / 0.97 | 2.05 / 0.92 | 2.42 | 12 | 215 | 9.2 / 14.7 / 15.0 | 19 / 215 | 19 | 3 | 2.04 / 2.50 | 10.38 / 16.26 | 0.13 |
| **Aligned 4000/20000** | **2.18 / 0.78** | **1.93 / 0.73** | 2.30 | 7 | 204 | 9.5 / 15.5 / 20.0 | 13 / 204 | 4 | **0** | 2.02 / 2.46 | 10.55 / 16.89 | 0.12 |
| Aligned 3000/25000 | 2.22 / 0.84 | 1.99 / 0.80 | 2.54 | 8 | 206 | 9.4 / 15.5 / 23.7 | 12 / 206 | 1 | 1 | 2.01 / 2.35 | 10.53 / 16.87 | 0.12 |

Silero v5 (repo model), same streams:

| Config | WER/CER kept % | WER/CER folded % | Forced | Halluc. | EoS lat. p50/p95 s |
|---|---|---|---|---|---|
| Fixed 5 s | 7.92 / 5.08 | 7.71 / 5.01 | - | 3 | 1.27 / 3.59 |
| Aligned 2000/10000 | 3.27 / 1.67 | 3.02 / 1.61 | 82 | 10 | 2.00 / 2.74 |
| Aligned 3000/15000 | 2.43 / 1.02 | 2.20 / 0.97 | 15 | 4 | 1.98 / 2.45 |
| Aligned 4000/20000 | 2.26 / 0.85 | 2.05 / 0.80 | 3 | 0 | 1.99 / 3.06 |
| Aligned 3000/25000 | 2.29 / 0.86 | 2.05 / 0.81 | 0 | 1 | 2.06 / 2.38 |

ASR time per chunk is 1.2-1.4 s (p50) in every config, because whisper.cpp pads the encoder
to 30 s. That is why the fixed 5 s run costs more in total (RTF 0.26) than the aligned runs
(about 0.12).

## Findings

1. **Speech-aligned chunking removes almost all of the chunking penalty.** WER drops from
   7.92 % (fixed 5 s) to 2.18-2.33 %, which is within noise of the 2.24 % oracle floor.
   Insertions drop from 142 to 7-13. With fixed chunks, 73 % of cuts land in voiced audio;
   with aligned chunks at a max of 15 s or more, it is 6-9 %.
2. **The value of `max_chunk_ms` drives the remaining errors and the hallucinations.** Every
   hallucination in the aligned runs is "Gracias." on a chunk shorter than about 1 s: the
   leftover after a forced cut, or a short tail. A 10 s max forces 87 cuts and produces 22
   hallucinations. A 15 s max gives 19 forced cuts and 3-4 hallucinations, whether `min` is
   3 s or 4 s. A max of 20 s or more brings forced cuts down to 0-4 and hallucinations to 0-1.
3. **Raising `min` alone barely helps.** 4000/15000 behaves like 3000/15000.
4. **Latency barely depends on (min, max) from 15 s up.** End-of-speech latency is about 2.0
   s p50 and 2.3-2.7 s p95: roughly 0.7 s of VAD hysteresis and 200 ms analysis block together,
   plus about 1.3 s of ASR. With fixed 5 s chunks the p50 is lower (1.27 s), but text lags
   speech by up to 6.5 s per word, and a word cut at the boundary is often wrong. Raising
   max from 15 s to 20 s moves the first-word p95 by only 0.6 s (16.3 to 16.9 s). The longer
   max only matters during long unbroken speech.
5. **v6.2 vs v5: no meaningful accuracy difference** on clean read speech. At the
   recommended config v6.2 is 0.08 points better (2.18 vs 2.26), and the differences between
   configs are similar in size. The case for v6.2 still rests on the T1 noise robustness
   result (0 % vs 5.1 % false positives on fan noise), which these streams do not exercise.
   With fixed 5 s chunks the two VADs produce identical transcripts, because there they only
   gate whole chunks that are almost all voiced.

## Recommendation

- Change the default to **`min_chunk_ms` = 4000 and `max_chunk_ms` = 20000**. It has the best
  WER (2.18 %, at the oracle floor), no hallucinations, 4 forced cuts in 38 min, and the same
  end-of-speech latency as 3000/15000. The cost is a worst-case chunk of 20 s instead of 15 s
  during unbroken speech.
- Alternative: keep 3000/15000 if the 15 s ceiling matters more than 0.15 WER points and
  about 4 "Gracias." outputs per 38 min.
- Avoid a max of 10 s or less.
- Follow-up worth a small task: do not transcribe speech chunks shorter than about 1 s, or
  merge them into the next chunk. That targets the only hallucination source left.

## Limitations

- The streams are read speech with synthetic noise gaps. They do not cover conversation
  (overlap, backchannels), music or real room noise, so the v6.2 noise advantage is not
  measured here.
- Latency is a proxy computed from stream positions with offline feeding. Real-time capture
  adds the device buffering, and a slower machine adds ASR time.
- The run for v5 at 3000/25000 was repeated because outside GPU contention slowed its first
  three streams by 4x. The table shows the clean rerun; its WER was identical.

## Reproduce (scratchpad `t5-bench/`)

- `scripts/build_streams.py <fleurs400> streams`: deterministic, seed 20261012. Writes
  `streams/composition.json` with the gaps and utterance positions.
- `harness/`: the `t5bench` crate. `t5bench stream <model> <vad> streams <out> fixed|aligned
  <min> <max> 5000` and `t5bench oracle <model> streams <out>`. It needs the
  `-Wl,-rpath,/usr/lib/swift` link argument, as the repo's `.cargo/config.toml` sets.
- `scripts/run_all.sh` runs everything sequentially. `scripts/score_t5.py` produces
  `results/scores.{json,txt}`.
