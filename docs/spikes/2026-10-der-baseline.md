# Baseline: diarization error rate (DER) of the current streaming diarizer

Date: 2026-10-12. Status: baseline for task T6 of `vad-diarization` (branch
`feat/vad-diarization` at `27bc03d`). The eval data, scripts and raw results live outside the
repo, in the session scratchpad (`der-eval/`). They are not committed.

## Question

The repo had no diarization metric. How good is the diarization the app runs today, measured
as DER on Spanish conversations and on real multi-speaker audio? How do the embedder, the
pyannote segmenter and the offline refinement each contribute?

## Harness

`echo-proto bench der --manifest <file>` (commit `27bc03d`). The scorer is
`echo_diarize::metrics` (`0d72ae3`).

| Item | Value |
|---|---|
| Pipeline | The real `StreamingPipeline`: a file capture in 20 ms frames fed as fast as it is pulled, the rubato resampler, `SileroVad::for_meetings` with speech-aligned chunks (defaults min 4000 / max 20000 ms), and `OnlineDiarizer`. The transcriber is a no-op, so there is no ASR cost |
| Diarizer | `EmbedderFamily::cluster_config` (`9f0afd3`) gives the same values `build_diarizer` uses: CAM++ similarity 0.85 with online merging off, `max_speakers` 4, offline refinement on. A segmenter is attached when `--segmenter` is passed |
| Hypothesis | A decorator records the speaker spans from each `assign_spans` call. They are moved to session time with each `Chunk` event's `offset_ms`, then relabelled with the `SpeakersMerged` pairs from the end-of-session refinement |
| Scoring | md-eval style. The timeline is split at every boundary, and each speaker label is mapped once to a reference speaker with the Hungarian algorithm. The collar is 0.25 s on each side of each reference boundary. Overlap is scored. DER = (miss + false alarm + confusion) / scored reference speech, pooled over files with time weighting |
| VAD | Silero v6.2, inlined (`silero_vad.onnx`, SHA-256 `76d1667a…08b8`, from the T1/T2 spike) |
| Segmenter | `models/segmenter/pyannote_segmentation_3_tract.onnx`, the tract-patched export |
| Embedders | multi: `campplus_zh_en_common_advanced.onnx` (catalog `embedder-campplus-multi`, 192-dim). en: `campplus_en_voxceleb.onnx` (512-dim) |
| Machine | Apple M1 Pro. Five runs in parallel. Each run takes 35-265 s |

The hypothesis is the diarizer's own span timeline. The app saves one speaker per Whisper
segment, chosen as the span with the most overlap. The DER a user sees therefore also depends
on how ASR splits the segments. This harness measures the diarizer on its own.

## Eval data

| Set | Content | Licence |
|---|---|---|
| `mls_es` (synthetic) | 20 seeded conversations (seed 20261012), 90.6 min, 183-361 s each, 2-4 speakers (2 speakers: 7, 3 speakers: 10, 4 speakers: 3). 17 distinct readers from the Multilingual LibriSpeech Spanish **test** split (HF `facebook/multilingual_librispeech`, `spanish/test-00000-of-00001.parquet`, 157.8 MB, SHA-256 `ffc32cb4a2d015c9…`). 1009 turns with alternating speakers and gaps of 0.1-1.0 s. Turn lengths are log-normal, clamped to 1.5-15 s. 142 turns (14 %) are short backchannels of 0.4-1.4 s. 57 turns (5.6 %) overlap the previous turn by 0.2-1.0 s, 32.8 s of overlap in total. Turns are random crops of utterances, levelled to the same RMS with a per-speaker gain of ±3 dB. The reference RTTM marks energy-based speech inside each turn: 20 ms frames above 10 % of the p95 RMS, pauses of 0.4 s or more split a turn | CC-BY-4.0 |
| `voxconverse` (real) | 12 VoxConverse **dev** recordings (v0.3 RTTMs from `github.com/joonson/voxconverse`). They are mostly English debates and news. 50.5 min, 2-8 min each, 2-6 speakers. The files were picked with seed 20261012 from the 92 dev files that have 2-6 speakers and last 2-8 min. Only these 12 wavs (82.5 MB) were extracted from the official 1.99 GB dev zip, using HTTP range requests | CC-BY-4.0 (copyright stays with the video owners) |
| `two_speakers_en.wav` | Skipped: it has no reference RTTM | — |

Total download: about 241 MB.

**Caveat:** `mls_es` is read audiobook speech mixed digitally. It has clean channels, no
reverberation and consistent levels. It is easier than real meetings, so treat its numbers as a
lower bound on the error. Three of the VoxConverse files have 6 speakers, more than the
`max_speakers = 4` cap. Those files cannot score well.

## Results

The **app** row is the current app configuration with the multilingual CAM++ embedder:
segmenter on, refinement on, threshold 0.85.

### Spanish synthetic (`mls_es`, 20 files)

| Embedder | Segmenter | Refine | DER % | Miss % | FA % | Conf % | Est. spk (mean) | True spk (mean) | Count exact | Median file DER % |
|---|---|---|---|---|---|---|---|---|---|---|
| multi | on | on (**app**) | **18.76** | 2.96 | 0.70 | 15.10 | 4.00 | 2.80 | 3/20 | 11.01 |
| multi | on | off | 18.76 | 2.96 | 0.70 | 15.10 | 4.00 | 2.80 | 3/20 | 11.01 |
| multi | off | on | 40.50 | 0.86 | 7.13 | 32.51 | 4.00 | 2.80 | 3/20 | 38.36 |
| multi | off | off | 40.50 | 0.86 | 7.13 | 32.51 | 4.00 | 2.80 | 3/20 | 38.36 |
| en | on | on | 45.97 | 2.97 | 0.71 | 42.29 | 4.00 | 2.80 | 3/20 | 48.48 |
| en | on | off | 45.97 | 2.97 | 0.71 | 42.29 | 4.00 | 2.80 | 3/20 | 48.48 |
| en | off | on | 52.74 | 0.86 | 7.13 | 44.75 | 3.70 | 2.80 | 2/20 | 53.00 |
| en | off | off | 52.74 | 0.86 | 7.13 | 44.75 | 3.70 | 2.80 | 2/20 | 53.00 |

### VoxConverse dev subset (12 files)

| Embedder | Segmenter | Refine | DER % | Miss % | FA % | Conf % | Est. spk (mean) | True spk (mean) | Count exact | Median file DER % |
|---|---|---|---|---|---|---|---|---|---|---|
| multi | on | on (**app**) | **20.19** | 2.99 | 0.90 | 16.30 | 4.00 | 3.33 | 0/12 | 13.34 |
| multi | on | off | 20.19 | 2.99 | 0.90 | 16.30 | 4.00 | 3.33 | 0/12 | 13.34 |
| multi | off | on | 35.36 | 2.67 | 2.86 | 29.83 | 4.00 | 3.33 | 0/12 | 33.50 |
| multi | off | off | 35.36 | 2.67 | 2.86 | 29.83 | 4.00 | 3.33 | 0/12 | 33.50 |
| en | on | on | 39.12 | 2.65 | 0.83 | 35.63 | 4.00 | 3.33 | 0/12 | 42.02 |
| en | on | off | 39.12 | 2.65 | 0.83 | 35.63 | 4.00 | 3.33 | 0/12 | 42.02 |
| en | off | on | 43.30 | 2.67 | 2.86 | 37.77 | 3.42 | 3.33 | 2/12 | 43.90 |
| en | off | off | 43.30 | 2.67 | 2.86 | 37.77 | 3.42 | 3.33 | 2/12 | 43.90 |

### Similarity threshold (multi, segmenter on, refine on)

This sweep used the same files as the tables above. There was no held-out split, so the best
value here is optimistic. It shows a direction, not a calibrated value.

| Threshold | `mls_es` DER % | `mls_es` conf % | `mls_es` est. spk / count exact | VoxConverse DER % | VoxConverse conf % | VoxConverse est. spk / count exact |
|---|---|---|---|---|---|---|
| 0.85 (app) | 18.76 | 15.10 | 4.00 / 3 of 20 | 20.19 | 16.30 | 4.00 / 0 of 12 |
| 0.80 | 17.72 | 14.07 | 4.00 / 3 of 20 | 19.77 | 16.01 | 4.00 / 0 of 12 |
| 0.75 | 14.47 | 10.83 | 4.00 / 3 of 20 | 17.93 | 14.13 | 4.00 / 0 of 12 |
| 0.70 | 13.64 | 9.99 | 4.00 / 3 of 20 | 15.95 | 12.16 | 3.83 / 1 of 12 |
| 0.60 | 12.52 | 8.88 | 3.95 / 3 of 20 | 13.78 | 10.05 | 3.75 / 1 of 12 |
| 0.50 | 8.65 | 5.00 | 3.90 / 4 of 20 | 12.97 | 9.25 | 3.58 / 2 of 12 |
| 0.40 | **6.67** | 3.03 | 3.20 / 10 of 20 | **10.80** | 7.09 | 3.42 / 2 of 12 |

A rerun of the app configuration with the binary rebuilt at `27bc03d` reproduced 18.76 %
exactly. Each configuration is deterministic.

## Findings

1. **The app over-splits every recording up to the speaker cap.** In the app configuration
   (threshold 0.85), every file in both sets ends with exactly 4 hypothesis speakers (`max_speakers`), whatever the true count.
   Confusion is 80 % or more of the DER. The 0.85 threshold was tuned for the 512-dim EN
   export, and it is too strict for the 192-dim multilingual CAM++ (this confirms the
   2026-07 note). Lowering it to 0.40-0.50 cuts DER by about 1.9-2.8× on both sets. The value
   must be calibrated on held-out data before it ships.
2. **The offline refinement never fires.** It produced 0 merges for every file (20 + 12) in all 14
   configurations of each set. Average linkage over segment-level embeddings at 0.85
   (`OfflineRefineConfig::default`) is above anything these embedders reach between two
   over-split clusters of the same speaker, so "refine on" equals "refine off" in every row.
   T10 (VBx/PLDA) replaces this pass. Until then, its threshold is dead weight.
3. **The segmenter is the largest single gain.** With the multilingual CAM++, DER drops from
   40.5 to 18.8 (es) and from 35.4 to 20.2 (VoxConverse). Without the segmenter, each chunk of
   4-20 s is labelled whole. That mixes turns, which raises confusion, and labels the pauses
   inside a chunk as speech, which gives about 7 % FA on `mls_es`. **End users currently run
   without it.** The catalog still downloads the upstream segmenter, which tract cannot load
   (T11), so their DER is in the "segmenter off" rows (about 35-40 %).
4. **The multilingual CAM++ beats the EN CAM++ on both sets**, even on English VoxConverse:
   18.8 vs 46.0 (es) and 20.2 vs 39.1 (VoxConverse), with the segmenter on. Both use the same
   0.85 threshold, so part of the gap may be calibration. The EN export is still the app's
   default when the user has not set a preference (`preferred_embed_model`).
5. **Miss is about 3 % with the segmenter on.** It comes from the VAD and segmenter
   boundaries and from backchannels shorter than the embedder minimum. FA is below 1 %.
   Neither is a priority compared with confusion.

## Reproduce

```
ECHO_LOG=warn echo-proto bench der --manifest <set>/manifest.txt \
  --embedder models/embedder/campplus_zh_en_common_advanced.onnx \
  --segmenter models/segmenter/pyannote_segmentation_3_tract.onnx \
  --vad <silero v6.2 inlined>.onnx [--no-refine] [--threshold X] \
  [--hyp-dir out/] [--json]
```

`ECHO_LOG=warn` keeps the tracing logs off stdout, so `--json` output stays parseable. A
manifest lists one `<wav> <rttm>` pair per line. The scratchpad scripts
`build_mls_convs.py`, `vc_subset.py` and `run_baseline.py` rebuild the sets and the run
matrix.

## Next

- T7: compare candidate embedders with this harness, each with its own calibrated threshold.
- Calibrate the multilingual CAM++ threshold on a held-out split, for example the MLS es
  `dev` split. Consider raising `max_speakers` or making it adaptive.
- `build_diarizer` should call `EmbedderFamily::cluster_config` so that the app and the
  benchmark cannot drift apart. That is a one-function change in `src-tauri`, not made here.
