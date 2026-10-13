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

## Calibration (held-out)

Task T6b, 2026-10-12, at `5ab2484` (bench flags `--max-speakers` and `--refine-threshold`).
Embedder: multilingual CAM++, segmenter on, Silero v6.2, same harness as above. The values
were chosen on the **dev** sets only. The T6 sets above are the untouched **test** sets.

### Dev data

| Set | Content | Licence |
|---|---|---|
| `mls_es_dev` (synthetic) | 20 conversations built by the same generator with seed 20261013 and speaker counts drawn from {2, 2, 3, 3, 4, 5, 6}: 2 speakers 4, 3 speakers 9, 4 speakers 2, 5 speakers 4, 6 speakers 1. 89.9 min, 183-346 s each. 20 readers from the MLS Spanish **dev** split (`spanish/dev-00000-of-00001.parquet`, 157.1 MB, SHA-256 `f0ae1ca05f1c5cf4…`). MLS splits are speaker-disjoint: 0 readers shared with the test set. 1003 turns, 138 backchannels, 60 overlapped turns (33.7 s) | CC-BY-4.0 |
| `voxconverse_dev` (real) | 12 VoxConverse dev recordings, disjoint from the 12 test files: picked with seed 20261013 from the same pool (2-6 speakers, 2-8 min) after excluding the test files. 47.4 min, 2-6 speakers (mean 3.67). 78 MB range-extracted | CC-BY-4.0 |

New downloads: about 237 MB.

### Grid

Objective: pooled DER weighted 2:1 for `mls_es_dev` : `voxconverse_dev`, because the app is
Spanish-first. 150 runs in three stages:

1. Online similarity 0.30-0.70 (step 0.05) × `max_speakers` {4, 6, 8}, refine off.
2. Online {0.40, 0.45, 0.50, 0.55} × `max_speakers` {6, 8} × refine {0.40, 0.50, 0.60, 0.70}.
3. The same online and cap values × refine {0.30, 0.35}.

Stage 1 results (weighted dev DER %, refine off):

| Online | cap 4 | cap 6 | cap 8 |
|---|---|---|---|
| 0.30 | 23.63 | 23.20 | 23.20 |
| 0.35 | 17.61 | 14.83 | 14.83 |
| 0.40 | 13.35 | 9.52 | 9.52 |
| 0.45 | 13.93 | 11.06 | 10.00 |
| 0.50 | 14.52 | 10.54 | 9.48 |
| 0.55 | 16.09 | 11.01 | 10.72 |
| 0.60 | 17.62 | 11.56 | 11.91 |
| 0.65 | 17.17 | 12.81 | 13.24 |
| 0.70 | 18.33 | 15.12 | 15.16 |

Stages 2-3 with cap 8 (weighted dev DER %):

| Online \ refine | 0.35 | 0.40 | 0.50 | 0.60 | off |
|---|---|---|---|---|---|
| 0.40 | 10.22 | 9.52 | 9.52 | 9.52 | 9.52 |
| 0.45 | 9.76 | 9.08 | **9.32** | 10.00 | 10.00 |
| 0.50 | 9.68 | 9.05 | 8.80 | 9.48 | 9.48 |
| 0.55 | 10.36 | 9.83 | 10.05 | 10.72 | 10.72 |

Refine 0.70 equals 0.60 in every row. With cap 6, every point from 0.45 to 0.55 is worse than
with cap 8. The best cap-6 point is 0.50 / 0.50 at 9.91.

What the grid shows:

- A cap of 4 costs 3-6 DER points at every threshold. It merges the 5-6 speaker recordings.
- Below 0.40, DER rises sharply because distinct speakers merge (0.35: 14.8, 0.30: 23.2).
  Above 0.40, the curve is flat and then rises slowly as speakers over-split.
- The offline refine starts merging only at 0.50 or below. From 0.60 up it is a no-op, as it
  was at 0.85. At 0.35 it merges more, and DER goes up again.

### Chosen values

| Knob | Before | After |
|---|---|---|
| Online similarity threshold | 0.85 | **0.45** |
| `max_speakers` | 4 | **8** |
| Offline refine merge threshold | 0.85 | **0.50** |
| Online centroid merging | off (1.0) | off (1.0) |

The single dev minimum is 0.50 / 8 / 0.50 (8.80). The chosen point, 0.45 / 8 / 0.50 (9.32),
has the lowest 3×3 neighbourhood mean on the cap-8 grid: 9.37, with a worst neighbour of 10.00.
It also sits 0.05 above the cliff at 0.40 rather than on its edge. Only the multilingual CAM++
was calibrated. EN CAM++ (0.85) and ERes2Net (0.65) keep their old values in
`echo_diarize::presets`.

### Dev results

| Config | Set | DER % | Miss % | FA % | Conf % | Est. / true spk (mean) | Count exact | Median file DER % | Merges |
|---|---|---|---|---|---|---|---|---|---|
| 0.85 / 4 / 0.85 (old) | `mls_es_dev` | 17.89 | 0.62 | 0.98 | 16.29 | 4.00 / 3.45 | 2/20 | 10.81 | 0 |
| **0.45 / 8 / 0.50** | `mls_es_dev` | **5.85** | 0.63 | 0.98 | 4.24 | 4.75 / 3.45 | 5/20 | 3.79 | 2 |
| 0.50 / 8 / 0.50 (dev min) | `mls_es_dev` | 6.19 | 0.61 | 0.98 | 4.60 | 5.55 / 3.45 | 1/20 | 4.23 | 2 |
| 0.40 / 6 / 0.50 | `mls_es_dev` | 6.16 | 0.64 | 0.98 | 4.53 | 3.85 / 3.45 | 12/20 | 3.81 | 0 |
| 0.85 / 4 / 0.85 (old) | `voxconverse_dev` | 27.51 | 8.09 | 1.32 | 18.10 | 4.00 / 3.67 | 3/12 | 24.19 | 0 |
| **0.45 / 8 / 0.50** | `voxconverse_dev` | **16.26** | 7.20 | 1.45 | 7.61 | 5.42 / 3.67 | 0/12 | 12.38 | 0 |
| 0.50 / 8 / 0.50 (dev min) | `voxconverse_dev` | 14.02 | 7.18 | 1.45 | 5.38 | 6.00 / 3.67 | 0/12 | 10.64 | 0 |
| 0.40 / 6 / 0.50 | `voxconverse_dev` | 16.23 | 7.37 | 1.45 | 7.41 | 4.67 / 3.67 | 3/12 | 15.29 | 0 |

### Test results (untouched T6 sets)

| Config | Set | DER % | Miss % | FA % | Conf % | Est. / true spk (mean) | Count exact | Median file DER % |
|---|---|---|---|---|---|---|---|---|
| 0.85 / 4 / 0.85 (T6 app baseline) | `mls_es` | 18.76 | 2.96 | 0.70 | 15.10 | 4.00 / 2.80 | 3/20 | 10.65 |
| **0.45 / 8 / 0.50 (new preset)** | `mls_es` | **7.31** | 2.94 | 0.70 | 3.67 | 4.15 / 2.80 | 3/20 | 6.30 |
| 0.50 / 8 / 0.50 (dev min) | `mls_es` | 6.72 | 2.95 | 0.70 | 3.07 | 5.25 / 2.80 | 1/20 | 6.74 |
| 0.40 / 6 / 0.50 | `mls_es` | 7.17 | 2.95 | 0.70 | 3.53 | 3.50 / 2.80 | 7/20 | 6.29 |
| 0.85 / 4 / 0.85 (T6 app baseline) | `voxconverse` | 20.19 | 2.99 | 0.90 | 16.30 | 4.00 / 3.33 | 0/12 | 13.11 |
| **0.45 / 8 / 0.50 (new preset)** | `voxconverse` | **5.56** | 2.53 | 0.92 | 2.11 | 4.67 / 3.33 | 2/12 | 1.80 |
| 0.50 / 8 / 0.50 (dev min) | `voxconverse` | 6.98 | 2.53 | 0.95 | 3.51 | 5.17 / 3.33 | 2/12 | 2.09 |
| 0.40 / 6 / 0.50 | `voxconverse` | 6.65 | 2.57 | 0.89 | 3.20 | 4.00 / 3.33 | 5/12 | 1.74 |

Medians here are the mean of the two middle files, so the baseline medians differ slightly from
the T6 tables. Pooled numbers match T6 exactly. A run with the committed preset and no
overrides (`1c80958`) reproduced the new-preset rows on all four sets.

On the test sets, DER falls 2.6× on Spanish (18.76 → 7.31) and 3.6× on VoxConverse (20.19 →
5.56). Confusion falls from about 80 % of DER to about 40-50 %. Miss is now the largest part on
VoxConverse.

**Caveat: speaker counts.** The preset over-counts by about 1.3 speakers on average (4.15 vs
2.80 on `mls_es`). The extra clusters are short, so they cost little DER. A user will still see
them as extra speakers in the transcript. The 0.40 / 6 point counts best (7/20 and 5/12 exact)
for about the same test DER, but it sits on the edge of the merge cliff on dev. Reducing the
over-count is a job for T10 (VBx), which can split and re-assign speakers, not only merge them.

## T9 resegment (AHC) check

T9 (ADR 0001) adds `--refine-mode off|merge-only|resegment` to `bench der`. `resegment` runs
average-linkage AHC over the timestamped embedding log, one cluster per logged span, cut at the
preset refine threshold (0.50), and maps clusters back to online speakers by Hungarian overlap.
The bench applies the diarizer's span relabel plan to the recorded spans before scoring. One run
per mode on the **dev** sets, multilingual CAM++ preset (0.45 / cap 8 / refine 0.50), segmenter
on, Silero v6.2 speech-aligned chunks:

| Set | Mode | DER % | miss % | FA % | conf % | est / true spk | exact | relabelled spans |
|---|---|---|---|---|---|---|---|---|
| `mls_es_dev` | merge-only | **5.85** | 0.63 | 0.98 | 4.24 | 4.75 / 3.45 | 5/20 | 0 (2 merges) |
| `mls_es_dev` | resegment | 5.94 | 0.59 | 0.99 | 4.36 | 8.55 / 3.45 | 0/20 | 207 |
| `voxconverse_dev` | merge-only | **16.26** | 7.20 | 1.45 | 7.61 | 5.42 / 3.67 | 0/12 | 0 |
| `voxconverse_dev` | resegment | 17.59 | 6.70 | 1.45 | 9.44 | 9.33 / 3.67 | 0/12 | 138 |

Weighted dev DER (2:1): merge-only **9.32**, resegment 9.82. Merge-only reproduces the T6b dev
numbers exactly. Entry-level AHC at the speaker-level threshold leaves short, poorly embedded
spans as their own clusters, so the plan creates new speakers and the over-count roughly doubles.
The default stays `MergeOnly`. This check proves the plan end to end; it is not a candidate
default. T10 (VBx with PLDA, AHC initialisation and a minimum-cluster prior) is the intended
`Resegment` implementation. Raw JSON: scratchpad `der-eval/results/t9/`.

## T10 VBx check

T10 replaces the AHC behind `--refine-mode resegment` with VBx: VB-HMM over a PLDA fitted on
Multilingual LibriSpeech train speakers, with an AHC initialisation. Its parameters were
calibrated on the dev sets above, then the untouched test sets were run once. Details, grids and
costs are in `docs/spikes/2026-10-vbx.md`.

| Set | Merge-only DER % / est spk | VBx DER % / est spk | True spk |
|---|---|---|---|
| `mls_es_dev` | 5.85 / 4.75 | **3.59** / 3.50 | 3.45 |
| `voxconverse_dev` | 16.26 / 5.42 | **14.88** / 3.83 | 3.67 |
| `mls_es` (test) | **7.31** / 4.15 | 10.91 / **3.45** | 2.80 |
| `voxconverse` (test) | **5.56** / 4.67 | 6.10 / **3.08** | 3.33 |

Weighted test DER: merge-only 6.73, VBx 9.31. VBx fixes the over-count but loses on test DER, so
the ADR 0001 Q4 criterion is not met and the default stays merge-only.

## Reproduce

```
ECHO_LOG=warn echo-proto bench der --manifest <set>/manifest.txt \
  --embedder models/embedder/campplus_zh_en_common_advanced.onnx \
  --segmenter models/segmenter/pyannote_segmentation_3_tract.onnx \
  --vad <silero v6.2 inlined>.onnx [--refine-mode off|merge-only|resegment] [--threshold X] \
  [--max-speakers N] [--refine-threshold Y] [--hyp-dir out/] [--json]
```

`ECHO_LOG=warn` keeps the tracing logs off stdout, so `--json` output stays parseable. A
manifest lists one `<wav> <rttm>` pair per line. The scratchpad scripts
`build_mls_convs.py`, `vc_subset.py` and `run_baseline.py` rebuild the sets and the run
matrix.

## Next

- T7 (done, no-go): candidate embedders were compared with this harness, see
  `docs/spikes/2026-10-embedder-spike.md`. EN CAM++ and ERes2Net are still uncalibrated.
- T10 (done, default unchanged): VBx fixes the speaker over-count but not the test DER, see
  `docs/spikes/2026-10-vbx.md`.
- Done in T6b: `build_diarizer` builds the diarizer from `EmbedderFamily` (`a7f843a`), and the
  app defaults to the multilingual CAM++ when the user has no preference (`bb29136`).
