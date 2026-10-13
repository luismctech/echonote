# Spike: a stronger speaker embedder than the multilingual CAM++

Date: 2026-10-12. Status: done for task T7 of `vad-diarization` (branch `feat/vad-diarization`).
Verdict: **no-go**. Nothing ships; the multilingual CAM++ stays the default. The models, eval data,
scripts and raw results live in the session scratchpad (`t7-*`, `der-eval/`) and are not committed.

## Question

Is there an embedder that diarizes better than the calibrated multilingual CAM++ (T6b: test DER
7.31 % Spanish / 5.56 % VoxConverse) and that the app may ship? The plan named WeSpeaker's
VoxBlink2 SimAM-ResNet34 (Vox1-O EER 0.415 % vs 0.803 % for CAM++). The user's rule: a model whose
licence forbids commercial use or redistribution is not shipped.

## Licence gate

| Model | Weights licence | Training data licence | Ship? |
|---|---|---|---|
| WeSpeaker VoxBlink2 SimAM-ResNet34 (and the VoxCeleb2-finetuned `_ft`, ResNet100) | No own licence; WeSpeaker states that "the pretrained model in WeNet follows the license of it's corresponding dataset" | VoxBlink2: **CC BY-NC-SA 4.0** (non-commercial, share-alike) | **No** |
| WeSpeaker ResNet34-LM (VoxCeleb2) | CC BY 4.0 (model card) | VoxCeleb: CC BY 4.0 | Yes, with attribution |
| WeSpeaker ResNet293-LM (VoxCeleb2) | CC BY 4.0 (model card) | VoxCeleb: CC BY 4.0 | Yes, with attribution |

Sources:

- WeSpeaker model licence rule and download list:
  <https://github.com/wenet-e2e/wespeaker/blob/master/docs/pretrained.md> ("Model License"). The
  VoxBlink2 models are only on ModelScope/wenet.org.cn, for example
  `https://wenet.org.cn/downloads?models=wespeaker&version=voxblink2_samresnet34.onnx`.
- VoxBlink2 dataset licence: <https://github.com/VoxBlink2/ScriptsForVoxBlink2/blob/main/LICENSE>
  and the README "License" section ("licensed under the **CC BY-NC-SA 4.0** license ... for
  non-commercial purposes"). The paper: <https://arxiv.org/abs/2407.11510>.
- WeSpeaker code: Apache-2.0 (<https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE>).
- ResNet34-LM: <https://huggingface.co/Wespeaker/wespeaker-resnet34-LM> (`license: cc-by-4.0`).
- ResNet293-LM: <https://huggingface.co/Wespeaker/wespeaker-voxceleb-resnet293-LM>
  (`license: cc-by-4.0`).

The VoxBlink2 models are **no-go for shipping**. They were not downloaded or benchmarked: the
NC licence allows non-commercial use only, and an evaluation to choose a model for this product
is not clearly non-commercial. Whether the project counts as non-commercial is the user's call.

Instead, two permissive WeSpeaker models were measured: ResNet34-LM (6.6 M parameters, about the
size of CAM++) and ResNet293-LM (Vox1-O EER 0.447 %, close to VoxBlink2).

## Technical fit

| Item | ResNet34-LM | ResNet293-LM |
|---|---|---|
| File (HF commit) | `voxceleb_resnet34_LM.onnx` (`f0c48c29`) | `voxceleb_resnet293_LM.onnx` (`6e6bffe5`) |
| Size / SHA-256 | 26,530,309 B / `7bb2f06e…c068` | 114,336,285 B / `dbb1ccc7…0e98` |
| SHA check | matches the HF LFS SHA-256 | matches the HF LFS SHA-256 |
| ONNX | opset 14, input `feats [B, T, 80]` (dynamic T), output `embs [B, 256]` | same, 296 Conv |
| tract 0.22.4 | loads and runs unmodified at T = 300 | loads and runs unmodified at T = 300 |

Both use the same `[1, T, 80]` fbank contract as the CAM++ adapter, so they ran through
`CamPlusPlusEmbedder` (300 frames, centre crop, probed width 256). Two front-end differences from
the 3D-Speaker models matter:

- **Waveform scale.** WeSpeaker computes Kaldi fbank on samples scaled to the int16 range
  (×32768). The app feeds `[-1, 1]` floats. After CMN a constant scale would cancel, but the
  `FLT_EPSILON` energy floor does not: embeddings at scale 1 vs 32768 have cosine 0.93 mean, 0.47
  minimum (onnxruntime, 30 three-second windows of `mls_esdev_conv00`). A WeSpeaker adapter must
  scale by 32768.
- **Window.** WeSpeaker trains with a Hamming window. The app's `mel_spec` fbank always uses
  Povey. At the same scale, Povey vs Hamming embeddings have cosine 0.995 mean, 0.990 minimum, so
  the window can stay.

The runs below scale the waveform by 32768 inside a throw-away build (an env-gated multiply in
the CAM++ adapter, reverted and never committed).

## DER

Harness and data as in `docs/spikes/2026-10-der-baseline.md`: `echo-proto bench der`, Silero
v6.2, the tract pyannote segmenter, collar 0.25 s, speaker cap 8. Calibration on the T6b dev sets
(`mls_es_dev`, `voxconverse_dev`) with the T6b objective (2 × ES + Vox) / 3, then one run on the
untouched T6 test sets.

### ResNet34-LM dev grid (cap 8)

| Online | Refine | ES DER | ES est. / true spk | Vox DER | Vox est. / true spk | Weighted |
|---|---|---|---|---|---|---|
| **0.55** | **0.40** | **4.45** | 7.20 / 3.45 | **14.23** | 7.00 / 3.67 | **7.71** |
| 0.60 | 0.40 | 4.46 | 7.15 / 3.45 | 14.83 | 7.33 / 3.67 | 7.91 |
| 0.55 | 0.50 | 5.60 | 7.40 / 3.45 | 14.29 | 7.25 / 3.67 | 8.49 |
| 0.60 | 0.30 | 4.34 | 6.80 / 3.45 | 16.92 | 6.83 / 3.67 | 8.53 |
| 0.50 | 0.40 | 4.63 | 6.40 / 3.45 | 17.48 | 6.58 / 3.67 | 8.91 |
| 0.45 | 0.40 | 5.56 | 5.75 / 3.45 | 15.76 | 6.25 / 3.67 | 8.96 |
| 0.45 | 0.50 | 5.85 | 5.80 / 3.45 | 15.23 | 6.42 / 3.67 | 8.98 |
| 0.40 | 0.40 | 9.38 | 4.80 / 3.45 | 15.11 | 5.75 / 3.67 | 11.29 |
| 0.35 | 0.40 | 11.42 | — | — | — | — |
| 0.30 | any | 20.19 | — | — | — | — |

The grid had 24 dev configurations (online 0.40-0.60 × refine 0.30/0.40/0.50, both sets) plus
the ES-only probes at 0.30-0.35. Multilingual CAM++ at its preset (0.45 / 8 / 0.50) scores 5.85 /
16.26 on dev (weighted 9.32). It estimates 4.75 / 5.42 speakers.

ResNet293-LM got one dev run at the CAM++ preset (0.45 / 8 / 0.50): ES DER 6.92 (estimate 5.70 /
3.45). It was not calibrated further because of its cost (see below).

### Test (untouched T6 sets, one run each)

| Embedder (online / cap / refine) | Set | DER % | Miss % | FA % | Conf % | Est. / true spk | Count exact | Median file DER % |
|---|---|---|---|---|---|---|---|---|
| Multilingual CAM++ (0.45 / 8 / 0.50) | `mls_es` | 7.31 | 2.94 | 0.70 | 3.67 | 4.15 / 2.80 | 3/20 | 6.30 |
| ResNet34-LM (0.55 / 8 / 0.40) | `mls_es` | **6.31** | 2.94 | 0.70 | 2.68 | 7.30 / 2.80 | 0/20 | 7.16 |
| Multilingual CAM++ (0.45 / 8 / 0.50) | `voxconverse` | **5.56** | 2.53 | 0.92 | 2.11 | 4.67 / 3.33 | 2/12 | 1.80 |
| ResNet34-LM (0.55 / 8 / 0.40) | `voxconverse` | 7.65 | 2.58 | 0.87 | 4.21 | 5.75 / 3.33 | 1/12 | 1.95 |

Weighted test DER: CAM++ 6.73, ResNet34-LM 6.76, a tie. ResNet34-LM wins 1.0 point on Spanish
pooled DER but loses on its median file, loses 2.1 points on VoxConverse, and its best
thresholds push nearly every Spanish recording to the cap of 8 speakers (7.30 estimated vs 2.80
true, 0/20 exact). The extra clusters are short, so DER barely sees them, but the transcript would
show them as extra speakers.

## Cost

Isolated runs on an Apple M1 Pro over two dev files (`mls_esdev_conv00`, `conv01`; 642 s), 227
embeddings of 3 s windows, nothing else running:

| Embedder | Embed latency mean / p95 | Embed RTF | Whole pipeline RTF (VAD + segmenter + embedder) |
|---|---|---|---|
| Multilingual CAM++ | 75.1 / 80.7 ms | 0.027 | 0.040 |
| ResNet34-LM | 59.8 / 65.6 ms | 0.021 | 0.034 |
| ResNet293-LM | 916.9 / 949.0 ms | 0.324 | 0.337 |

ResNet34-LM is slightly cheaper than CAM++. ResNet293-LM is 12× slower per window. That would
add about a second per segmenter window to live latency, which rules it out for the live path.

## Verdict

- **VoxBlink2 SimAM-ResNet34: no-go.** Its weights follow the VoxBlink2 dataset licence, CC
  BY-NC-SA 4.0, so they cannot ship.
- **ResNet34-LM: no-go.** It ships legally (CC BY 4.0) and runs in tract, but it is not better
  on held-out data (weighted test DER 6.76 vs 6.73), it is worse on VoxConverse, and it inflates
  speaker counts far more than CAM++.
- **ResNet293-LM: no-go.** Too slow for the live pipeline (embed RTF 0.32), and no better than
  ResNet34-LM in its one dev run.

No adapter or preset was added. The id dispatch landed anyway (`EmbedderFamily::from_model_id`,
`echo-proto bench der --embedder-id`), so a future embedder plugs in by catalog id. A WeSpeaker
adapter would need the ×32768 waveform scale.

If a WeSpeaker model is added later (for T11 hosting): ResNet34-LM is
`https://huggingface.co/Wespeaker/wespeaker-resnet34-LM/resolve/main/voxceleb_resnet34_LM.onnx`,
SHA-256 `7bb2f06e9df17cdf1ef14ee8a15ab08ed28e8d0ef5054ee135741560df2ec068`, 26,530,309 B, CC BY
4.0 (attribution required).

## Next

- T10 (VBx + PLDA) remains the lever for the speaker over-count. The calibrated CAM++ already
  over-counts by about 1.3 speakers; both ResNets over-count more.
- If the user decides the project may evaluate NC models, VoxBlink2 SimAM-ResNet34 could be
  measured locally with this harness for reference, still without shipping it.
