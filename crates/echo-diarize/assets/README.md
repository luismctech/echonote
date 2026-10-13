# echo-diarize assets

## `plda_campplus_multi.plda`

LDA + two-covariance PLDA back-end of the VBx re-segmentation for the
multilingual CAM++ embedder (`campplus_zh_en_common_advanced.onnx`, catalog id
`embedder-campplus-multi`, 192 dimensions). It is compiled into the binary
(`Plda::campplus_multilingual`); the format is documented in `src/plda.rs`.

| Item | Value |
|---|---|
| Size | 166,164 bytes |
| SHA-256 | `928d73eb4f526d0a554ba6cfa29d19d79c9927b11e47f4b3ac06a9d646203558` |
| Dimensions | 192 -> 128 (LDA), PLDA on the 128-dim length-normalised space |
| Fitted with | `scripts/fit-plda.py --lda-dim 128` (numpy 2.5.4, scipy 1.18.1); re-running it on the same embeddings gives the same bytes |
| Embeddings | `echo-proto bench embed-dump` with the app's CAM++ adapter |
| Training data | 4,176 crops of 1.5-6 s (24 per speaker) from 174 speakers of the **Multilingual LibriSpeech train splits**: Spanish 70, Italian 48, French 45, Portuguese 11 |
| Selection | One parquet row group per speaker of `facebook/multilingual_librispeech` on Hugging Face, read with HTTP range requests; crops seeded (20261013/20261014) |

### Licence and attribution

Multilingual LibriSpeech (MLS) is © its authors (Pratap et al., "MLS: A
Large-Scale Multilingual Dataset for Speech Research", Interspeech 2020),
derived from LibriVox public-domain audiobooks, and released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). This file contains
only statistics derived from it (means, an LDA projection, covariance
eigenvectors and eigenvalues), no audio and no per-speaker data.

The MLS train speakers are disjoint from the MLS dev and test speakers used by
the DER evaluation sets (`docs/spikes/2026-10-der-baseline.md`); the overlap
was checked to be zero.
