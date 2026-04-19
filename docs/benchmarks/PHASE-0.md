# Phase 0 — benchmark baseline

This document captures the first WER and LLM measurements taken during
Sprint 0. It is the baseline that all subsequent sprints compare against
when changing the resampler, the streaming pipeline, the Whisper version
or the model selection.

> Reproduce locally: `./scripts/build-fixtures.sh && cargo run --release -p echo-proto -- bench wer`

## Host

| Field            | Value                                |
|------------------|--------------------------------------|
| Hardware         | Apple M1 Pro (arm64)                 |
| OS               | macOS 26.4.1                         |
| Whisper backend  | whisper.cpp (Metal acceleration on)  |
| Whisper model    | `ggml-base.en.bin` (147 MB)          |
| Toolchain        | rustc 1.88.0 (release profile)       |
| Date             | 2026-04-18                           |

## Fixtures

5 synthetic English clips generated with the macOS `say` command (voice
`Samantha`). See `fixtures/README.md` for the rationale and how to add
more clips. Total audio: **84.7 s**.

## ASR (`bench wer`)

| Clip                    | Ref | Hyp |  S |  D |  I |  WER  |  RTF  |
|-------------------------|----:|----:|---:|---:|---:|------:|------:|
| 01_short_meeting        |  38 |  39 |  1 |  0 |  1 |  5.26 % | 0.03 |
| 02_technical_terms      |  41 |  42 |  4 |  0 |  1 | 12.20 % | 0.02 |
| 03_numbers_and_dates    |  42 |  36 |  5 |  6 |  0 | 26.19 % | 0.02 |
| 04_questions            |  37 |  36 |  1 |  1 |  0 |  5.41 % | 0.02 |
| 05_long_passage         |  92 |  91 |  0 |  1 |  0 |  1.09 % | 0.02 |

**Aggregate**

| Metric          | Value     |
|-----------------|-----------|
| Global WER      | **8.40 %** |
| RTF p50         | 0.02      |
| RTF p95         | 0.03      |
| Total audio     | 84.7 s    |
| Total elapsed   | 1.8 s     |

### Notes on individual clips

- **03_numbers_and_dates** drives most of the global error. Numbers
  written as words in the reference (`forty seven thousand`) come back
  as digits in the hypothesis (`$47,000`). After normalization those
  count as substitutions and deletions. This is a known Whisper
  preference, not a regression. We will revisit when we add a number
  normalization step in Sprint 1.
- **05_long_passage** is the cleanest clip and the longest one — a good
  signal that Whisper handles longer windows well, which matters for
  the 5 s streaming chunks.
- **All clips finish in ≤ 0.03 RTF** on Metal. We have plenty of headroom
  for the streaming pipeline, which targets RTF ≤ 0.5 even on older
  hardware.

## LLM (`bench llm`)

The CLI scaffolding is in place but the `echo-llm` adapter has not been
wired yet. The contract is:

- Inputs: every transcript under `fixtures/transcripts/*.txt`.
- Metrics to capture per run: tokens/s, time-to-first-token, total
  latency, peak RSS.
- Targets: Qwen2.5-3B Q4_K_M as the default model; document tradeoffs
  vs Phi-3.5-mini and Llama-3.2-3B.

This benchmark goes live in **Sprint 1** alongside the chat use case.

## Quality gates

| Gate                                  | Threshold | Current  | Status |
|---------------------------------------|-----------|----------|--------|
| Phase-0 ASR global WER                | ≤ 25 %    | 8.40 %   | PASS   |
| Streaming RTF (live) on Apple Silicon | ≤ 0.5     | ~0.08    | PASS   |
| Bench RTF p95 on Apple Silicon        | ≤ 0.5     | 0.03     | PASS   |

When a gate regresses by more than 10 % relative on the baseline above,
open an issue tagged `regression` before merging the offending PR.

## How CI uses this

The `.github/workflows/bench.yml` workflow is opt-in
(`workflow_dispatch`). Trigger it from the GitHub UI or via:

```bash
gh workflow run bench.yml -f whisper_model=base.en -f max_wer=0.25
```

The job builds `echo-proto --release`, regenerates the fixtures, downloads
the chosen `ggml-*.bin`, runs `bench wer` with the requested gate, and
uploads `target/bench-reports/wer-<model>.json` as an artifact.
