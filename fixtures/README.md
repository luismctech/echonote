# Bench fixtures

Phase-0 ASR fixtures used by `echo-proto bench wer`.

## Layout

```
fixtures/
├── transcripts/       ← gold (checked in, source of truth)
│   ├── 01_short_meeting.txt
│   ├── 02_technical_terms.txt
│   ├── 03_numbers_and_dates.txt
│   ├── 04_questions.txt
│   └── 05_long_passage.txt
└── audio/             ← generated locally (git-ignored)
    └── *.wav
```

The bench discovers pairs by basename: every `transcripts/<name>.txt`
must have a matching `audio/<name>.wav`.

## Generating audio (macOS)

WAV files are not committed (size + recoverability). Regenerate them
locally with:

```bash
./scripts/build-fixtures.sh
```

The script uses macOS `say` (default voice: `Samantha`) and `afconvert`
to produce 16 kHz mono PCM WAVs. Override the voice with
`VOICE=Daniel ./scripts/build-fixtures.sh`.

## Adding new fixtures

1. Drop the gold transcript in `fixtures/transcripts/<name>.txt`. Keep
   it concise (1–4 sentences) so a single Whisper pass stays fast.
2. Re-run `./scripts/build-fixtures.sh`. Existing WAVs are kept if
   newer than their transcript, so you only pay for the new one.
3. Run `cargo run -p echo-proto -- bench wer` to update the baseline.

## Why synthetic audio

Phase 0 is about catching **regressions** in the streaming pipeline,
the resampler, and the Whisper integration — not about benchmarking
Whisper itself against natural speech. Synthetic audio:

- is reproducible across machines,
- does not raise copyright / consent issues,
- already exposes the failure modes we care about
  (resampling, chunking boundaries, end-pointing).

Real-speech benchmarks land in **Sprint 2** alongside the diarizer.
