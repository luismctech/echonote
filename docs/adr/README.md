# Architecture Decision Records

This directory holds every substantive architectural decision made for
EchoNote. Format follows [MADR](https://adr.github.io/madr/) 3.x.

## Lifecycle

1. A decision surfaces in design review or in a PR discussion.
2. Author drafts a new file `NNNN-short-title.md`, status `proposed`.
3. Tech Lead reviews, opens a PR and requests feedback.
4. Once merged with 2 approvals (including Tech Lead), status flips to
   `accepted`.
5. If a later ADR supersedes this one, edit the status to
   `superseded by ADR-XXXX` and add a forward link.

Only the **status** field of a historical ADR may change after merge.
Everything else is append-only — new ADRs must document the new
direction instead of rewriting the old record.

## Numbering

ADRs are numbered sequentially with a four-digit prefix (`0001`, `0002`, ...).
Gaps are not reused.

## Index

| ID | Title | Status |
|---|---|---|
| [ADR-0001](./0001-tauri-over-electron.md) | Tauri 2.x over Electron for the desktop shell | accepted |
| [ADR-0002](./0002-rust-plus-react-stack.md) | Rust + React + TypeScript as the base stack | accepted |
| [ADR-0003](./0003-whisper-cpp-over-faster-whisper.md) | whisper.cpp via whisper-rs for on-device ASR | accepted |
| [ADR-0007](./0007-diarization-via-onnx-eres2net.md) | Diarization via ONNX embeddings (3D-Speaker ERes2Net) with online clustering | accepted |

### Pending (see `DEVELOPMENT_PLAN.md` §14)

The following ADRs will be authored as the corresponding work starts:

- ADR-0004 — llama.cpp over Ollama as the embedded LLM runtime
- ADR-0005 — Separate audio tracks for microphone and system output
- ADR-0006 — Hybrid pipeline: streaming ASR + full-file refinement
- ADR-0008 — Zustand + TanStack Query over Redux for frontend state
- ADR-0009 — SQLite + FTS5 with optional SQLCipher for persistence
- ADR-0010 — Clean Architecture with ports and adapters as the layering
