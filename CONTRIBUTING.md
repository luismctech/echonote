# Contributing to EchoNote

Thanks for taking the time to contribute. This document is the single source
of truth for how work lands in this repository.

If anything here is unclear or out of date, open a PR — improving the
onboarding path is itself a valid contribution.

## Table of contents

1. [Code of conduct](#1-code-of-conduct)
2. [Local setup](#2-local-setup)
3. [Commit messages](#3-commit-messages)
4. [Branching model](#4-branching-model)
5. [Pull requests](#5-pull-requests)
6. [Architecture Decision Records](#6-architecture-decision-records)
7. [Tests and quality gates](#7-tests-and-quality-gates)
8. [Security and privacy disclosures](#8-security-and-privacy-disclosures)
9. [Releases](#9-releases)

## 1. Code of conduct

All contributors are expected to follow our
[Code of Conduct](./CODE_OF_CONDUCT.md). Report unacceptable behaviour via
the channel listed in that document.

## 2. Local setup

### Prerequisites

- **macOS 13+** (primary dev target) or **Linux** (Ubuntu 22.04+ tested).
  Windows is supported via WSL2 only.
- **Xcode Command Line Tools** on macOS (`xcode-select --install`).
- **Rust 1.88.0+** (pinned via `rust-toolchain.toml`).
- **Node.js 20 LTS** and **pnpm 9** (needed from Sprint 0 day 4 onwards).

### First run

```bash
git clone https://github.com/AlbertoMZCruz/echonote.git
cd echonote
./scripts/bootstrap.sh
```

`bootstrap.sh` is idempotent. It verifies your toolchain, wires up the
versioned git hooks in `.githooks/`, and prints actionable guidance for
anything missing. It never installs packages without your consent.

### Day-to-day

```bash
cargo build --workspace          # incremental build
cargo test  --workspace          # all tests
cargo clippy --workspace --all-targets -- -D warnings
cargo fmt --all                  # auto-format
```

The Phase 0 CLI prototype lives in `crates/echo-proto`:

```bash
cargo run -p echo-proto -- --help
```

## 3. Commit messages

We use [Conventional Commits](https://www.conventionalcommits.org/) and
enforce them via the `commit-msg` git hook.

```
<type>(<scope>)?!?: <subject>

<body>

<footer>
```

### Allowed types

| Type | When to use |
|---|---|
| `feat`     | New user-facing capability |
| `fix`      | Bug fix observable by a user |
| `perf`     | Performance improvement |
| `refactor` | Internal restructuring, no behavioural change |
| `docs`     | Documentation only |
| `test`     | Tests only (new, refactored or fixed) |
| `build`    | Build system, toolchain, Cargo manifests |
| `ci`       | GitHub Actions, Dependabot, repo automation |
| `chore`    | Housekeeping that does not fit the above |
| `deps`     | Dependency bumps (Dependabot uses this prefix) |
| `style`    | Formatting, whitespace, pure cosmetic |
| `revert`   | Reverts a previous commit |

### Scope

Optional but encouraged. Use the crate, module, or subsystem:

- `audio`, `asr`, `diarize`, `llm`, `storage`, `app`, `domain`, `proto`
- `ui`, `adr`, `toolchain`, `ci`

### Subject

- Imperative mood ("add", not "added").
- Start lower case, no trailing period.
- Hard limit of 100 characters; target 72.

### Breaking changes

Append `!` after the type/scope and add a `BREAKING CHANGE:` footer:

```
feat(domain)!: rename Segment.end to Segment.ended_at

BREAKING CHANGE: persisted meetings from before commit abcd1234 must be
migrated via `echo-proto migrate v2`.
```

### Examples

```
feat(audio): capture system output via ScreenCaptureKit
fix: resume transcription after OS sleep
docs(adr): add ADR-0004 for llama.cpp runtime
ci: pin actions/checkout to v4.2.0
deps: bump tokio 1.47 -> 1.48
```

## 4. Branching model

We follow a simplified Git Flow:

- `main` is the **production** branch. Only tagged releases land here.
  Protected — no direct pushes, PR required, CI must be green, linear
  history, conversation resolution required.
- `develop` is the **integration** branch and the default target for PRs.
- `feat/*`, `fix/*`, `docs/*`, `chore/*` are **short-lived** branches cut
  from `develop`. Keep them under 5 days old or rebase onto `develop`.
- `release/*` branches are cut from `develop` when we stabilise a version.
- `hotfix/*` branches are cut from `main` for emergency production fixes
  and merged back into both `main` and `develop`.

All merges are **squash** merges. History on `main` and `develop` is
linear.

## 5. Pull requests

- Target `develop` unless you are opening a hotfix.
- Fill in the [PR template](./.github/PULL_REQUEST_TEMPLATE.md) completely.
- Keep PRs under ~400 changed lines when possible. Split larger changes.
- Link to the issue, ADR or ticket that motivates the change.
- Include a short demo (screenshot, asciicast, or sample command output)
  for user-visible changes.
- CI must be green. You may not merge around a red status check.
- Respond to review comments by either pushing a change or explaining why
  you disagree. Conversations must be resolved before merge.

### Review expectations

- At least one approval from CODEOWNERS (currently the Tech Lead).
- Security-sensitive changes (crypto, IPC surface, storage format, update
  channel) require a second reviewer from the security rotation.

## 6. Architecture Decision Records

Any change that meaningfully alters the architecture deserves an ADR. See
[`docs/adr/README.md`](./docs/adr/README.md) for the lifecycle, format
(MADR 3.x) and numbering rules.

If you are not sure whether a change needs an ADR, open a draft PR and ask.
The cost of an unnecessary ADR is one file; the cost of a missing one is
a team-wide argument six months later.

## 7. Tests and quality gates

Every PR must pass the full CI pipeline:

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets -- -D warnings`
- `cargo test --workspace --all-targets --locked`
- `cargo build --workspace --locked` (macOS)
- `cargo check --workspace --all-targets --locked` (Linux)

Target test coverage per `DEVELOPMENT_PLAN.md` §6:

- Domain crate: **90 %+ line coverage**, pure unit tests, no IO.
- Application crate: **80 %+ line coverage** with mocked ports.
- Infrastructure crates: exercised via integration tests against real
  adapters when feasible, mocked otherwise.
- UI: component tests with Vitest + Testing Library, critical flows with
  Playwright end-to-end (Sprint 2 onwards).

The pre-commit hook runs `fmt` and `clippy` locally to catch issues before
they hit CI. Do not `--no-verify` routinely.

## 8. Security and privacy disclosures

EchoNote is a privacy-first product. Please do **not** file public issues
for security or privacy concerns. Follow the process in
[`SECURITY.md`](./SECURITY.md).

## 9. Releases

Releases are cut from `release/*` branches, tagged `vX.Y.Z` on `main`,
and distributed as signed installers. See `DEVELOPMENT_PLAN.md` §7 for
the full release checklist (to be expanded before v1.0).
