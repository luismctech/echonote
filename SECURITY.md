# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| latest `develop` | Yes (pre-release) |
| < v1.0 | Not yet released |

## Reporting a vulnerability

EchoNote is a privacy-first product. We take security and privacy issues
seriously.

**Please do NOT open a public GitHub issue for security concerns.**

Instead, email **albertomzcruz@gmail.com** with:

1. A description of the vulnerability or privacy concern.
2. Steps to reproduce (or a proof of concept).
3. The impact you believe it has.
4. Any suggested fix, if you have one.

You will receive an acknowledgement within **48 hours** and a detailed
response within **5 business days** with next steps.

## Scope

The following are explicitly in scope:

- Any code in this repository.
- Audio data handling and storage.
- Local database encryption and key management.
- IPC surface between the Rust backend and the webview frontend.
- Auto-update signature verification.
- Any unintended network communication (EchoNote should make zero network
  calls during normal operation).

## Disclosure policy

We follow coordinated disclosure:

1. Reporter notifies us privately.
2. We confirm, triage and develop a fix.
3. We release a patched version and publish a security advisory.
4. Reporter is credited (unless they prefer anonymity).

We aim to resolve critical issues within **14 days** of confirmation.

## Privacy principles

EchoNote processes sensitive meeting audio. Our core privacy commitments:

- All processing happens on-device. No audio, transcripts or summaries
  leave the machine unless the user explicitly exports them.
- No telemetry, analytics or crash reporting is sent without explicit
  opt-in.
- Database contents are encrypted at rest when SQLCipher is enabled.
- ML model weights are downloaded once from documented URLs and verified
  via SHA-256 checksums.
