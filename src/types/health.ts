/**
 * Backend health payload.
 *
 * Mirrors the Rust struct returned by `health_check` in
 * `src-tauri/src/commands.rs`. When `tauri-specta` codegen ships
 * (Sprint 1+), this file will be regenerated from the Rust source.
 */

export type HealthStatus = {
  /** ISO 8601 instant the backend answered. */
  timestamp: string;
  /** EchoNote semver, pulled from Cargo.toml at compile time. */
  version: string;
  /** Target triple the backend was compiled for. */
  target: string;
  /** Short git hash, `unknown` outside CI or when .git is absent. */
  commit: string;
};
