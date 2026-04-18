/**
 * Typed IPC client for the Tauri backend.
 *
 * Sprint 0 day 4 hand-rolls these types. Once the backend surface grows
 * (Sprint 1), generation will be delegated to `tauri-specta`, which emits
 * this file from Rust `#[specta::specta]` annotations. Hand-rolled shapes
 * here must match `src-tauri/src/commands.rs` one-for-one.
 */

import { invoke } from "@tauri-apps/api/core";

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

/** True when the frontend is running inside a Tauri webview. */
export const isTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;

export async function healthCheck(): Promise<HealthStatus> {
  return invoke<HealthStatus>("health_check");
}
