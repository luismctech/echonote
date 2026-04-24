/**
 * Structured IPC error type returned by every Tauri command.
 *
 * Mirrors the Rust `IpcError` struct in `src-tauri/src/ipc_error.rs`.
 * The backend serializes errors as JSON objects on the IPC wire, so
 * the frontend receives a machine-readable `code` alongside the
 * human-readable `message`.
 */

/** Machine-readable error codes matching the Rust `ErrorCode` enum. */
export type ErrorCode =
  | "notFound"
  | "storage"
  | "llm"
  | "asr"
  | "invalidInput"
  | "sessionConflict"
  | "modelNotReady"
  | "audio"
  | "vad"
  | "diarization"
  | "network"
  | "internal";

/** Structured error returned by every Tauri command. */
export interface IpcError {
  code: ErrorCode;
  message: string;
  retriable: boolean;
}

/**
 * Type guard: check whether an unknown thrown value is a structured
 * `IpcError` (as opposed to a plain string or generic Error).
 */
export function isIpcError(err: unknown): err is IpcError {
  return (
    typeof err === "object" &&
    err !== null &&
    "code" in err &&
    "message" in err &&
    "retriable" in err
  );
}
