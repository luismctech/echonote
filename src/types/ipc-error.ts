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

/**
 * Human-readable text for any thrown value. Tauri commands reject with
 * a structured {@link IpcError} object, which `String(err)` renders as
 * `[object Object]`; this reads its `message` instead.
 */
export function errorMessage(err: unknown): string {
  if (err instanceof Error) return err.message;
  if (typeof err === "string") return err;
  if (typeof err === "object" && err !== null && "message" in err) {
    const { message } = err as { message: unknown };
    if (typeof message === "string") return message;
  }
  if (typeof err === "object" && err !== null) {
    try {
      return JSON.stringify(err);
    } catch {
      // Fall through for cyclic values.
    }
  }
  return String(err);
}

/**
 * Like {@link errorMessage}, but replaces backend errors the user can act
 * on with a translated explanation (backend messages are English).
 */
export function userErrorMessage(err: unknown, t: (key: string) => string): string {
  if (isIpcError(err) && err.code === "modelNotReady") {
    return t("errors.noLanguageModel");
  }
  return errorMessage(err);
}
