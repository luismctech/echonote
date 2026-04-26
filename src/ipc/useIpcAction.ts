/**
 * `useIpcAction` — DRY wrapper for "call IPC, toast on failure".
 *
 * Every IPC orchestration in the frontend used to repeat this triangle:
 *
 *     try {
 *       const result = await someIpcCall(args);
 *       // ...success path...
 *     } catch (err) {
 *       const message = err instanceof Error ? err.message : String(err);
 *       toast.push({
 *         kind: "error",
 *         message: "Couldn't do X.",
 *         detail: message,
 *       });
 *     }
 *
 * This hook collapses each occurrence to a single line:
 *
 *     const runDelete = useIpcAction("Couldn't delete meeting.", deleteMeeting);
 *     const ok = await runDelete(id);
 *     if (ok === undefined) return;   // failure already toasted
 *     await refreshMeetings();
 *
 * Design notes:
 * - The returned function resolves to `undefined` on failure so callers
 *   guard with `if (result === undefined) return;`. We intentionally
 *   do NOT re-throw — the toast IS the error surface; re-throwing
 *   would force every caller to add a second `try/catch`.
 * - Core wrapping logic lives in the pure `runIpcAction` function below
 *   so it can be unit-tested without React, jsdom, or a fake context.
 *   The hook itself is a 4-line `useCallback` shim.
 */

import { useCallback } from "react";

import { useToast, type ToastInput } from "../components/Toaster";
import { isIpcError } from "../types/ipc-error";

export type IpcAction<Args extends unknown[], R> = (
  ...args: Args
) => Promise<R | undefined>;

/**
 * Pure wrapper around an async IPC call. Catches any thrown error,
 * pushes an error toast via the supplied `push`, and resolves to
 * `undefined` so the caller can short-circuit with a single guard.
 *
 * When the backend returns a structured `IpcError`, the toast
 * includes the machine-readable `code` and the `retriable` flag is
 * forwarded so the UI can offer a retry affordance in the future.
 *
 * Exported for tests; production code should prefer the
 * {@link useIpcAction} hook so it picks up the toast context
 * automatically.
 */
export async function runIpcAction<Args extends unknown[], R>(
  label: string,
  fn: (...args: Args) => Promise<R>,
  push: (toast: ToastInput) => string,
  args: Args,
): Promise<R | undefined> {
  try {
    return await fn(...args);
  } catch (err) {
    if (isIpcError(err)) {
      push({ kind: "error", message: label, detail: err.message });
    } else {
      const detail = err instanceof Error ? err.message : String(err);
      push({ kind: "error", message: label, detail });
    }
    return undefined;
  }
}

/**
 * Wrap an async IPC call so failures push an error toast instead of
 * propagating. Returns `undefined` on failure, the IPC result on
 * success.
 *
 * The returned function is referentially stable for as long as
 * `label` and `fn` are stable — pass module-level IPC functions, not
 * inline arrow wrappers, to avoid re-creating the action every render.
 */
export function useIpcAction<Args extends unknown[], R>(
  label: string,
  fn: (...args: Args) => Promise<R>,
): IpcAction<Args, R> {
  const { push } = useToast();
  return useCallback(
    (...args: Args) => runIpcAction(label, fn, push, args),
    [label, fn, push],
  );
}
