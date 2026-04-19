/**
 * Debounce primitive used by the meetings search box.
 *
 * Returns a stable value that only updates after `delayMs` of input
 * inactivity. Wraps `setTimeout` instead of `requestIdleCallback`
 * because the window is small (≈ 200 ms) and we want predictable
 * timing across browsers — the FTS query is cheap, so we err on the
 * side of "feel responsive" over "save the CPU a few cycles".
 *
 * Lives outside `lib/ipc.ts` so it can be reused by other inputs
 * later (e.g. chat composer typing indicators in Sprint 1 day 9+).
 */
import { useEffect, useState } from "react";

export function useDebouncedValue<T>(value: T, delayMs: number): T {
  const [debounced, setDebounced] = useState(value);

  useEffect(() => {
    const handle = window.setTimeout(() => setDebounced(value), delayMs);
    return () => window.clearTimeout(handle);
  }, [value, delayMs]);

  return debounced;
}
