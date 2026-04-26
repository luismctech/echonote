/**
 * Pure presentation formatters shared by every view.
 *
 * Kept dependency-free (no React, no IPC, no Tailwind) so they're
 * trivially unit-testable and reusable across the live transcript,
 * stored meeting detail, and sidebar list.
 */

/** `mm:ss` from a millisecond offset. Falsy / NaN inputs render as `00:00`. */
export function formatTimestamp(ms: number): string {
  const safe = Number.isFinite(ms) ? Math.max(0, ms) : 0;
  const totalSeconds = Math.floor(safe / 1000);
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m.toString().padStart(2, "0")}:${s.toString().padStart(2, "0")}`;
}

/** Short locale-aware date+time. Falls back to the input when unparseable. */
export function formatDate(rfc3339: string): string {
  const d = new Date(rfc3339);
  if (Number.isNaN(d.getTime())) return rfc3339;
  return d.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/** Human-readable duration: `30s` or `5m 03s`. */
export function formatDurationMs(ms: number): string {
  const totalSeconds = Math.round(ms / 1000);
  if (totalSeconds < 60) return `${totalSeconds}s`;
  const m = Math.floor(totalSeconds / 60);
  const s = totalSeconds % 60;
  return `${m}m ${s.toString().padStart(2, "0")}s`;
}
