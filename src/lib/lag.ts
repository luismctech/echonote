/**
 * Processing-lag indicator for the live view.
 *
 * The backend reports how far transcription runs behind capture
 * (`progress` events). Small lags are normal (a chunk is only
 * transcribed once it closes), so the indicator stays hidden until the
 * lag is clearly noticeable.
 */

/** Lag, in milliseconds, up to which the indicator stays hidden. */
export const LAG_INDICATOR_THRESHOLD_MS = 3000;

/**
 * Whole seconds of lag to show, or `null` when the indicator should be
 * hidden (lag at or below {@link LAG_INDICATOR_THRESHOLD_MS}).
 */
export function lagIndicatorSeconds(lagMs: number): number | null {
  if (lagMs <= LAG_INDICATOR_THRESHOLD_MS) return null;
  return Math.round(lagMs / 1000);
}
