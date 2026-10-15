/**
 * Pure helpers for filtering a meeting transcript by speaker.
 *
 * The filter is a set of speaker ids: empty means "show everything",
 * otherwise only segments owned by one of the selected speakers are
 * shown. Every helper returns a new value (never mutates its input) so
 * it can feed React state directly.
 */

/** Add `speakerId` to the filter, or remove it when already selected. */
export function toggleSpeaker(filter: ReadonlySet<string>, speakerId: string): Set<string> {
  const next = new Set(filter);
  if (next.has(speakerId)) next.delete(speakerId);
  else next.add(speakerId);
  return next;
}

/**
 * Segments visible under `filter`, in their original order. Returns the
 * input array itself when the filter is empty so memoised consumers do
 * not re-render. Segments without a speaker are hidden while filtering.
 */
export function filterSegmentsBySpeakers<T extends { speakerId: string | null }>(
  segments: readonly T[],
  filter: ReadonlySet<string>,
): readonly T[] {
  if (filter.size === 0) return segments;
  return segments.filter((seg) => seg.speakerId !== null && filter.has(seg.speakerId));
}

/**
 * Drop filter ids that no longer exist (a speaker merged away or left
 * without segments). Returns the same set when nothing was dropped so a
 * state update is a no-op.
 */
export function pruneSpeakerFilter(
  filter: ReadonlySet<string>,
  presentIds: Iterable<string>,
): ReadonlySet<string> {
  if (filter.size === 0) return filter;
  const present = new Set(presentIds);
  const next = new Set([...filter].filter((id) => present.has(id)));
  return next.size === filter.size ? filter : next;
}
