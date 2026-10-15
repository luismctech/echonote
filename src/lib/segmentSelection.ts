/**
 * Pure helpers for multi-selecting transcript segments.
 *
 * Anchor semantics follow file managers: a plain click toggles one
 * segment and moves the anchor to it; a shift-click adds every segment
 * between the anchor and the clicked one (inclusive, in the currently
 * visible order) and keeps the anchor, so successive shift-clicks extend
 * from the same point. Without a visible anchor a shift-click behaves as
 * a plain click.
 */

/** Selected segment ids plus the anchor used for shift-click ranges. */
export type SegmentSelection = {
  readonly selected: ReadonlySet<string>;
  readonly anchor: string | null;
};

/** Selection with nothing selected and no anchor. */
export const EMPTY_SELECTION: SegmentSelection = { selected: new Set(), anchor: null };

/** Toggle one segment and make it the anchor. */
export function toggleSegment(state: SegmentSelection, id: string): SegmentSelection {
  const selected = new Set(state.selected);
  if (selected.has(id)) selected.delete(id);
  else selected.add(id);
  return { selected, anchor: id };
}

/**
 * Add the segments between the anchor and `id` (inclusive) in
 * `visibleIds` order to the selection. Falls back to
 * {@link toggleSegment} when there is no anchor or it is not visible.
 */
export function selectRange(
  state: SegmentSelection,
  visibleIds: readonly string[],
  id: string,
): SegmentSelection {
  const from = state.anchor === null ? -1 : visibleIds.indexOf(state.anchor);
  const to = visibleIds.indexOf(id);
  if (from < 0 || to < 0) return toggleSegment(state, id);
  const selected = new Set(state.selected);
  for (const id of visibleIds.slice(Math.min(from, to), Math.max(from, to) + 1)) {
    selected.add(id);
  }
  return { selected, anchor: state.anchor };
}

/** Add every visible segment to the selection, keeping the anchor. */
export function selectAllVisible(
  state: SegmentSelection,
  visibleIds: readonly string[],
): SegmentSelection {
  return { selected: new Set([...state.selected, ...visibleIds]), anchor: state.anchor };
}

/**
 * Keep only segments in `presentIds` (after a delete or a filter
 * change). Clears the anchor if it was dropped. Returns the same state
 * when nothing changed so a state update is a no-op.
 */
export function pruneSelection(
  state: SegmentSelection,
  presentIds: Iterable<string>,
): SegmentSelection {
  if (state.selected.size === 0 && state.anchor === null) return state;
  const present = new Set(presentIds);
  const selected = new Set([...state.selected].filter((id) => present.has(id)));
  const anchor = state.anchor !== null && present.has(state.anchor) ? state.anchor : null;
  if (selected.size === state.selected.size && anchor === state.anchor) return state;
  return { selected, anchor };
}
