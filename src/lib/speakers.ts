/**
 * Speaker rendering helpers shared by the live and meeting-detail views.
 *
 * The diarizer assigns every speaker a 0-based `slot` in arrival
 * order. We want a stable colour per slot so the same speaker keeps
 * its visual identity across renames and across reloads of the same
 * meeting (the slot survives both, the SpeakerId may not survive a
 * future re-cluster). The palette lives here, not in App.tsx, so the
 * mapping is unit-testable and so any other surface (export to PDF,
 * future electron viewer, …) can reuse it.
 *
 * Colours are picked from the Tailwind palette so they read well
 * against both the light (`bg-surface-sunken`) and dark transcript
 * transcript backgrounds without per-theme overrides.
 */

import type { Speaker } from "../types/speaker";

/**
 * Tailwind class pairs for each palette slot. Indexed modulo the
 * palette length, so very-busy meetings (>8 speakers) cycle the
 * colours; the slot number stays in the chip text in those cases so
 * users can still tell two same-coloured speakers apart.
 */
export const SPEAKER_PALETTE = [
  { bg: "bg-emerald-100 dark:bg-emerald-900/40", text: "text-emerald-800 dark:text-emerald-200", ring: "ring-emerald-300 dark:ring-emerald-700" },
  { bg: "bg-sky-100      dark:bg-sky-900/40",     text: "text-sky-800      dark:text-sky-200",     ring: "ring-sky-300      dark:ring-sky-700" },
  { bg: "bg-amber-100    dark:bg-amber-900/40",   text: "text-amber-800    dark:text-amber-200",   ring: "ring-amber-300    dark:ring-amber-700" },
  { bg: "bg-violet-100   dark:bg-violet-900/40",  text: "text-violet-800   dark:text-violet-200",  ring: "ring-violet-300   dark:ring-violet-700" },
  { bg: "bg-rose-100     dark:bg-rose-900/40",    text: "text-rose-800     dark:text-rose-200",    ring: "ring-rose-300     dark:ring-rose-700" },
  { bg: "bg-teal-100     dark:bg-teal-900/40",    text: "text-teal-800     dark:text-teal-200",    ring: "ring-teal-300     dark:ring-teal-700" },
  { bg: "bg-fuchsia-100  dark:bg-fuchsia-900/40", text: "text-fuchsia-800  dark:text-fuchsia-200", ring: "ring-fuchsia-300  dark:ring-fuchsia-700" },
  { bg: "bg-orange-100   dark:bg-orange-900/40",  text: "text-orange-800   dark:text-orange-200",  ring: "ring-orange-300   dark:ring-orange-700" },
] as const;

export type SpeakerPaletteEntry = (typeof SPEAKER_PALETTE)[number];

/** Stable palette entry for a slot. Cycles past `SPEAKER_PALETTE.length`. */
export function paletteFor(slot: number): SpeakerPaletteEntry {
  // `>>> 0` clamps NaN/negative to a non-negative integer so the
  // modulo always lands in-bounds even when the backend hands us a
  // surprising slot value. The non-null assertion is safe because
  // `idx` is provably `0..SPEAKER_PALETTE.length` and the palette is
  // a non-empty `as const` tuple, but TS's noUncheckedIndexedAccess
  // does not see through that.
  const idx = (slot >>> 0) % SPEAKER_PALETTE.length;
  return SPEAKER_PALETTE[idx]!;
}

/**
 * Display name for a speaker. Mirrors `Speaker::display_name` on the
 * Rust side: trimmed user label when present, otherwise
 * `Speaker {slot+1}`. Whitespace-only labels fall back to anonymous
 * so the UI never renders an empty chip.
 */
export function displayName(speaker: Pick<Speaker, "slot" | "label">): string {
  const label = speaker.label?.trim();
  if (label && label.length > 0) return label;
  return `Speaker ${speaker.slot + 1}`;
}

/** Short tag used in dense rows (live transcript): `S1`, `S2`, … */
export function shortTag(slot: number): string {
  return `S${(slot >>> 0) + 1}`;
}

/**
 * Build a quick `speakerId → Speaker` map for O(1) chip lookups while
 * rendering segments. Returns an empty map when `speakers` is empty
 * so callers can use `?.` without guards.
 */
export function indexSpeakers(speakers: readonly Speaker[]): Map<string, Speaker> {
  const m = new Map<string, Speaker>();
  for (const s of speakers) m.set(s.id, s);
  return m;
}
