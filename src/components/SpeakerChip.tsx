import { paletteFor } from "../lib/speakers";

/**
 * Coloured pill identifying a speaker by slot.
 *
 * `compact` halves the padding for dense rows (the live transcript
 * scroller); the meeting-detail speakers panel uses the full size.
 * The colour is sourced from the slot palette so the same speaker
 * keeps its visual identity across the live and replay views.
 */
export function SpeakerChip({
  slot,
  label,
  compact,
}: {
  slot: number;
  label: string;
  compact?: boolean;
}) {
  const palette = paletteFor(slot);
  const sizing = compact ? "px-1.5 py-0 text-[10px]" : "px-2 py-0.5 text-xs";
  return (
    <span
      className={`inline-flex shrink-0 items-center rounded-full font-medium tabular-nums ring-1 ring-inset ${palette.bg} ${palette.text} ${palette.ring} ${sizing}`}
      title={`Speaker slot ${slot + 1}`}
    >
      {label}
    </span>
  );
}
