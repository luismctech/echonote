import { memo } from "react";

import { SpeakerChip } from "../../components/SpeakerChip";
import { formatTimestamp } from "../../lib/format";
import { chipLabel } from "../../lib/speakers";
import type { Speaker } from "../../types/speaker";

export type SegmentRowProps = {
  startMs: number;
  text: string;
  speaker: Speaker | undefined;
  noSpeechLabel: string;
};

/** Single segment row in the meeting detail view. */
export const SegmentRow = memo(function SegmentRow({
  startMs,
  text,
  speaker,
  noSpeechLabel,
}: SegmentRowProps) {
  return (
    <>
      <span className="w-12 shrink-0 font-mono text-ui-sm tabular-nums text-content-tertiary">
        {formatTimestamp(startMs)}
      </span>
      {speaker && (
        <SpeakerChip
          slot={speaker.slot}
          label={chipLabel(speaker)}
          compact
        />
      )}
      <span className="flex-1 min-w-0 break-all">{text.trim() || noSpeechLabel}</span>
    </>
  );
});
