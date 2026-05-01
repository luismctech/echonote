import { memo } from "react";

import { SpeakerChip } from "../../components/SpeakerChip";
import { formatTimestamp } from "../../lib/format";
import { shortTag } from "../../lib/speakers";
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
      <span className="w-12 shrink-0 font-mono text-xs tabular-nums text-zinc-500">
        {formatTimestamp(startMs)}
      </span>
      {speaker && (
        <SpeakerChip
          slot={speaker.slot}
          label={shortTag(speaker.slot)}
          compact
        />
      )}
      <span className="flex-1 min-w-0 break-all">{text.trim() || noSpeechLabel}</span>
    </>
  );
});
