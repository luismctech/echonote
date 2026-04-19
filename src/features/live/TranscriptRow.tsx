import { SpeakerChip } from "../../components/SpeakerChip";
import { formatTimestamp } from "../../lib/format";
import { shortTag } from "../../lib/speakers";
import type { StreamLine } from "../../types/view";

/** One row in the live transcript scroller. */
export function TranscriptRow({ line }: { line: StreamLine }) {
  const ts = formatTimestamp(line.offsetMs);
  if (line.kind === "skipped") {
    return (
      <li className="flex gap-3 text-zinc-400">
        <span className="w-12 shrink-0 tabular-nums">{ts}</span>
        <span className="italic">silence (rms={line.rms.toFixed(4)})</span>
      </li>
    );
  }
  return (
    <li className="flex items-baseline gap-3">
      <span className="w-12 shrink-0 tabular-nums text-zinc-500">{ts}</span>
      {line.speakerSlot !== undefined && (
        <SpeakerChip
          slot={line.speakerSlot}
          label={shortTag(line.speakerSlot)}
          compact
        />
      )}
      <span className="flex-1">{line.text}</span>
      <span className="shrink-0 text-zinc-400">
        {line.language ?? "?"} · rtf {line.rtf.toFixed(2)}
      </span>
    </li>
  );
}
