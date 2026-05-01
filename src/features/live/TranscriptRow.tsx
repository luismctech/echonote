import { memo } from "react";
import { useTranslation } from "react-i18next";

import { SpeakerChip } from "../../components/SpeakerChip";
import { formatTimestamp } from "../../lib/format";
import { shortTag } from "../../lib/speakers";
import type { StreamLine } from "../../types/view";

/** One row in the live transcript scroller. */
export const TranscriptRow = memo(function TranscriptRow({
  line,
}: {
  line: StreamLine;
}) {
  const { t } = useTranslation();
  const ts = formatTimestamp(line.offsetMs);
  if (line.kind === "skipped") {
    return (
      <>
        <span className="w-12 shrink-0 tabular-nums">{ts}</span>
        <span className="italic">{t("live.silenceRms", { rms: line.rms.toFixed(4) })}</span>
      </>
    );
  }
  return (
    <>
      <span className="w-12 shrink-0 tabular-nums text-content-tertiary">{ts}</span>
      {line.speakerSlot !== undefined && (
        <SpeakerChip
          slot={line.speakerSlot}
          label={shortTag(line.speakerSlot)}
          compact
        />
      )}
      <span className="flex-1 min-w-0 break-all">{line.text}</span>
      <span className="shrink-0 text-content-placeholder">
        {line.language ?? "?"} · rtf {line.rtf.toFixed(2)}
      </span>
    </>
  );
});
