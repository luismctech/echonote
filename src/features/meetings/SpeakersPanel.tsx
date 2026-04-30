import { useMemo } from "react";
import { useTranslation } from "react-i18next";

import type { Segment } from "../../types/meeting";
import type { Speaker, SpeakerId } from "../../types/speaker";
import { SpeakerEditor } from "./SpeakerEditor";

/** Compute talk-time ms per speaker from segments. */
function computeTalkTime(segments: Segment[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const seg of segments) {
    if (!seg.speakerId) continue;
    const dur = seg.endMs - seg.startMs;
    map.set(seg.speakerId, (map.get(seg.speakerId) ?? 0) + dur);
  }
  return map;
}

/**
 * List of diarized speakers with an inline rename input each. Saves
 * on Enter or blur; clearing the input reverts to anonymous.
 */
export function SpeakersPanel({
  speakers,
  segments,
  onRename,
}: {
  speakers: Speaker[];
  segments: Segment[];
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  const { t } = useTranslation();

  const talkTime = useMemo(() => computeTalkTime(segments), [segments]);
  const totalMs = useMemo(
    () => Array.from(talkTime.values()).reduce((a, b) => a + b, 0),
    [talkTime],
  );

  return (
    <section
      aria-label={t("speakers.label")}
      className="flex flex-wrap gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-2 dark:border-zinc-900 dark:bg-zinc-900/40"
    >
      {speakers.map((sp) => {
        const ms = talkTime.get(sp.id) ?? 0;
        const pct = totalMs > 0 ? Math.round((ms / totalMs) * 100) : 0;
        return (
          <SpeakerEditor
            key={sp.id}
            speaker={sp}
            talkTimePct={pct}
            onRename={onRename}
          />
        );
      })}
    </section>
  );
}
