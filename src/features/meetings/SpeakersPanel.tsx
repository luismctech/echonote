import { useMemo, type ReactNode } from "react";
import { useTranslation } from "react-i18next";

import type { Segment } from "../../types/meeting";
import type { Speaker, SpeakerId } from "../../types/speaker";
import { SpeakerEditor } from "./SpeakerEditor";

/** Compute talk-time ms per speaker from segments. */
function computeTalkTime(segments: readonly Segment[]): Map<string, number> {
  const map = new Map<string, number>();
  for (const seg of segments) {
    if (!seg.speakerId) continue;
    const dur = seg.endMs - seg.startMs;
    map.set(seg.speakerId, (map.get(seg.speakerId) ?? 0) + dur);
  }
  return map;
}

/**
 * Diarized speakers of a meeting. Clicking a chip toggles it in the
 * transcript filter (several may be selected); each chip keeps a
 * rename action and its share of the talk time. `renderActions` adds
 * per-speaker controls such as the merge menu.
 */
export function SpeakersPanel({
  speakers,
  segments,
  filter,
  onToggleFilter,
  onRename,
  renderActions,
}: {
  speakers: Speaker[];
  segments: readonly Segment[];
  /** Speaker ids currently filtering the transcript (empty = all). */
  filter: ReadonlySet<SpeakerId>;
  onToggleFilter: (speakerId: SpeakerId) => void;
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
  renderActions?: (speaker: Speaker) => ReactNode;
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
      className="flex flex-wrap gap-2 rounded-md border border-subtle bg-surface-sunken p-2"
    >
      {speakers.map((sp) => {
        const ms = talkTime.get(sp.id) ?? 0;
        const pct = totalMs > 0 ? Math.round((ms / totalMs) * 100) : 0;
        const selected = filter.has(sp.id);
        return (
          <SpeakerEditor
            key={sp.id}
            speaker={sp}
            talkTimePct={pct}
            selected={selected}
            dimmed={filter.size > 0 && !selected}
            onToggle={onToggleFilter}
            onRename={onRename}
            actions={renderActions?.(sp)}
          />
        );
      })}
    </section>
  );
}
