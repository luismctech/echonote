import { memo, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { displayName, paletteFor, shortTag } from "../../lib/speakers";
import type { Speaker, SpeakerId } from "../../types/speaker";

/**
 * Inline rename input for a single speaker.
 *
 * Local input state lets the user keep typing without every keystroke
 * triggering an IPC round-trip. We commit on Enter or blur; the
 * canonical state still flows from props (the post-rename meeting)
 * so an external refresh would override stale local input.
 */
export const SpeakerEditor = memo(function SpeakerEditor({
  speaker,
  talkTimePct,
  onRename,
}: {
  speaker: Speaker;
  /** Percentage of total talk-time for this speaker (0–100). */
  talkTimePct?: number;
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
}) {
  const { t } = useTranslation();
  const [draft, setDraft] = useState(speaker.label ?? "");
  // Re-sync local draft whenever the upstream speaker label changes
  // (e.g. after renameSpeaker resolves with the canonical row), so
  // the input does not show stale text after a successful save.
  useEffect(() => {
    setDraft(speaker.label ?? "");
  }, [speaker.label]);

  const palette = paletteFor(speaker.slot);
  const placeholder = t("speakers.speaker", { n: speaker.slot + 1 });
  const commit = () => {
    const next = draft.trim();
    const current = speaker.label ?? "";
    if (next === current) return;
    void onRename(speaker.id, next.length > 0 ? next : null);
  };
  return (
    <div
      className={`flex items-center gap-1.5 rounded-full px-2 py-0.5 text-ui-sm ring-1 ring-inset ${palette.bg} ${palette.text} ${palette.ring}`}
    >
      <span className="font-semibold tabular-nums">{shortTag(speaker.slot)}</span>
      <input
        type="text"
        value={draft}
        placeholder={placeholder}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") {
            e.currentTarget.blur();
          } else if (e.key === "Escape") {
            setDraft(speaker.label ?? "");
            e.currentTarget.blur();
          }
        }}
        aria-label={t("speakers.rename", { name: displayName(speaker) })}
        className="w-28 bg-transparent outline-none placeholder:text-current placeholder:opacity-60"
      />
      {talkTimePct != null && talkTimePct > 0 && (
        <span className="ml-0.5 tabular-nums opacity-70">{talkTimePct}%</span>
      )}
    </div>
  );
});
