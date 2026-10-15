import { memo, useEffect, useRef, useState, type ReactNode } from "react";
import { useTranslation } from "react-i18next";
import { Pencil } from "lucide-react";

import { displayName, paletteFor, shortTag } from "../../lib/speakers";
import type { Speaker, SpeakerId } from "../../types/speaker";

/**
 * One speaker chip in the meeting speakers panel.
 *
 * Clicking the chip toggles the speaker in the transcript filter
 * (`aria-pressed` reflects it). Renaming lives on the pencil button:
 * it swaps the chip for an inline input that commits on Enter or blur
 * and cancels on Escape. The canonical label still flows from props
 * (the post-rename meeting), so an external refresh overrides stale
 * local input. `actions` renders extra controls (the speaker menu)
 * after the pencil.
 */
export const SpeakerEditor = memo(function SpeakerEditor({
  speaker,
  talkTimePct,
  selected,
  dimmed,
  onToggle,
  onRename,
  actions,
}: {
  speaker: Speaker;
  /** Percentage of total talk-time for this speaker (0–100). */
  talkTimePct?: number;
  /** The speaker is part of the active transcript filter. */
  selected: boolean;
  /** Another speaker is filtered and this one is not: render faded. */
  dimmed: boolean;
  onToggle: (speakerId: SpeakerId) => void;
  onRename: (speakerId: SpeakerId, label: string | null) => Promise<void>;
  actions?: ReactNode;
}) {
  const { t } = useTranslation();
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(speaker.label ?? "");
  const inputRef = useRef<HTMLInputElement>(null);
  // Re-sync local draft whenever the upstream speaker label changes
  // (e.g. after renameSpeaker resolves with the canonical row), so
  // the input does not show stale text after a successful save.
  useEffect(() => {
    setDraft(speaker.label ?? "");
  }, [speaker.label]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const palette = paletteFor(speaker.slot);
  const name = displayName(speaker, t);
  const placeholder = t("speakers.speaker", { n: speaker.slot + 1 });
  const label = speaker.label?.trim();
  const commit = () => {
    setEditing(false);
    const next = draft.trim();
    const current = speaker.label ?? "";
    if (next === current) return;
    void onRename(speaker.id, next.length > 0 ? next : null);
  };

  const ring = selected
    ? "ring-2 ring-content-primary/60"
    : `ring-1 ring-inset ${palette.ring}`;
  const fade = dimmed ? "opacity-50 hover:opacity-80" : "";

  return (
    <div
      className={`flex items-center gap-0.5 rounded-full pr-1 text-ui-sm transition-opacity ${palette.bg} ${palette.text} ${ring} ${fade}`}
    >
      {editing ? (
        <span className="flex items-center gap-1.5 py-0.5 pl-2">
          <span className="font-semibold tabular-nums">
            {shortTag(speaker.slot)}
          </span>
          <input
            ref={inputRef}
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
                setEditing(false);
              }
            }}
            aria-label={t("speakers.rename", { name })}
            className="w-28 bg-transparent outline-hidden placeholder:text-current placeholder:opacity-60"
          />
        </span>
      ) : (
        <>
          <button
            type="button"
            aria-pressed={selected}
            onClick={() => onToggle(speaker.id)}
            title={t("speakers.filterBy", { name })}
            className="flex items-center gap-1.5 rounded-full py-0.5 pl-2 pr-1 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-accent-400"
          >
            <span className="font-semibold tabular-nums">
              {shortTag(speaker.slot)}
            </span>
            {label ? (
              <span className="max-w-32 truncate">{label}</span>
            ) : (
              <span className="opacity-60">{placeholder}</span>
            )}
            {talkTimePct != null && talkTimePct > 0 && (
              <span className="ml-0.5 tabular-nums opacity-70">
                {talkTimePct}%
              </span>
            )}
          </button>
          <button
            type="button"
            onClick={() => setEditing(true)}
            aria-label={t("speakers.rename", { name })}
            title={t("speakers.rename", { name })}
            className="rounded-full p-1 opacity-60 hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-accent-400"
          >
            <Pencil className="h-3 w-3" />
          </button>
          {actions}
        </>
      )}
    </div>
  );
});
