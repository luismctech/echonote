import { memo } from "react";
import { useTranslation } from "react-i18next";

import { formatDate, formatDurationMs } from "../../lib/format";
import type { MeetingId, MeetingSummary } from "../../types/meeting";

/** Sidebar list of stored meetings; click to open, hover to delete. */
export const MeetingsList = memo(function MeetingsList({
  meetings,
  activeId,
  onSelect,
  onDelete,
}: {
  meetings: MeetingSummary[];
  activeId: MeetingId | null;
  onSelect: (m: MeetingSummary) => void;
  onDelete: (m: MeetingSummary) => void;
}) {
  const { t } = useTranslation();
  if (meetings.length === 0) {
    return (
      <p className="text-xs text-zinc-400">
        {t("sidebar.noMeetings")}
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-1">
      {meetings.map((m) => {
        const active = m.id === activeId;
        return (
          <li key={m.id}>
            <div
              className={`group flex items-start gap-2 rounded-md border px-2.5 py-2 text-xs ${
                active
                  ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"
                  : "border-transparent hover:bg-zinc-50 dark:hover:bg-zinc-900"
              }`}
            >
              <button
                type="button"
                onClick={() => onSelect(m)}
                className="flex flex-1 flex-col items-start gap-0.5 text-left"
              >
                <span className="line-clamp-1 font-medium text-zinc-800 dark:text-zinc-100">
                  {m.title}
                </span>
                <span className="text-[10px] tabular-nums text-zinc-500 dark:text-zinc-400">
                  {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)}
                </span>
              </button>
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(m);
                }}
                aria-label={t("sidebar.deleteConfirm", { title: m.title })}
                className="opacity-0 transition-opacity group-hover:opacity-100 text-zinc-400 hover:text-rose-500"
              >
                ×
              </button>
            </div>
          </li>
        );
      })}
    </ul>
  );
});
