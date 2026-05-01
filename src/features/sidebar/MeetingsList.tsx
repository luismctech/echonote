import { memo, useMemo } from "react";
import { useTranslation } from "react-i18next";

import { ContextMenu, type MenuItem } from "../../components/ContextMenu";
import { formatDate, formatDurationMs } from "../../lib/format";
import type { MeetingId, MeetingSummary } from "../../types/meeting";

/** Sidebar list of stored meetings; click to open, right-click for context menu. */
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
      <p className="text-ui-sm text-content-placeholder">
        {t("sidebar.noMeetings")}
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-1">
      {meetings.map((m) => (
        <MeetingItem
          key={m.id}
          meeting={m}
          active={m.id === activeId}
          onSelect={onSelect}
          onDelete={onDelete}
        />
      ))}
    </ul>
  );
});

const MeetingItem = memo(function MeetingItem({
  meeting: m,
  active,
  onSelect,
  onDelete,
}: {
  meeting: MeetingSummary;
  active: boolean;
  onSelect: (m: MeetingSummary) => void;
  onDelete: (m: MeetingSummary) => void;
}) {
  const { t } = useTranslation();

  const contextItems = useMemo<MenuItem[]>(
    () => [
      { label: t("sidebar.open"), onClick: () => onSelect(m) },
      {
        label: t("sidebar.delete"),
        danger: true,
        onClick: () => onDelete(m),
      },
    ],
    [t, m, onSelect, onDelete],
  );

  return (
    <li>
      <ContextMenu items={contextItems}>
        <div
          className={`group flex items-start gap-2 rounded-md border px-2.5 py-2 text-ui-sm ${
            active
              ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"
              : "border-transparent hover:bg-surface-sunken"
          }`}
        >
          <button
            type="button"
            onClick={() => onSelect(m)}
            className="flex flex-1 flex-col items-start gap-0.5 text-left"
          >
            <span className="line-clamp-1 font-medium text-content-primary">
              {m.title}
            </span>
            <span className="text-micro tabular-nums text-content-tertiary">
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
            className="opacity-0 transition-opacity group-hover:opacity-100 text-content-placeholder hover:text-rose-500"
          >
            ×
          </button>
        </div>
      </ContextMenu>
    </li>
  );
});
