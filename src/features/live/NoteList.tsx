import { useTranslation } from "react-i18next";
import type { Note } from "../../types/meeting";

/** Format milliseconds as "MM:SS". */
function formatTimestamp(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

export function NoteList({ notes }: Readonly<{ notes: Note[] }>) {
  const { t } = useTranslation();

  if (notes.length === 0) return null;

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center px-1 py-1">
        <span className="type-section-header text-content-placeholder">
          {t("live.notes")} ({notes.length})
        </span>
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3">
        {notes.map((note) => (
          <li
            key={note.id}
            className="flex items-baseline gap-2 rounded px-1.5 py-1 text-ui-md transition-colors hover:bg-amber-100/60 dark:hover:bg-amber-900/30"
          >
            <span className="shrink-0 font-mono text-ui-xs text-amber-600 dark:text-amber-400">
              {formatTimestamp(note.timestampMs)}
            </span>
            <span className="flex-1 text-content-secondary">{note.text}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
