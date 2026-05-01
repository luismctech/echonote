import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import type { Note, NoteId } from "../../types/meeting";
import { formatTimestamp } from "../../lib/format";
import { deleteNote } from "../../ipc/client";
import { CopyButton } from "../../components/CopyButton";

export function NotesPanel({
  notes,
  onDeleted,
}: Readonly<{
  notes: Note[];
  /** Callback after a note is deleted so parent can refresh. */
  onDeleted?: (noteId: NoteId) => void;
}>) {
  const { t } = useTranslation();
  const [deletingId, setDeletingId] = useState<NoteId | null>(null);

  const getNotesText = useCallback(() => {
    return notes
      .map((n) => `[${formatTimestamp(n.timestampMs)}] ${n.text}`)
      .join("\n");
  }, [notes]);

  if (notes.length === 0) return null;

  const handleDelete = async (noteId: NoteId) => {
    setDeletingId(noteId);
    try {
      await deleteNote(noteId);
      onDeleted?.(noteId);
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex items-center justify-between px-1 py-1">
        <span className="text-ui-xs font-medium uppercase tracking-wide text-content-placeholder">
          {t("meeting.notes")}
        </span>
        <CopyButton getText={getNotesText} title={t("meeting.copyNotes")} />
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3">
        {notes.map((note) => (
          <li
            key={note.id}
            className="group flex items-baseline gap-2 rounded px-1.5 py-1 text-ui-md transition-colors hover:bg-amber-100/60 dark:hover:bg-amber-900/30"
          >
            <span className="shrink-0 font-mono text-ui-xs text-amber-600 dark:text-amber-400">
              {formatTimestamp(note.timestampMs)}
            </span>
            <span className="flex-1 text-content-secondary">
              {note.text}
            </span>
            <button
              type="button"
              onClick={() => handleDelete(note.id)}
              disabled={deletingId === note.id}
              className="shrink-0 text-micro text-content-placeholder opacity-0 transition-opacity hover:text-rose-500 group-hover:opacity-100 disabled:opacity-50"
              title={t("meeting.deleteNote")}
            >
              ✕
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}
