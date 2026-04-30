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
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
          {t("meeting.notes")} ({notes.length})
        </span>
        <CopyButton getText={getNotesText} title={t("meeting.copyNotes")} />
      </div>
      <ul className="flex min-h-0 flex-1 flex-col gap-1.5 overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-900 dark:bg-zinc-900">
        {notes.map((note) => (
          <li
            key={note.id}
            className="group flex items-baseline gap-2 rounded px-1.5 py-1 text-sm transition-colors hover:bg-amber-100/60 dark:hover:bg-amber-900/30"
          >
            <span className="shrink-0 font-mono text-[11px] text-amber-600 dark:text-amber-400">
              {formatTimestamp(note.timestampMs)}
            </span>
            <span className="flex-1 text-zinc-700 dark:text-zinc-300">
              {note.text}
            </span>
            <button
              type="button"
              onClick={() => handleDelete(note.id)}
              disabled={deletingId === note.id}
              className="shrink-0 text-[10px] text-zinc-400 opacity-0 transition-opacity hover:text-rose-500 group-hover:opacity-100 disabled:opacity-50"
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
