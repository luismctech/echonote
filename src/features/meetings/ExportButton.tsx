import { useCallback, useState } from "react";
import { useTranslation } from "react-i18next";
import { save } from "@tauri-apps/plugin-dialog";

import { exportMeeting, type ExportFormat } from "../../ipc/client";
import type { MeetingId } from "../../types/meeting";

const FORMATS: ReadonlyArray<{ id: ExportFormat; labelKey: string; ext: string }> =
  [
    { id: "markdown", labelKey: "export.markdown", ext: "md" },
    { id: "txt", labelKey: "export.text", ext: "txt" },
  ];

export function ExportButton({
  meetingId,
  title,
}: Readonly<{ meetingId: MeetingId; title: string }>) {
  const { t } = useTranslation();
  const [exporting, setExporting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleExport = useCallback(
    async (format: ExportFormat, ext: string) => {
      setError(null);
      const safeName = title.replaceAll(/[^a-zA-Z0-9_-]/g, "_");
      const path = await save({
        title: t("export.title"),
        defaultPath: `${safeName}.${ext}`,
        filters: [{ name: ext.toUpperCase(), extensions: [ext] }],
      });
      if (!path) return;
      setExporting(true);
      try {
        await exportMeeting(meetingId, format, path);
      } catch (e) {
        setError(String(e));
      } finally {
        setExporting(false);
      }
    },
    [meetingId, title],
  );

  return (
    <div className="relative inline-block">
      <details className="group">
        <summary
          className={`cursor-pointer list-none rounded-md border border-zinc-200 bg-white px-3 py-1.5 text-xs font-medium hover:bg-zinc-100 dark:border-zinc-700 dark:bg-zinc-900 dark:hover:bg-zinc-800 ${
            exporting ? "pointer-events-none opacity-60" : ""
          }`}
        >
          {exporting ? t("export.exporting") : t("export.button")}
        </summary>
        <div className="absolute right-0 z-10 mt-1 w-44 rounded-md border border-zinc-200 bg-white py-1 shadow-lg dark:border-zinc-700 dark:bg-zinc-900">
          {FORMATS.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => handleExport(f.id, f.ext)}
              className="block w-full px-3 py-1.5 text-left text-xs hover:bg-zinc-100 dark:hover:bg-zinc-800"
            >
              {t(f.labelKey)}
            </button>
          ))}
        </div>
      </details>
      {error && (
        <p className="mt-1 text-[10px] text-amber-600 dark:text-amber-400">
          {error}
        </p>
      )}
    </div>
  );
}
