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
          className={`cursor-pointer list-none rounded-md border bg-surface-elevated px-3 py-1.5 text-ui-sm font-medium hover:bg-surface-sunken ${
            exporting ? "pointer-events-none opacity-60" : ""
          }`}
        >
          {exporting ? t("export.exporting") : t("export.button")}
        </summary>
        <div className="absolute right-0 z-10 mt-1 w-44 rounded-md border bg-surface-elevated py-1 shadow-lg">
          {FORMATS.map((f) => (
            <button
              key={f.id}
              type="button"
              onClick={() => handleExport(f.id, f.ext)}
              className="block w-full px-3 py-1.5 text-left text-ui-sm hover:bg-surface-sunken"
            >
              {t(f.labelKey)}
            </button>
          ))}
        </div>
      </details>
      {error && (
        <p className="mt-1 text-micro text-amber-600 dark:text-amber-400">
          {error}
        </p>
      )}
    </div>
  );
}
