import type { RefObject } from "react";
import { useTranslation } from "react-i18next";

import { StatsBar } from "../../components/StatsBar";
import { statusLabel, type RecordingState } from "../../state/recording";
import type { StreamLine } from "../../types/view";
import { TranscriptRow } from "./TranscriptRow";

/** Subtitle text under the live pane title — model + input format. */
function modelLabel(stream: RecordingState): string {
  if (stream.kind === "recording" && stream.inputFormat) {
    const { sampleRateHz, channels } = stream.inputFormat;
    return `${sampleRateHz} Hz · ${channels} ch`;
  }
  return "";
}

export function LivePane({
  stream,
  stats,
  lines,
  listRef,
  canStart,
  canStop,
  diarize,
  onToggleDiarize,
  language,
  onChangeLanguage,
  onStart,
  onStop,
  onDismissError,
}: {
  stream: RecordingState;
  stats: { chunks: number; skipped: number; audioMs: number };
  lines: StreamLine[];
  listRef: RefObject<HTMLDivElement>;
  canStart: boolean;
  canStop: boolean;
  diarize: boolean;
  onToggleDiarize: (next: boolean) => void;
  language: string;
  onChangeLanguage: (next: string) => void;
  onStart: () => void;
  onStop: () => void;
  onDismissError: () => void;
}) {
  // Toggle is locked once a session is in flight: changing the
  // diarize flag (or language hint) mid-recording would mix
  // half-and-half results and is confusing to render.
  const { t } = useTranslation();
  const toggleLocked =
    stream.kind === "starting" ||
    stream.kind === "recording" ||
    stream.kind === "stopping";
  return (
    <>
      <header className="flex flex-shrink-0 flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-base font-medium sm:text-lg">{t("live.title")}</h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {t("live.description", { label: modelLabel(stream) || t("live.modelLoads") })}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <label
            className={`flex select-none items-center gap-1.5 text-xs ${
              toggleLocked ? "opacity-60" : "cursor-pointer"
            }`}
            title={t("live.langHint")}
          >
            <span className="text-zinc-500 dark:text-zinc-400">{t("live.langLabel")}</span>
            <select
              value={language}
              disabled={toggleLocked}
              onChange={(e) => onChangeLanguage(e.target.value)}
              className="rounded border border-zinc-200 bg-white px-1.5 py-0.5 text-xs dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200"
              aria-label={t("live.langTooltip")}
            >
              <option value="">{t("live.langAuto")}</option>
              <option value="es">es</option>
              <option value="en">en</option>
              <option value="pt">pt</option>
              <option value="fr">fr</option>
              <option value="de">de</option>
              <option value="it">it</option>
            </select>
          </label>
          <label
            className={`flex select-none items-center gap-2 text-xs ${
              toggleLocked ? "opacity-60" : "cursor-pointer"
            }`}
          >
            <input
              type="checkbox"
              checked={diarize}
              disabled={toggleLocked}
              onChange={(e) => onToggleDiarize(e.target.checked)}
              className="h-3.5 w-3.5 accent-emerald-600"
            />
            <span className="text-zinc-600 dark:text-zinc-300">{t("live.diarize")}</span>
          </label>
          <button
            type="button"
            onClick={onStart}
            disabled={!canStart}
            className="rounded-md bg-emerald-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-emerald-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
          >
            {stream.kind === "starting" ? t("live.starting") : t("live.start")}
          </button>
          <button
            type="button"
            onClick={onStop}
            disabled={!canStop}
            className="rounded-md bg-rose-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-rose-500 disabled:cursor-not-allowed disabled:bg-zinc-300 dark:disabled:bg-zinc-700"
          >
            {stream.kind === "stopping" ? t("live.stopping") : t("live.stop")}
          </button>
        </div>
      </header>

      {stream.kind === "error" && (
        <div className="flex items-start justify-between gap-3 rounded-md bg-rose-50 px-3 py-2 text-xs text-rose-900 dark:bg-rose-950/40 dark:text-rose-200">
          <p>
            <strong className="font-semibold">
              {stream.recoverable ? t("live.error") : t("live.fatal")}
            </strong>{" "}
            {stream.message}
          </p>
          <button
            type="button"
            onClick={onDismissError}
            className="text-xs underline opacity-80 hover:opacity-100"
          >
            {t("live.dismiss")}
          </button>
        </div>
      )}

      <StatsBar status={statusLabel(stream)} stats={stats} />

      <div
        ref={listRef}
        className="min-h-0 flex-1 overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 font-mono text-xs leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60"
      >
        {lines.length === 0 ? (
          <p className="text-zinc-400">
            {stream.kind === "recording"
              ? t("live.listening")
              : t("live.pressStart")}
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {lines.map((line) => (
              <TranscriptRow key={line.key} line={line} />
            ))}
          </ul>
        )}
      </div>
    </>
  );
}
