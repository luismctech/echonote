import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";

import { LogoAnimated } from "../../components/Logo";
import { ResizableHandleVertical } from "../../components/ResizableHandleVertical";
import type { RecordingState } from "../../state/recording";
import type { Note } from "../../types/meeting";
import type { StreamLine } from "../../types/view";
import { NoteInput } from "./NoteInput";
import { NoteList } from "./NoteList";
import { TranscriptRow } from "./TranscriptRow";

const isMac =
  typeof navigator !== "undefined" &&
  /mac|iphone|ipad/i.test(navigator.userAgent);

const SHORTCUT = isMac ? "⌘⇧R" : "Ctrl+Shift+R";
const FOCUS_SHORTCUT = isMac ? "⌘⇧F" : "Ctrl+Shift+F";

/** Format ms as MM:SS for the timer display. */
function formatTimer(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const min = Math.floor(totalSec / 60);
  const sec = totalSec % 60;
  return `${String(min).padStart(2, "0")}:${String(sec).padStart(2, "0")}`;
}

/** Hint text for the empty transcript area. */
function emptyHint(stream: RecordingState, t: ReturnType<typeof useTranslation>["t"]): string {
  if (stream.kind === "recording") return String(t("live.listening"));
  if (stream.kind === "paused") return String(t("live.pausedHint"));
  return String(t("live.pressStart", { shortcut: SHORTCUT }));
}

/* ── Post-stop progress bar — segmented refining stages ── */
const REFINE_STAGES = ["live.refineTranscript", "live.refineSpeakers", "live.refineSummary"] as const;

function RefiningProgress({ stage }: Readonly<{ stage: number }>) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-2 rounded-md border border-subtle bg-surface-elevated px-4 py-3">
      <p className="text-ui-sm font-medium text-content-secondary">
        {t(REFINE_STAGES[Math.min(stage, REFINE_STAGES.length - 1)] ?? REFINE_STAGES[0])}
      </p>
      <div className="flex gap-1">
        {REFINE_STAGES.map((key, i) => {
          let cls = "bg-content-placeholder/20";
          if (i < stage) cls = "bg-emerald-500";
          else if (i === stage) cls = "bg-emerald-400 animate-progress-glow";
          return (
            <div
              key={key}
              className={`h-1.5 flex-1 rounded-full transition-colors duration-500 ${cls} ${i === stage ? "bg-gradient-to-r from-emerald-300 via-emerald-500 to-emerald-300 bg-[length:200%_100%] dark:from-emerald-400 dark:via-emerald-600 dark:to-emerald-400" : ""}`}
            />
          );
        })}
      </div>
    </div>
  );
}

export function LivePane({
  stream,
  stats,
  lines,
  notes,
  listRef,
  canStart,
  canStop,
  canPause,
  canResume,
  diarize,
  onToggleDiarize,
  language,
  onChangeLanguage,
  onStart,
  onStop,
  onPause,
  onResume,
  onAddNote,
  focusMode,
  onToggleFocusMode,
  refineStage,
}: Readonly<{
  stream: RecordingState;
  stats: { chunks: number; skipped: number; audioMs: number };
  lines: StreamLine[];
  notes: Note[];
  listRef: RefObject<HTMLDivElement>;
  canStart: boolean;
  canStop: boolean;
  canPause: boolean;
  canResume: boolean;
  diarize: boolean;
  onToggleDiarize: (next: boolean) => void;
  language: string;
  onChangeLanguage: (next: string) => void;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onAddNote: (text: string) => void;
  focusMode: boolean;
  onToggleFocusMode: () => void;
  /** Current refining stage index (0-2) when stream is stopping/persisted. -1 if not refining. */
  refineStage?: number;
}>) {
  const { t } = useTranslation();

  // Toggle is locked once a session is in flight
  const toggleLocked =
    stream.kind === "starting" ||
    stream.kind === "recording" ||
    stream.kind === "paused" ||
    stream.kind === "stopping";

  const isActive =
    stream.kind === "recording" || stream.kind === "paused";

  // Focus mode: hides the sidebar (history panel)
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((isMac ? e.metaKey : e.ctrlKey) && e.shiftKey && e.key.toLowerCase() === "f") {
        e.preventDefault();
        onToggleFocusMode();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onToggleFocusMode]);

  const virtualizer = useVirtualizer({
    count: lines.length,
    getScrollElement: () => listRef.current,
    estimateSize: () => 28,
    overscan: 5,
  });

  // Track which lines are "new" for streaming fade-in
  const prevLineCountRef = useRef(lines.length);
  useEffect(() => {
    // Update after render so next frame knows the new baseline
    const handle = requestAnimationFrame(() => {
      prevLineCountRef.current = lines.length;
    });
    return () => cancelAnimationFrame(handle);
  }, [lines.length]);

  const splitRef = useRef<HTMLDivElement>(null);
  const [splitRatio, setSplitRatio] = useState(0.5);
  const clampedRatio = Math.min(0.85, Math.max(0.15, splitRatio));

  const handleRatioChange = useCallback((r: number) => {
    setSplitRatio(Math.min(0.85, Math.max(0.15, r)));
  }, []);

  return (
    <>
      {/* ── Top bar: minimal status + controls ── */}
      <header className="flex flex-shrink-0 items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          {/* Status indicator — morphs between states with crossfade */}
          <div className="relative flex h-5 items-center">
            <span
              key={stream.kind}
              className="flex animate-slide-in items-center gap-2 text-ui-sm font-medium"
            >
              {stream.kind === "recording" && (
                <>
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-rose-500 opacity-75" />
                    <span className="relative inline-flex h-2 w-2 rounded-full bg-rose-500" />
                  </span>
                  <span className="text-rose-600 dark:text-rose-400">{t("live.recording")}</span>
                </>
              )}
              {stream.kind === "paused" && (
                <>
                  <span className="inline-flex h-2 w-2 rounded-full bg-amber-500" />
                  <span className="text-amber-600 dark:text-amber-400">{t("live.paused")}</span>
                </>
              )}
              {stream.kind === "starting" && (
                <>
                  <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-emerald-500" />
                  <span className="text-emerald-600 dark:text-emerald-400">{t("live.starting")}</span>
                </>
              )}
              {stream.kind === "stopping" && (
                <>
                  <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-content-placeholder" />
                  <span className="text-content-tertiary">{t("live.stopping")}</span>
                </>
              )}
              {!isActive && stream.kind !== "starting" && stream.kind !== "stopping" && (
                <span className="text-content-tertiary">{t("live.title")}</span>
              )}
            </span>
          </div>

          {/* Timer */}
          {(isActive || stream.kind === "stopping") && (
            <span className="font-mono text-ui-sm tabular-nums text-content-secondary">
              {formatTimer(stats.audioMs)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Language selector */}
          <label
            className={`flex select-none items-center gap-1.5 text-ui-sm ${
              toggleLocked ? "opacity-60" : "cursor-pointer"
            }`}
            title={t("live.langHint")}
          >
            <span className="text-content-tertiary">{t("live.langLabel")}</span>
            <select
              value={language}
              disabled={toggleLocked}
              onChange={(e) => onChangeLanguage(e.target.value)}
              className="rounded border border-subtle bg-surface-elevated px-1.5 py-0.5 text-ui-sm text-content-primary"
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

          {/* Diarize toggle */}
          <label
            className={`flex select-none items-center gap-1.5 text-ui-sm ${
              toggleLocked ? "opacity-60" : "cursor-pointer"
            }`}
          >
            <input
              type="checkbox"
              checked={diarize}
              disabled={toggleLocked}
              onChange={(e) => onToggleDiarize(e.target.checked)}
              className=""
            />
            <span className="text-content-secondary">{t("live.diarize")}</span>
          </label>

          {/* Focus mode toggle — hides sidebar */}
          <button
            type="button"
            onClick={onToggleFocusMode}
            className={`rounded px-1.5 py-0.5 text-ui-xs transition-colors ${
              focusMode
                ? "bg-accent-100 text-accent-700 dark:text-accent-400"
                : "text-content-tertiary hover:text-content-secondary"
            }`}
            title={`Focus mode (${FOCUS_SHORTCUT})`}
            aria-label="Toggle focus mode"
          >
            <svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="inline-block">
              <rect x="1" y="1" width="14" height="14" rx="2" stroke="currentColor" strokeWidth="1.5" />
              {!focusMode && <line x1="4" y1="1" x2="4" y2="15" stroke="currentColor" strokeWidth="1.5" />}
            </svg>
          </button>
        </div>

      {/* ── Post-stop refining progress ── */}
      {stream.kind === "stopping" && refineStage != null && refineStage >= 0 && (
        <RefiningProgress stage={refineStage} />
      )}

      {/* ── Main content area ── */}
      <div ref={splitRef} className="flex min-h-0 flex-1 flex-row">
        {/* ── Notes (left panel — primary) ── */}
        <div
          className="flex min-w-0 flex-col gap-2 overflow-hidden"
          style={{ width: `${clampedRatio * 100}%` }}
        >
          <NoteInput
            elapsedMs={stats.audioMs}
            onSubmit={onAddNote}
            disabled={stream.kind !== "recording" && stream.kind !== "paused"}
          />
          <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3">
            <NoteList notes={notes} />
          </div>
        </div>

        {/* ── Vertical resize handle ── */}
        <ResizableHandleVertical
          containerRef={splitRef}
          ratio={clampedRatio}
          onRatioChange={handleRatioChange}
        />

        {/* ── Transcript (right panel — secondary) ── */}
        <div
          className="flex min-w-0 flex-col overflow-hidden"
          style={{ width: `${(1 - clampedRatio) * 100}%` }}
        >
            <div
              ref={listRef}
              className="min-h-0 flex-1 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3 font-mono text-ui-sm leading-relaxed"
            >
              {lines.length === 0 ? (
                <div className="flex flex-col items-center justify-center gap-3 py-8">
                  <LogoAnimated size={48} className="opacity-40" />
                  <p className="text-ui-sm text-content-placeholder">
                    {emptyHint(stream, t)}
                  </p>
                </div>
              ) : (
                <ul
                  className="relative w-full"
                  style={{ height: `${virtualizer.getTotalSize()}px` }}
                >
                  {virtualizer.getVirtualItems().map((vItem) => {
                    const line = lines[vItem.index]!;
                    const isNew = vItem.index >= prevLineCountRef.current;
                    return (
                      <li
                        key={line.key}
                        className={`absolute left-0 top-0 flex w-full items-baseline gap-3${
                          line.kind === "skipped" ? " text-content-placeholder" : ""
                        }${isNew ? " animate-text-appear opacity-0" : ""}`}
                        style={{ transform: `translateY(${vItem.start}px)` }}
                        data-index={vItem.index}
                        ref={virtualizer.measureElement}
                      >
                        <TranscriptRow line={line} />
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          </div>
      </div>

      {/* ── Bottom bar: controls + audio level ── */}
      <footer className="flex flex-shrink-0 items-center justify-between gap-4 rounded-md border border-subtle bg-surface-sunken px-4 py-2">
        {/* Left: audio level indicator */}
        <div className="flex items-center gap-2">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" className="text-content-tertiary">
            <path d="M8 1.5a2.5 2.5 0 0 0-2.5 2.5v4a2.5 2.5 0 0 0 5 0V4A2.5 2.5 0 0 0 8 1.5Z" stroke="currentColor" strokeWidth="1.2" />
            <path d="M4 7.5a4 4 0 0 0 8 0" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
            <line x1="8" y1="12" x2="8" y2="14.5" stroke="currentColor" strokeWidth="1.2" strokeLinecap="round" />
          </svg>
          {/* Simple level bars */}
          <div className="flex items-end gap-0.5">
            {Array.from({ length: 5 }).map((_, i) => (
              <div
                key={i}
                className={`w-1 rounded-full transition-all duration-150 ${
                  isActive
                    ? i < 3
                      ? "h-2.5 bg-emerald-500"
                      : "h-1.5 bg-emerald-500/30"
                    : "h-1 bg-content-placeholder/30"
                }`}
              />
            ))}
          </div>
          {isActive && (
            <span className="font-mono text-micro tabular-nums text-content-tertiary">
              {stats.chunks} {t("stats.chunks")}
            </span>
          )}
        </div>

        {/* Center: primary action — visceral recording button */}
        <div className="flex items-center gap-2">
          {!isActive && stream.kind !== "starting" && stream.kind !== "stopping" && (
            <button
              type="button"
              onClick={onStart}
              disabled={!canStart}
              className="group relative flex items-center gap-2 rounded-full bg-emerald-600 px-5 py-1.5 text-ui-sm font-medium text-white shadow-sm transition-all hover:bg-emerald-500 hover:shadow-md disabled:cursor-not-allowed disabled:bg-content-placeholder disabled:shadow-none"
            >
              {/* Breathing pulse on idle — the "alive" hint */}
              <span className="absolute inset-0 rounded-full bg-emerald-500 opacity-0 group-enabled:animate-rec-breathe group-disabled:hidden" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-white/80" />
              <span className="relative">{t("live.start")}</span>
            </button>
          )}
          {stream.kind === "starting" && (
            <button
              type="button"
              disabled
              className="relative flex items-center gap-2 rounded-full bg-emerald-600/80 px-5 py-1.5 text-ui-sm font-medium text-white shadow-sm"
            >
              <span className="inline-flex h-2 w-2 animate-pulse rounded-full bg-white/80" />
              {t("live.starting")}
            </button>
          )}
          {canPause && (
            <button
              type="button"
              onClick={onPause}
              className="rounded-full bg-amber-600/10 px-3 py-1.5 text-ui-sm font-medium text-amber-700 ring-1 ring-amber-500/30 transition-colors hover:bg-amber-600/20 dark:text-amber-400"
            >
              {t("live.pause")}
            </button>
          )}
          {canResume && (
            <button
              type="button"
              onClick={onResume}
              className="rounded-full bg-emerald-600/10 px-3 py-1.5 text-ui-sm font-medium text-emerald-700 ring-1 ring-emerald-500/30 transition-colors hover:bg-emerald-600/20 dark:text-emerald-400"
            >
              {t("live.resume")}
            </button>
          )}
          {canStop && (
            <button
              type="button"
              onClick={onStop}
              className="relative flex items-center gap-2 rounded-full bg-rose-600 px-5 py-1.5 text-ui-sm font-medium text-white shadow-sm transition-colors hover:bg-rose-500"
            >
              {/* Active ring radiating while recording */}
              {stream.kind === "recording" && (
                <span className="absolute inset-0 rounded-full border-2 border-rose-400 animate-rec-ring" />
              )}
              <span className="relative inline-flex h-2.5 w-2.5 rounded-sm bg-white/90" />
              <span className="relative">{stream.kind === "stopping" ? t("live.stopping") : t("live.stop")}</span>
            </button>
          )}
        </div>

        {/* Right: timer echo */}
        <div className="flex items-center gap-2">
          {isActive && (
            <span className="font-mono text-ui-xs tabular-nums text-content-tertiary">
              {formatTimer(stats.audioMs)}
            </span>
          )}
        </div>
      </footer>
    </>
  );
}
