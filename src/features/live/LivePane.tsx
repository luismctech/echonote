import { type RefObject, useCallback, useEffect, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ChevronDown, Globe, PanelLeft, Mic, MonitorSpeaker, Pause, Play, Users, X } from "lucide-react";

import { LogoAnimated } from "../../components/Logo";
import { ResizableHandleVertical } from "../../components/ResizableHandleVertical";
import type { RecordingState } from "../../state/recording";
import type { AudioSourceKind } from "../../types/streaming";
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
  audioSource,
  onChangeSource,
  micActive,
  sysActive,
  onToggleMic,
  onToggleSys,
  externalDeviceName,
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
  audioSource: AudioSourceKind;
  onChangeSource: (next: AudioSourceKind) => void;
  /** Active mic flag for Mixed mode (undefined when not Mixed). */
  micActive?: boolean;
  /** Active sys flag for Mixed mode (undefined when not Mixed). */
  sysActive?: boolean;
  onToggleMic?: (active: boolean) => void;
  onToggleSys?: (active: boolean) => void;
  /** Device name for the external-output banner (null = hide banner). */
  externalDeviceName: string | null;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onAddNote: (text: string) => Promise<boolean>;
  focusMode: boolean;
  onToggleFocusMode: () => void;
  /** Current refining stage index (0-2) when stream is stopping/persisted. -1 if not refining. */
  refineStage?: number;
}>) {
  const { t } = useTranslation();

  const [bannerDismissed, setBannerDismissed] = useState(false);
  const showExternalBanner =
    externalDeviceName !== null && !bannerDismissed && audioSource !== "mixed";

  // Toggle is locked once a session is in flight
  const toggleLocked =
    stream.kind === "starting" ||
    stream.kind === "recording" ||
    stream.kind === "paused" ||
    stream.kind === "stopping";

  const isActive =
    stream.kind === "recording" || stream.kind === "paused";

  const sourceLabels: Record<AudioSourceKind, string> = {
    microphone: t("live.sourceMic"),
    systemOutput: t("live.sourceSystem"),
    mixed: t("live.sourceMixed"),
  };

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

        <div className="flex items-center gap-2">
          {/* Unified capture controls — language · voices · source */}
          <div
            className={`flex items-center divide-x divide-subtle rounded-lg border border-subtle bg-surface-elevated text-ui-sm ${
              toggleLocked ? "opacity-60" : ""
            }`}
          >
            {/* Language */}
            <label
              className={`relative flex select-none items-center gap-1.5 px-2.5 py-1 ${
                toggleLocked ? "" : "cursor-pointer"
              }`}
              title={t("live.langHint")}
            >
              <Globe className="h-3.5 w-3.5 shrink-0 text-content-tertiary" aria-hidden />
              <span className="relative inline-flex items-center pr-4 text-content-primary">
                <span aria-hidden>
                  {language === "" ? t("live.langAuto") : t(`languages.${language}`)}
                </span>
                <select
                  value={language}
                  disabled={toggleLocked}
                  onChange={(e) => onChangeLanguage(e.target.value)}
                  className="absolute inset-0 w-full cursor-pointer appearance-none border-0 bg-transparent p-0 text-ui-sm text-transparent opacity-0 focus:opacity-0 focus:outline-none focus:ring-0"
                  aria-label={t("live.langTooltip")}
                >
                  <option value="">{t("live.langAuto")}</option>
                  <option value="es">{t("languages.es")}</option>
                  <option value="en">{t("languages.en")}</option>
                  <option value="pt">{t("languages.pt")}</option>
                  <option value="fr">{t("languages.fr")}</option>
                  <option value="de">{t("languages.de")}</option>
                  <option value="it">{t("languages.it")}</option>
                </select>
                <ChevronDown
                  className="pointer-events-none absolute right-0 h-3 w-3 text-content-tertiary"
                  aria-hidden
                />
              </span>
            </label>

            {/* Voices (formerly Diarize) */}
            <button
              type="button"
              onClick={() => !toggleLocked && onToggleDiarize(!diarize)}
              disabled={toggleLocked}
              role="switch"
              aria-checked={diarize}
              title={t("live.diarizeHint")}
              className={`flex select-none items-center gap-1.5 px-2.5 py-1 transition-colors ${
                toggleLocked ? "" : "cursor-pointer hover:bg-surface-sunken"
              } ${
                diarize
                  ? "text-emerald-700 dark:text-emerald-400"
                  : "text-content-secondary"
              }`}
            >
              <Users
                className={`h-3.5 w-3.5 ${diarize ? "text-emerald-600 dark:text-emerald-400" : "text-content-tertiary"}`}
                aria-hidden
              />
              <span>{t("live.diarize")}</span>
              <span
                className={`ml-0.5 inline-block h-2 w-2 rounded-full ${
                  diarize
                    ? "bg-emerald-500 shadow-[0_0_0_2px_rgba(16,185,129,0.18)]"
                    : "border border-content-placeholder/50"
                }`}
                aria-hidden
              />
            </button>

            {/* Audio source */}
            <label
              className={`relative flex select-none items-center gap-1.5 px-2.5 py-1 ${
                toggleLocked ? "" : "cursor-pointer"
              }`}
              title={t("live.sourceHint")}
            >
              <Mic className="h-3.5 w-3.5 shrink-0 text-content-tertiary" aria-hidden />
              <span className="relative inline-flex items-center pr-4 text-content-primary">
                <span aria-hidden>{sourceLabels[audioSource]}</span>
                <select
                  value={audioSource}
                  disabled={toggleLocked}
                  onChange={(e) => onChangeSource(e.target.value as AudioSourceKind)}
                  className="absolute inset-0 w-full cursor-pointer appearance-none border-0 bg-transparent p-0 text-ui-sm text-transparent opacity-0 focus:opacity-0 focus:outline-none focus:ring-0"
                  aria-label={t("live.sourceLabel")}
                >
                  <option value="microphone">{t("live.sourceMic")}</option>
                  <option value="systemOutput">{t("live.sourceSystem")}</option>
                  <option value="mixed">{t("live.sourceMixed")}</option>
                </select>
                <ChevronDown
                  className="pointer-events-none absolute right-0 h-3 w-3 text-content-tertiary"
                  aria-hidden
                />
              </span>
            </label>
          </div>

          {/* Focus mode toggle — hides sidebar */}
          <button
            type="button"
            onClick={onToggleFocusMode}
            className={`rounded-lg border px-2 py-1 text-ui-xs transition-colors ${
              focusMode
                ? "border-accent-400 bg-accent-100 text-accent-700 dark:border-accent-700 dark:bg-accent-900/30 dark:text-accent-400"
                : "border-subtle text-content-tertiary hover:text-content-secondary"
            }`}
            title={t("live.focusMode", { shortcut: FOCUS_SHORTCUT })}
            aria-label={t("live.toggleFocusMode")}
          >
            <PanelLeft className="h-3.5 w-3.5 inline-block" />
          </button>
        </div>
      </header>

      {/* ── External output device banner ── */}
      {showExternalBanner && (
        <div className="flex items-center justify-between gap-3 rounded-md border border-blue-500/30 bg-blue-50 px-3 py-2 text-ui-sm dark:bg-blue-950/30">
          <span className="text-blue-700 dark:text-blue-300">
            {t("live.externalDeviceDetected", { device: externalDeviceName })}{" "}
            <button
              type="button"
              onClick={() => { onChangeSource("mixed"); setBannerDismissed(true); }}
              className="ml-1 font-semibold underline underline-offset-2 hover:no-underline"
            >
              {t("live.enableMixed")}
            </button>
          </span>
          <button
            type="button"
            onClick={() => setBannerDismissed(true)}
            aria-label={t("live.dismissBanner")}
            className="flex-shrink-0 rounded p-0.5 text-blue-500 hover:bg-blue-100 dark:hover:bg-blue-900"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}

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
        {/* Left: audio level indicator + optional mix source toggles */}
        <div className="flex items-center gap-2">
          {audioSource === "systemOutput" ? (
            <MonitorSpeaker className="h-3.5 w-3.5 text-content-tertiary" />
          ) : (
            <Mic className="h-3.5 w-3.5 text-content-tertiary" />
          )}
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
          {/* Mix source toggles — visible only during an active Mixed session */}
          {audioSource === "mixed" && isActive && (
            <div className="flex items-center gap-1 border-l border-subtle pl-2">
              <button
                type="button"
                title={t("live.toggleMic")}
                onClick={() => onToggleMic?.(!micActive)}
                className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-micro ring-1 transition-colors ${
                  micActive
                    ? "bg-emerald-500/10 text-emerald-700 ring-emerald-500/30 dark:text-emerald-400"
                    : "bg-content-placeholder/10 text-content-placeholder ring-content-placeholder/20 line-through"
                }`}
              >
                <Mic className="h-2.5 w-2.5" />
                {t("live.mic")}
              </button>
              <button
                type="button"
                title={t("live.toggleSys")}
                onClick={() => onToggleSys?.(!sysActive)}
                className={`flex items-center gap-1 rounded-full px-2 py-0.5 text-micro ring-1 transition-colors ${
                  sysActive
                    ? "bg-violet-500/10 text-violet-700 ring-violet-500/30 dark:text-violet-400"
                    : "bg-content-placeholder/10 text-content-placeholder ring-content-placeholder/20 line-through"
                }`}
              >
                <MonitorSpeaker className="h-2.5 w-2.5" />
                {t("live.sys")}
              </button>
            </div>
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
              className="relative z-10 flex items-center gap-1.5 rounded-full bg-amber-600/10 px-3 py-1.5 text-ui-sm font-medium text-amber-700 ring-1 ring-amber-500/30 transition-colors hover:bg-amber-600/20 dark:text-amber-400"
            >
              <Pause className="h-3.5 w-3.5" />
              {t("live.pause")}
            </button>
          )}
          {canResume && (
            <button
              type="button"
              onClick={onResume}
              className="relative z-10 flex items-center gap-1.5 rounded-full bg-emerald-600 px-5 py-1.5 text-ui-sm font-medium text-white shadow-sm transition-colors hover:bg-emerald-500"
            >
              <Play className="h-3.5 w-3.5 fill-white" />
              {t("live.resume")}
            </button>
          )}
          {canStop && (
            <button
              type="button"
              onClick={onStop}
              className="relative flex items-center gap-2 rounded-full bg-rose-600 px-5 py-1.5 text-ui-sm font-medium text-white shadow-sm transition-colors hover:bg-rose-500"
            >
              {/* Active ring radiating while recording — pointer-events-none so it never intercepts clicks on adjacent buttons */}
              {stream.kind === "recording" && (
                <span className="pointer-events-none absolute inset-0 rounded-full border-2 border-rose-400 animate-rec-ring" />
              )}
              <span className="relative inline-flex h-2.5 w-2.5 rounded-sm bg-white/90" />
              <span className="relative">{stream.kind === "stopping" ? t("live.stopping") : t("live.stop")}</span>
            </button>
          )}
        </div>

        {/* Right: timer + paused badge */}
        <div className="flex items-center gap-2">
          {canResume && (
            <span className="flex items-center gap-1 rounded-full bg-amber-500/15 px-2.5 py-1 text-micro font-semibold text-amber-600 ring-1 ring-amber-500/30 animate-pulse dark:text-amber-400">
              <Pause className="h-2.5 w-2.5 fill-current" />
              {t("live.paused")}
            </span>
          )}
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
