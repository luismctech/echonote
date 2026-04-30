import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";
import { ResizableHandle } from "../../components/ResizableHandle";
import { ResizableHandleVertical } from "../../components/ResizableHandleVertical";
import { CopyButton } from "../../components/CopyButton";

import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { LogoAnimated } from "../../components/Logo";
import { formatDate, formatDurationMs, formatTimestamp } from "../../lib/format";
import { displayName, indexSpeakers } from "../../lib/speakers";
import type { Note, NoteId } from "../../types/meeting";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ExportButton } from "./ExportButton";
import { NotesPanel } from "./NotesPanel";
import { SegmentRow } from "./SegmentRow";
import { SpeakersPanel } from "./SpeakersPanel";
import { SummaryPanel } from "./SummaryPanel";

/** Inline-editable meeting title. Click to edit, Enter/blur to save. */
function EditableTitle({
  value,
  onCommit,
}: Readonly<{
  value: string;
  onCommit: (title: string) => Promise<void>;
}>) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const inputRef = useRef<HTMLInputElement>(null);
  const { t } = useTranslation();

  // Sync draft when external value changes (e.g. after backend rename).
  useEffect(() => {
    if (!editing) setDraft(value);
  }, [value, editing]);

  useEffect(() => {
    if (editing) inputRef.current?.select();
  }, [editing]);

  const commit = useCallback(() => {
    const trimmed = draft.trim();
    setEditing(false);
    if (trimmed.length === 0 || trimmed === value) {
      setDraft(value);
      return;
    }
    void onCommit(trimmed);
  }, [draft, value, onCommit]);

  if (!editing) {
    return (
      <button
        type="button"
        className="group flex items-center gap-1.5 text-left text-base font-medium sm:text-lg"
        onClick={() => setEditing(true)}
        title={t("meeting.editTitle")}
      >
        <span>{value}</span>
        <svg
          className="h-3.5 w-3.5 shrink-0 text-zinc-400 opacity-0 transition-opacity group-hover:opacity-100"
          viewBox="0 0 16 16"
          fill="currentColor"
        >
          <path d="M12.146.854a.5.5 0 0 1 .708 0l2.292 2.292a.5.5 0 0 1 0 .708l-9.5 9.5a.5.5 0 0 1-.168.11l-4 1.5a.5.5 0 0 1-.632-.632l1.5-4a.5.5 0 0 1 .11-.168l9.5-9.5zM11.207 2.5 13.5 4.793 14.793 3.5 12.5 1.207 11.207 2.5zm1.586 3L10.5 3.207 3 10.707V11h.5a.5.5 0 0 1 .5.5v.5h.5a.5.5 0 0 1 .5.5v.5h.293l7.5-7.5z" />
        </svg>
      </button>
    );
  }

  return (
    <input
      ref={inputRef}
      className="w-full rounded border border-blue-400 bg-transparent px-1 text-base font-medium outline-none ring-1 ring-blue-400 sm:text-lg"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          setDraft(value);
          setEditing(false);
        }
      }}
    />
  );
}

/** Replay view for a single stored meeting. */
export function MeetingDetail({
  view,
  onRenameSpeaker,
  onRenameMeeting,
}: {
  view: Extract<MainView, { kind: "meeting" }>;
  onRenameSpeaker: (
    speakerId: SpeakerId,
    label: string | null,
  ) => Promise<void>;
  onRenameMeeting: (title: string) => Promise<void>;
}) {
  const meetingId = view.kind === "meeting" ? view.id : null;
  const summaryState = useMeetingSummary(meetingId);
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const splitRef = useRef<HTMLDivElement>(null);
  const summaryRef = useRef<HTMLDivElement>(null);
  const hSplitRef = useRef<HTMLDivElement>(null);

  /** Fraction of the split container given to the summary (top panel). */
  const MIN_RATIO = 1 / 3;
  const MAX_RATIO = 2 / 3;
  const [summaryRatio, setSummaryRatio] = useState(0.5);

  /** Horizontal split: transcript (left) vs notes (right). */
  const H_MIN = 0.4;
  const H_MAX = 0.85;
  const [hRatio, setHRatio] = useState(0.65);
  const clampedHRatio = Math.min(H_MAX, Math.max(H_MIN, hRatio));
  const handleHRatioChange = useCallback((r: number) => {
    setHRatio(Math.min(H_MAX, Math.max(H_MIN, r)));
  }, []);
  /** Track whether the user has manually dragged the handle. */
  const [userResized, setUserResized] = useState(false);
  const clampedRatio = Math.min(MAX_RATIO, Math.max(MIN_RATIO, summaryRatio));
  const handleRatioChange = useCallback(
    (r: number) => {
      const clamped = Math.min(MAX_RATIO, Math.max(MIN_RATIO, r));
      setUserResized(true);
      setSummaryRatio(clamped);
    },
    [],
  );

  // Reset to auto-fit when the summary is regenerated.
  const summaryId = summaryState.summary?.id;
  useEffect(() => {
    setUserResized(false);
    setSummaryRatio(0.5);
  }, [summaryId]);

  const m = view.meeting;
  const speakerIndex = useMemo(
    () => (m ? indexSpeakers(m.speakers) : new Map()),
    [m?.speakers],
  );

  // Local notes state allows optimistic deletes without refetching.
  const [notes, setNotes] = useState<Note[]>(m?.notes ?? []);
  useEffect(() => {
    setNotes(m?.notes ?? []);
  }, [m?.notes]);
  const handleNoteDeleted = useCallback((noteId: NoteId) => {
    setNotes((prev) => prev.filter((n) => n.id !== noteId));
  }, []);

  const getTranscriptText = useCallback(() => {
    if (!m) return "";
    return m.segments
      .map((seg) => {
        const ts = formatTimestamp(seg.startMs);
        const speaker = seg.speakerId ? speakerIndex.get(seg.speakerId) : undefined;
        const prefix = speaker ? `[${ts}] ${displayName(speaker)}:` : `[${ts}]`;
        return `${prefix} ${seg.text}`;
      })
      .join("\n");
  }, [m, speakerIndex]);

  const virtualizer = useVirtualizer({
    count: m?.segments.length ?? 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 36,
    overscan: 8,
  });

  if (view.loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-12">
        <LogoAnimated size={40} className="opacity-40" />
        <p className="text-sm text-zinc-500">{t("meeting.loading")}</p>
      </div>
    );
  }
  if (view.error || !m) {
    return (
      <p className="text-sm text-amber-700 dark:text-amber-400">
        {view.error ?? t("meeting.unavailable")}
      </p>
    );
  }

  // Compute summary panel sizing: auto-fit by default, fixed ratio
  // after the user drags the resize handle.
  let summaryPanelStyle: React.CSSProperties | undefined;
  if (summaryState.summary && userResized) {
    summaryPanelStyle = { flex: `0 0 ${(clampedRatio * 100).toFixed(1)}%` };
  } else if (summaryState.summary) {
    summaryPanelStyle = { flex: "0 1 auto", maxHeight: `${(MAX_RATIO * 100).toFixed(0)}%` };
  }

  return (
    <>
      <header className="flex flex-shrink-0 items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <EditableTitle value={m.title} onCommit={onRenameMeeting} />
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)} ·{" "}
            {m.language ?? "?"} · {m.segmentCount} {t("meeting.segments")}
          </p>
          <p className="font-mono text-[10px] text-zinc-400">{m.id}</p>
        </div>
        <ExportButton meetingId={m.id} title={m.title} />
      </header>

      <div className="flex min-h-0 flex-1 flex-col gap-3">
        {m.speakers.length > 0 && (
          <SpeakersPanel speakers={m.speakers} segments={m.segments} onRename={onRenameSpeaker} />
        )}

        {/* Resizable split: Summary (top) + Transcript+Notes (bottom) */}
        <div ref={splitRef} className="flex min-h-0 flex-1 flex-col">
          {/* ── Summary (fixed ratio only when content exists) ── */}
          <div
            ref={summaryRef}
            className="flex min-h-0 flex-col"
            style={summaryPanelStyle}
          >
            <SummaryPanel state={summaryState} />
          </div>

          {summaryState.summary && (
            <ResizableHandle
              containerRef={splitRef}
              ratio={clampedRatio}
              onRatioChange={handleRatioChange}
            />
          )}

          {/* ── Transcript + Notes side-by-side ── */}
          <div ref={hSplitRef} className="flex min-h-0 flex-1">
            {/* Left: Transcript */}
            <div
              className="flex min-h-0 min-w-0 flex-col"
              style={notes.length > 0 ? { flex: `0 0 ${(clampedHRatio * 100).toFixed(1)}%` } : { flex: "1 1 0%" }}
            >
              <div className="flex items-center justify-between px-1 py-1">
                <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
                  {t("meeting.transcript")}
                </span>
                {m.segments.length > 0 && (
                  <CopyButton getText={getTranscriptText} title={t("meeting.copyTranscript")} />
                )}
              </div>
              <div
                ref={scrollRef}
                className="min-h-0 flex-1 overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 text-sm leading-relaxed dark:border-zinc-900 dark:bg-zinc-900"
              >
                {m.segments.length === 0 ? (
                  <p className="text-zinc-400">{t("meeting.noSegments")}</p>
                ) : (
                  <ol
                    className="relative w-full"
                    style={{ height: `${virtualizer.getTotalSize()}px` }}
                  >
                    {virtualizer.getVirtualItems().map((vItem) => {
                      const seg = m.segments[vItem.index]!;
                      const speaker = seg.speakerId
                        ? speakerIndex.get(seg.speakerId)
                        : undefined;
                      return (
                        <li
                          key={seg.id}
                          data-segment-id={seg.id}
                          className="absolute left-0 top-0 flex w-full items-baseline gap-3 rounded-sm"
                          style={{ transform: `translateY(${vItem.start}px)` }}
                          data-index={vItem.index}
                          ref={virtualizer.measureElement}
                        >
                          <SegmentRow
                            startMs={seg.startMs}
                            text={seg.text}
                            speaker={speaker}
                            noSpeechLabel={t("meeting.noSpeech")}
                          />
                        </li>
                      );
                    })}
                  </ol>
                )}
              </div>
            </div>

            {/* Vertical resize handle + Notes panel (only when notes exist) */}
            {notes.length > 0 && (
              <>
                <ResizableHandleVertical
                  containerRef={hSplitRef}
                  ratio={clampedHRatio}
                  onRatioChange={handleHRatioChange}
                />
                <div className="flex min-h-0 min-w-0 flex-1 flex-col">
                  <NotesPanel notes={notes} onDeleted={handleNoteDeleted} />
                </div>
              </>
            )}
          </div>
        {/* ── end split ── */}
        </div>
      </div>
    </>
  );
}
