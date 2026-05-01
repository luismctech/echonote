import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";
import { CopyButton } from "../../components/CopyButton";
import { ResizableHandleVertical } from "../../components/ResizableHandleVertical";

import { useChat } from "../../hooks/useChat";
import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { LogoAnimated } from "../../components/Logo";
import { formatDate, formatDurationMs, formatTimestamp } from "../../lib/format";
import { displayName, indexSpeakers } from "../../lib/speakers";
import type { Note, NoteId } from "../../types/meeting";
import type { SegmentId } from "../../types/chat";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ChatPanel } from "./ChatPanel";
import { ExportButton } from "./ExportButton";
import { NotesPanel } from "./NotesPanel";
import { SegmentRow } from "./SegmentRow";
import { SpeakersPanel } from "./SpeakersPanel";
import { SummaryPanel } from "./SummaryPanel";

type DetailTab = "summary" | "transcript" | "chat";

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
        className="group flex items-center gap-1.5 text-left text-[22px] font-semibold leading-tight tracking-tight"
        onClick={() => setEditing(true)}
        title={t("meeting.editTitle")}
      >
        <span>{value}</span>
        <svg
          className="h-3.5 w-3.5 shrink-0 text-content-placeholder opacity-0 transition-opacity group-hover:opacity-100"
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
      className="w-full rounded border border-blue-400 bg-transparent px-1 text-[22px] font-semibold leading-tight tracking-tight outline-none ring-1 ring-blue-400"
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
}: Readonly<{
  view: Extract<MainView, { kind: "meeting" }>;
  onRenameSpeaker: (
    speakerId: SpeakerId,
    label: string | null,
  ) => Promise<void>;
  onRenameMeeting: (title: string) => Promise<void>;
}>) {
  const meetingId = view.kind === "meeting" ? view.id : null;
  const summaryState = useMeetingSummary(meetingId);
  const chat = useChat(meetingId);
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);
  const splitRef = useRef<HTMLDivElement>(null);

  const [activeTab, setActiveTab] = useState<DetailTab>("transcript");

  // Resizable split ratio for notes | transcript
  const SPLIT_MIN = 0.25;
  const SPLIT_MAX = 0.65;
  const [splitRatio, setSplitRatio] = useState(0.5);
  const clampedSplit = Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, splitRatio));
  const handleSplitChange = useCallback((r: number) => {
    setSplitRatio(Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, r)));
  }, []);

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

  const handleScrollToSegment = useCallback((segmentId: SegmentId) => {
    if (!m) return;
    const idx = m.segments.findIndex((s) => s.id === segmentId);
    if (idx >= 0) {
      setActiveTab("transcript");
      // Small delay so the tab renders the virtualizer first
      requestAnimationFrame(() => virtualizer.scrollToIndex(idx, { align: "center" }));
    }
  }, [m, virtualizer]);

  if (view.loading) {
    return (
      <div className="flex flex-col items-center justify-center gap-3 py-12">
        <LogoAnimated size={40} className="opacity-40" />
        <p className="text-ui-md text-content-tertiary">{t("meeting.loading")}</p>
      </div>
    );
  }
  if (view.error || !m) {
    return (
      <p className="text-ui-md text-amber-700 dark:text-amber-400">
        {view.error ?? t("meeting.unavailable")}
      </p>
    );
  }

  const tabs: { id: DetailTab; label: string; count?: number }[] = [
    { id: "transcript", label: t("meeting.transcript"), count: m.segments.length },
    { id: "summary", label: t("summary.label") },
    { id: "chat", label: t("chat.label") },
  ];

  return (
    <>
      {/* ── Header ── */}
      <header className="flex flex-shrink-0 items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <EditableTitle value={m.title} onCommit={onRenameMeeting} />
          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 text-ui-sm text-content-tertiary">
            <span>{formatDate(m.startedAt)}</span>
            <span>·</span>
            <span>{formatDurationMs(m.durationMs)}</span>
            <span>·</span>
            <span>{m.language ?? "?"}</span>
            <span>·</span>
            <span>{m.segmentCount} {t("meeting.segments")}</span>
          </div>
        </div>
        <ExportButton meetingId={m.id} title={m.title} />
      </header>

      {/* ── Speakers (inline, compact) ── */}
      {m.speakers.length > 0 && (
        <SpeakersPanel speakers={m.speakers} segments={m.segments} onRename={onRenameSpeaker} />
      )}

      {/* ── Tab bar ── */}
      <nav className="flex gap-1 border-b border-subtle" aria-label="Meeting sections">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setActiveTab(tab.id)}
            className={`relative px-3 py-2 text-ui-sm font-medium transition-colors ${
              activeTab === tab.id
                ? "text-content-primary"
                : "text-content-tertiary hover:text-content-secondary"
            }`}
          >
            {tab.label}
            {tab.count != null && tab.count > 0 && (
              <span className="ml-1.5 inline-flex min-w-[18px] items-center justify-center rounded-full bg-surface-sunken px-1 text-micro font-medium tabular-nums text-content-tertiary">
                {tab.count}
              </span>
            )}
            {activeTab === tab.id && (
              <span className="absolute inset-x-0 -bottom-px h-0.5 rounded-full bg-content-primary" />
            )}
          </button>
        ))}
      </nav>

      {/* ── Tab content ── */}
      <div className="flex min-h-0 flex-1 flex-col">
        {activeTab === "summary" && (
          <SummaryPanel state={summaryState} />
        )}

        {activeTab === "transcript" && (
          <div ref={splitRef} className="flex min-h-0 flex-1 flex-row">
            {/* Notes panel (left) — only when notes exist */}
            {notes.length > 0 && (
              <>
                <div
                  className="flex min-h-0 min-w-0 flex-col overflow-y-auto"
                  style={{ width: `${clampedSplit * 100}%` }}
                >
                  <NotesPanel notes={notes} onDeleted={handleNoteDeleted} />
                </div>
                <ResizableHandleVertical
                  containerRef={splitRef}
                  ratio={clampedSplit}
                  onRatioChange={handleSplitChange}
                />
              </>
            )}

            {/* Transcript panel (right, or full-width if no notes) */}
            <div
              className="flex min-h-0 min-w-0 flex-1 flex-col gap-1"
              style={notes.length > 0 ? { width: `${(1 - clampedSplit) * 100}%` } : undefined}
            >
              <div className="flex items-center justify-between px-1 py-1">
                <span className="text-ui-xs font-medium uppercase tracking-wide text-content-placeholder">
                  {t("meeting.transcript")}
                </span>
                {m.segments.length > 0 && (
                  <CopyButton getText={getTranscriptText} title={t("meeting.copyTranscript")} />
                )}
              </div>
              <div
                ref={scrollRef}
                className="min-h-0 flex-1 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3 text-ui-md leading-relaxed"
              >
                {m.segments.length === 0 ? (
                  <p className="text-content-placeholder">{t("meeting.noSegments")}</p>
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
          </div>
        )}

        {activeTab === "chat" && (
          <div className="flex min-h-0 flex-1 flex-col">
            <ChatPanel
              chat={chat}
              onScrollToSegment={handleScrollToSegment}
              segmentTimestamps={m.segments.reduce<Record<string, number>>((acc, seg) => {
                acc[seg.id] = seg.startMs;
                return acc;
              }, {})}
            />
          </div>
        )}
      </div>
    </>
  );
}
