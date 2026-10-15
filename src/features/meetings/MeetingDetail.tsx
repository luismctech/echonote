import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";
import { MoreHorizontal, Pencil } from "lucide-react";
import { useConfirm } from "../../components/ConfirmDialog";
import { CopyButton } from "../../components/CopyButton";
import { MenuButton } from "../../components/MenuButton";
import { ResizableHandleVertical } from "../../components/ResizableHandleVertical";

import { useChat } from "../../hooks/useChat";
import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { useSpeakerTools } from "../../hooks/useSpeakerTools";
import { LogoAnimated } from "../../components/Logo";
import { formatDate, formatDurationMs, formatTimestamp } from "../../lib/format";
import {
  EMPTY_SELECTION,
  pruneSelection,
  selectAllVisible,
  selectRange,
  toggleSegment,
  type SegmentSelection,
} from "../../lib/segmentSelection";
import {
  chipLabel,
  displayName,
  indexSpeakers,
  speakerMenuLabel,
  speakersWithSegments,
} from "../../lib/speakers";
import {
  filterSegmentsBySpeakers,
  pruneSpeakerFilter,
  toggleSpeaker,
} from "../../lib/speakerFilter";
import type { Note, NoteId } from "../../types/meeting";
import type { Speaker } from "../../types/speaker";
import type { SegmentId } from "../../types/chat";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ChatPanel } from "./ChatPanel";
import { ExportButton } from "./ExportButton";
import { NotesPanel } from "./NotesPanel";
import { SegmentRow } from "./SegmentRow";
import { SelectionToolbar } from "./SelectionToolbar";
import { SpeakersPanel } from "./SpeakersPanel";
import { SummaryPanel } from "./SummaryPanel";

type DetailTab = "summary" | "transcript" | "chat";

/** Shared empty speaker filter (stable identity for memoisation). */
const NO_FILTER: ReadonlySet<SpeakerId> = new Set();

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
        <Pencil
          className="h-3.5 w-3.5 shrink-0 text-content-placeholder opacity-0 transition-opacity group-hover:opacity-100"
        />
      </button>
    );
  }

  return (
    <input
      ref={inputRef}
      className="w-full rounded-sm border border-blue-400 bg-transparent px-1 text-[22px] font-semibold leading-tight tracking-tight outline-hidden ring-1 ring-blue-400"
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
  onMeetingEdited,
}: Readonly<{
  view: Extract<MainView, { kind: "meeting" }>;
  onRenameSpeaker: (
    speakerId: SpeakerId,
    label: string | null,
  ) => Promise<void>;
  onRenameMeeting: (title: string) => Promise<void>;
  /** Called after a speaker merge or a segment edit (refresh the list). */
  onMeetingEdited?: () => Promise<void>;
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

  // Merge / reassign / delete reload the meeting and expose the
  // freshest copy as `freshMeeting` (falls back to the store's `view.meeting`).
  const {
    meeting: freshMeeting,
    busy: applyingEdit,
    error: editError,
    merge,
    reassign,
    remove,
    clearError,
  } = useSpeakerTools(view.meeting, onMeetingEdited);
  const { ask: askConfirm, dialog: confirmDialog } = useConfirm();
  const m = freshMeeting;
  const speakerIndex = useMemo(
    () => (m ? indexSpeakers(m.speakers) : new Map()),
    [m?.speakers],
  );
  // Speakers without a segment have no talk time: keep them out of the panel.
  const allSpeakers = m?.speakers;
  const segments = m?.segments;
  const activeSpeakers = useMemo(
    () => (allSpeakers && segments ? speakersWithSegments(allSpeakers, segments) : []),
    [allSpeakers, segments],
  );

  // Transcript filter by speaker (empty = every segment). It is stored
  // with the meeting it belongs to, so navigating to another meeting
  // starts unfiltered, and ids that disappear (a speaker merged away)
  // are pruned so the filter never hides everything.
  const [filterState, setFilterState] = useState<{
    meetingId: string | null;
    ids: ReadonlySet<SpeakerId>;
  }>({ meetingId: null, ids: NO_FILTER });
  const ownFilter = filterState.meetingId === meetingId ? filterState.ids : NO_FILTER;
  const speakerFilter = useMemo(
    () => pruneSpeakerFilter(ownFilter, activeSpeakers.map((sp) => sp.id)),
    [ownFilter, activeSpeakers],
  );
  const setSpeakerFilter = useCallback(
    (ids: ReadonlySet<SpeakerId>) => setFilterState({ meetingId, ids }),
    [meetingId],
  );
  const handleToggleSpeaker = useCallback(
    (speakerId: SpeakerId) => setSpeakerFilter(toggleSpeaker(speakerFilter, speakerId)),
    [setSpeakerFilter, speakerFilter],
  );
  const clearSpeakerFilter = useCallback(() => setSpeakerFilter(NO_FILTER), [setSpeakerFilter]);
  const visibleSegments = useMemo(
    () => (segments ? filterSegmentsBySpeakers(segments, speakerFilter) : []),
    [segments, speakerFilter],
  );

  // Segment selection mode. Like the filter it is tied to its meeting;
  // the selection is pruned to the visible segments, so hidden (filtered)
  // or deleted segments are never part of an action.
  const [selectionState, setSelectionState] = useState<{
    meetingId: string | null;
    active: boolean;
    selection: SegmentSelection;
  }>({ meetingId: null, active: false, selection: EMPTY_SELECTION });
  const selecting = selectionState.meetingId === meetingId && selectionState.active;
  const visibleIds = useMemo(() => visibleSegments.map((s) => s.id), [visibleSegments]);
  const selection = useMemo(
    () => (selecting ? pruneSelection(selectionState.selection, visibleIds) : EMPTY_SELECTION),
    [selecting, selectionState.selection, visibleIds],
  );
  const setSelection = useCallback(
    (next: SegmentSelection) => setSelectionState({ meetingId, active: true, selection: next }),
    [meetingId],
  );
  const toggleSelecting = useCallback(() => {
    setSelectionState({ meetingId, active: !selecting, selection: EMPTY_SELECTION });
  }, [meetingId, selecting]);
  const handleSelectSegment = useCallback(
    (id: SegmentId, shift: boolean) =>
      setSelection(shift ? selectRange(selection, visibleIds, id) : toggleSegment(selection, id)),
    [selection, visibleIds, setSelection],
  );

  const selectedIds = useCallback(
    () => visibleIds.filter((id) => selection.selected.has(id)),
    [visibleIds, selection],
  );
  const handleAssign = useCallback(
    async (speakerId: SpeakerId | null) => {
      if (await reassign(selectedIds(), speakerId)) setSelection(EMPTY_SELECTION);
    },
    [reassign, selectedIds, setSelection],
  );
  const handleDeleteSelected = useCallback(() => {
    const ids = selectedIds();
    askConfirm({
      title: t("meeting.deleteTitle"),
      message: t("meeting.deleteConfirm", { count: ids.length }),
      confirmLabel: t("meeting.deleteSelected", { count: ids.length }),
      danger: true,
      onConfirm: () => {
        void remove(ids).then((ok) => {
          if (ok) setSelection(EMPTY_SELECTION);
        });
      },
    });
  }, [selectedIds, askConfirm, t, remove, setSelection]);
  const handleMerge = useCallback(
    (absorbed: Speaker, surviving: Speaker) => {
      const from = chipLabel(absorbed);
      const to = chipLabel(surviving);
      askConfirm({
        title: t("speakers.mergeTitle"),
        message: t("speakers.mergeConfirm", { from, to }),
        confirmLabel: t("speakers.mergeAction"),
        danger: true,
        onConfirm: () => void merge(absorbed.id, surviving.id),
      });
    },
    [askConfirm, t, merge],
  );
  const renderSpeakerActions = useCallback(
    (sp: Speaker) => {
      const others = activeSpeakers.filter((other) => other.id !== sp.id);
      if (others.length === 0) return null;
      const name = displayName(sp, t);
      return (
        <MenuButton
          label={<MoreHorizontal className="h-3 w-3" />}
          ariaLabel={t("speakers.actions", { name })}
          title={t("speakers.actions", { name })}
          heading={t("speakers.mergeInto")}
          disabled={applyingEdit}
          triggerClassName="rounded-full p-1 opacity-60 hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-hidden focus-visible:ring-2 focus-visible:ring-accent-400"
          items={others.map((other) => ({
            label: speakerMenuLabel(other),
            onClick: () => handleMerge(sp, other),
          }))}
        />
      );
    },
    [activeSpeakers, t, applyingEdit, handleMerge],
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
    count: visibleSegments.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 36,
    overscan: 8,
  });

  const handleScrollToSegment = useCallback((segmentId: SegmentId) => {
    if (!m) return;
    let idx = visibleSegments.findIndex((s) => s.id === segmentId);
    if (idx < 0) {
      // The cited segment is hidden by the speaker filter: show all.
      idx = m.segments.findIndex((s) => s.id === segmentId);
      if (idx >= 0) setSpeakerFilter(NO_FILTER);
    }
    if (idx >= 0) {
      setActiveTab("transcript");
      // Small delay so the tab renders the virtualizer first
      requestAnimationFrame(() => virtualizer.scrollToIndex(idx, { align: "center" }));
    }
  }, [m, visibleSegments, virtualizer, setSpeakerFilter]);

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
      <header className="flex shrink-0 items-start justify-between gap-2">
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
      {activeSpeakers.length > 0 && (
        <SpeakersPanel
          speakers={activeSpeakers}
          segments={m.segments}
          filter={speakerFilter}
          onToggleFilter={handleToggleSpeaker}
          onRename={onRenameSpeaker}
          renderActions={renderSpeakerActions}
        />
      )}

      {editError && (
        <p
          role="alert"
          className="flex items-start justify-between gap-2 rounded-md bg-red-50 px-3 py-2 text-ui-sm text-red-700 dark:bg-red-950/40 dark:text-red-300"
        >
          <span>{editError}</span>
          <button
            type="button"
            onClick={clearError}
            className="shrink-0 text-ui-xs font-medium underline-offset-2 hover:underline"
          >
            {t("meeting.dismissError")}
          </button>
        </p>
      )}
      {confirmDialog}

      {/* ── Tab bar ── */}
      <nav className="flex gap-1 border-b border-subtle" aria-label={t("meeting.sectionsNav")}>
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
                  <span className="flex items-center gap-1">
                    <button
                      type="button"
                      aria-pressed={selecting}
                      onClick={toggleSelecting}
                      className={`rounded-md px-2 py-0.5 text-ui-xs font-medium transition-colors ${
                        selecting
                          ? "bg-surface-elevated text-content-primary ring-1 ring-content-primary/40"
                          : "text-content-tertiary hover:bg-surface-sunken hover:text-content-secondary"
                      }`}
                    >
                      {t("meeting.select")}
                    </button>
                    <CopyButton getText={getTranscriptText} title={t("meeting.copyTranscript")} />
                  </span>
                )}
              </div>
              {selecting && m.segments.length > 0 && (
                <SelectionToolbar
                  selectedCount={selection.selected.size}
                  speakers={activeSpeakers}
                  busy={applyingEdit}
                  onSelectAllVisible={() => setSelection(selectAllVisible(selection, visibleIds))}
                  onAssign={(speakerId) => void handleAssign(speakerId)}
                  onDelete={handleDeleteSelected}
                />
              )}
              {speakerFilter.size > 0 && (
                <p className="px-1 text-ui-xs text-content-tertiary" aria-live="polite">
                  {t("meeting.filterStatus", {
                    shown: visibleSegments.length,
                    total: m.segments.length,
                  })}
                  {" · "}
                  <button
                    type="button"
                    onClick={clearSpeakerFilter}
                    className="font-medium text-content-secondary underline-offset-2 hover:underline"
                  >
                    {t("meeting.clearFilter")}
                  </button>
                </p>
              )}
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
                      const seg = visibleSegments[vItem.index]!;
                      const speaker = seg.speakerId
                        ? speakerIndex.get(seg.speakerId)
                        : undefined;
                      const checked = selection.selected.has(seg.id);
                      return (
                        <li
                          key={seg.id}
                          data-segment-id={seg.id}
                          className={`absolute left-0 top-0 flex w-full items-baseline gap-3 rounded-xs ${
                            checked ? "bg-accent-400/15" : ""
                          } ${selecting ? "select-none" : ""}`}
                          style={{ transform: `translateY(${vItem.start}px)` }}
                          data-index={vItem.index}
                          ref={virtualizer.measureElement}
                        >
                          {selecting && (
                            <input
                              type="checkbox"
                              readOnly
                              checked={checked}
                              onClick={(e) => handleSelectSegment(seg.id, e.shiftKey)}
                              aria-label={t("meeting.selectSegment", {
                                time: formatTimestamp(seg.startMs),
                              })}
                              className="shrink-0 translate-y-0.5 accent-accent-600"
                            />
                          )}
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
