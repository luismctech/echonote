import { useCallback, useRef } from "react";

import { SpeakerChip } from "../../components/SpeakerChip";
import { useChat } from "../../hooks/useChat";
import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { formatDate, formatDurationMs, formatTimestamp } from "../../lib/format";
import { indexSpeakers, shortTag } from "../../lib/speakers";
import type { SegmentId } from "../../types/chat";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ChatPanel } from "./ChatPanel";
import { SpeakersPanel } from "./SpeakersPanel";
import { SummaryPanel } from "./SummaryPanel";

/** Replay view for a single stored meeting. */
export function MeetingDetail({
  view,
  onRenameSpeaker,
}: {
  view: Extract<MainView, { kind: "meeting" }>;
  onRenameSpeaker: (
    speakerId: SpeakerId,
    label: string | null,
  ) => Promise<void>;
}) {
  // Hooks must run on every render — call them BEFORE any of the
  // early returns below so React's rules-of-hooks invariant holds
  // when the meeting transitions from loading → loaded.
  const meetingId = view.kind === "meeting" ? view.id : null;
  const summaryState = useMeetingSummary(meetingId);
  const chat = useChat(meetingId);

  // Ref to the scrollable transcript container so citation clicks
  // in the ChatPanel can scroll the segment into view.
  const transcriptRef = useRef<HTMLDivElement>(null);

  const scrollToSegment = useCallback((segmentId: SegmentId) => {
    const container = transcriptRef.current;
    if (!container) return;
    const el = container.querySelector(`[data-segment-id="${segmentId}"]`);
    if (!el) return;
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    // Brief highlight so the user notices which segment was cited.
    el.classList.add("bg-blue-100", "dark:bg-blue-900/30");
    setTimeout(() => {
      el.classList.remove("bg-blue-100", "dark:bg-blue-900/30");
    }, 2000);
  }, []);

  if (view.loading) {
    return <p className="text-sm text-zinc-500">Loading meeting…</p>;
  }
  if (view.error || !view.meeting) {
    return (
      <p className="text-sm text-amber-700 dark:text-amber-400">
        {view.error ?? "Meeting unavailable."}
      </p>
    );
  }
  const m = view.meeting;
  const speakerIndex = indexSpeakers(m.speakers);
  return (
    <>
      <header className="flex flex-shrink-0 flex-col gap-1">
        <h2 className="text-base font-medium sm:text-lg">{m.title}</h2>
        <p className="text-xs text-zinc-500 dark:text-zinc-400">
          {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)} ·{" "}
          {m.language ?? "?"} · {m.segmentCount} segments
        </p>
        <p className="font-mono text-[10px] text-zinc-400">{m.id}</p>
      </header>

      {m.speakers.length > 0 && (
        <SpeakersPanel speakers={m.speakers} onRename={onRenameSpeaker} />
      )}

      <SummaryPanel
        summary={summaryState.summary}
        loading={summaryState.loading}
        generating={summaryState.generating}
        error={summaryState.error}
        onGenerate={() => {
          void summaryState.generate();
        }}
      />

      <ChatPanel chat={chat} onScrollToSegment={scrollToSegment} />

      <div
        ref={transcriptRef}
        className="min-h-0 flex-1 overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 text-sm leading-relaxed dark:border-zinc-900 dark:bg-zinc-900/60"
      >
        {m.segments.length === 0 ? (
          <p className="text-zinc-400">No segments persisted for this meeting.</p>
        ) : (
          <ol className="flex flex-col gap-2">
            {m.segments.map((seg) => {
              const speaker = seg.speakerId
                ? speakerIndex.get(seg.speakerId)
                : undefined;
              return (
                <li
                  key={seg.id}
                  data-segment-id={seg.id}
                  className="flex items-baseline gap-3 rounded-sm transition-colors duration-500"
                >
                  <span className="w-12 shrink-0 font-mono text-xs tabular-nums text-zinc-500">
                    {formatTimestamp(seg.startMs)}
                  </span>
                  {speaker && (
                    <SpeakerChip
                      slot={speaker.slot}
                      label={shortTag(speaker.slot)}
                      compact
                    />
                  )}
                  <span className="flex-1">
                    {seg.text.trim() || "[no speech]"}
                  </span>
                </li>
              );
            })}
          </ol>
        )}
      </div>
    </>
  );
}
