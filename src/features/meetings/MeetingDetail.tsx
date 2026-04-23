import { SpeakerChip } from "../../components/SpeakerChip";
import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { formatDate, formatDurationMs, formatTimestamp } from "../../lib/format";
import { indexSpeakers, shortTag } from "../../lib/speakers";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ExportButton } from "./ExportButton";
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
  const meetingId = view.kind === "meeting" ? view.id : null;
  const summaryState = useMeetingSummary(meetingId);

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
      <header className="flex flex-shrink-0 items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <h2 className="text-base font-medium sm:text-lg">{m.title}</h2>
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            {formatDate(m.startedAt)} · {formatDurationMs(m.durationMs)} ·{" "}
            {m.language ?? "?"} · {m.segmentCount} segments
          </p>
          <p className="font-mono text-[10px] text-zinc-400">{m.id}</p>
        </div>
        <ExportButton meetingId={m.id} title={m.title} />
      </header>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto">
        {m.speakers.length > 0 && (
          <SpeakersPanel speakers={m.speakers} onRename={onRenameSpeaker} />
        )}

        <SummaryPanel state={summaryState} />

        <div className="rounded-md border border-zinc-100 bg-zinc-50 p-3 text-sm leading-relaxed dark:border-zinc-900 dark:bg-zinc-900">
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
      </div>
    </>
  );
}
