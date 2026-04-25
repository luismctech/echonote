import { useRef } from "react";
import { useTranslation } from "react-i18next";
import { useVirtualizer } from "@tanstack/react-virtual";

import { useMeetingSummary } from "../../hooks/useMeetingSummary";
import { formatDate, formatDurationMs } from "../../lib/format";
import { indexSpeakers } from "../../lib/speakers";
import type { SpeakerId } from "../../types/speaker";
import type { MainView } from "../../types/view";
import { ExportButton } from "./ExportButton";
import { SegmentRow } from "./SegmentRow";
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
  const { t } = useTranslation();
  const scrollRef = useRef<HTMLDivElement>(null);

  const m = view.meeting;
  const virtualizer = useVirtualizer({
    count: m?.segments.length ?? 0,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 36,
    overscan: 8,
  });

  if (view.loading) {
    return <p className="text-sm text-zinc-500">{t("meeting.loading")}</p>;
  }
  if (view.error || !m) {
    return (
      <p className="text-sm text-amber-700 dark:text-amber-400">
        {view.error ?? t("meeting.unavailable")}
      </p>
    );
  }
  const speakerIndex = indexSpeakers(m.speakers);

  return (
    <>
      <header className="flex flex-shrink-0 items-start justify-between gap-2">
        <div className="flex flex-col gap-1">
          <h2 className="text-base font-medium sm:text-lg">{m.title}</h2>
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
          <SpeakersPanel speakers={m.speakers} onRename={onRenameSpeaker} />
        )}

        <SummaryPanel state={summaryState} />

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
    </>
  );
}
