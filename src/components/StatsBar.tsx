import { statusLabel, type RecordingState } from "../state/recording";
import { Stat } from "./Stat";

/** Compact 4-column grid summarising the live session. */
export function StatsBar({
  stats,
  stream,
}: {
  stats: { chunks: number; skipped: number; audioMs: number };
  stream: RecordingState;
}) {
  return (
    <dl className="grid grid-cols-4 gap-x-4 text-xs">
      <Stat label="status" value={statusLabel(stream)} />
      <Stat label="chunks" value={String(stats.chunks)} />
      <Stat label="skipped" value={String(stats.skipped)} />
      <Stat label="audio" value={`${(stats.audioMs / 1000).toFixed(1)} s`} />
    </dl>
  );
}
