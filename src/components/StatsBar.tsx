import { Stat } from "./Stat";

/**
 * Compact 4-column grid summarising the live session.
 *
 * Receives `status` as a pre-formatted string so this primitive stays
 * dependency-free of the recording state machine — the live pane that
 * owns the FSM is responsible for translating it into a label.
 */
export function StatsBar({
  status,
  stats,
}: {
  status: string;
  stats: { chunks: number; skipped: number; audioMs: number };
}) {
  return (
    <dl className="grid grid-cols-4 gap-x-4 text-xs">
      <Stat label="status" value={status} />
      <Stat label="chunks" value={String(stats.chunks)} />
      <Stat label="skipped" value={String(stats.skipped)} />
      <Stat label="audio" value={`${(stats.audioMs / 1000).toFixed(1)} s`} />
    </dl>
  );
}
