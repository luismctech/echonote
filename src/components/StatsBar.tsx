import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
  return (
    <dl className="grid grid-cols-4 gap-x-4 text-xs">
      <Stat label={t("stats.status")} value={status} />
      <Stat label={t("stats.chunks")} value={String(stats.chunks)} />
      <Stat label={t("stats.skipped")} value={String(stats.skipped)} />
      <Stat label={t("stats.audio")} value={`${(stats.audioMs / 1000).toFixed(1)} s`} />
    </dl>
  );
}
