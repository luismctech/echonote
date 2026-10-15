import { useTranslation } from "react-i18next";

import { Button } from "../../components/Button";
import { MenuButton } from "../../components/MenuButton";
import type { Speaker, SpeakerId } from "../../types/speaker";
import { speakerMenuLabel } from "../../lib/speakers";

/**
 * Toolbar shown while segment selection mode is on: selected count,
 * "select all visible", reassign to a speaker (or none) and delete.
 * Actions are disabled while nothing is selected or an edit runs.
 */
export function SelectionToolbar({
  selectedCount,
  speakers,
  busy,
  onSelectAllVisible,
  onAssign,
  onDelete,
}: Readonly<{
  selectedCount: number;
  /** Speakers offered as reassignment targets (those with segments). */
  speakers: readonly Speaker[];
  busy: boolean;
  onSelectAllVisible: () => void;
  onAssign: (speakerId: SpeakerId | null) => void;
  onDelete: () => void;
}>) {
  const { t } = useTranslation();
  const none = selectedCount === 0 || busy;
  return (
    <div
      role="toolbar"
      aria-label={t("meeting.selectionToolbar")}
      className="flex flex-wrap items-center gap-2 rounded-md border border-subtle bg-surface-elevated px-2 py-1.5 text-ui-sm"
    >
      <span className="tabular-nums text-content-secondary" aria-live="polite">
        {t("meeting.selectedCount", { count: selectedCount })}
      </span>
      <Button size="sm" variant="ghost" onClick={onSelectAllVisible} disabled={busy}>
        {t("meeting.selectAllVisible")}
      </Button>
      <span className="flex-1" />
      <MenuButton
        label={t("meeting.assignTo")}
        disabled={none}
        align="right"
        triggerClassName="rounded-md border border-subtle bg-surface-elevated px-2.5 py-1 text-ui-xs font-medium text-content-secondary hover:bg-surface-sunken disabled:pointer-events-none disabled:opacity-50"
        items={[
          ...speakers.map((sp) => ({
            label: speakerMenuLabel(sp),
            onClick: () => onAssign(sp.id),
          })),
          { label: t("meeting.noSpeaker"), onClick: () => onAssign(null) },
        ]}
      />
      <Button size="sm" variant="danger" onClick={onDelete} disabled={none}>
        {t("meeting.deleteSelected", { count: selectedCount })}
      </Button>
    </div>
  );
}
