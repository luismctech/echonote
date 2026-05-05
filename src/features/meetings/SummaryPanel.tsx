/**
 * `SummaryPanel` — render the LLM summary for a single meeting.
 *
 * Three view states, in this order:
 *
 *   1. **Loading**   the initial `getSummary` is in flight.
 *   2. **Empty**     the meeting has no summary yet → render a CTA
 *                    button that calls `summarize_meeting`.
 *   3. **Loaded**    a `Summary` exists; render the structured
 *                    sections, plus a "Regenerate" affordance.
 *
 * A template selector lets the user pick any of the six built-in
 * templates or a user-defined custom template before generating. The
 * selector syncs with the loaded summary's template on mount so
 * "Regenerate" targets the right one.
 */

import { useCallback, useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
import Markdown from "react-markdown";
import { Settings, FileText } from "lucide-react";
import { CopyButton } from "../../components/CopyButton";
import { LogoAnimated } from "../../components/Logo";
import { formatDate } from "../../lib/format";
import type { Summary, TemplateId } from "../../types/summary";
import { TEMPLATE_IDS, TEMPLATE_LABELS } from "../../types/summary";
import type { UseMeetingSummary, SelectedTemplate } from "../../hooks/useMeetingSummary";
import type { CustomTemplate } from "../../types/custom-template";
import { listCustomTemplates } from "../../ipc/client";
import { TemplateManager } from "../settings/TemplateManager";

export function SummaryPanel({
  state,
}: Readonly<{
  state: UseMeetingSummary;
}>) {
  const { summary, loading, generating, streamingText, error, selectedTemplate, setSelectedTemplate, includeNotes, setIncludeNotes } = state;
  const { t } = useTranslation();
  const [customTemplates, setCustomTemplates] = useState<CustomTemplate[]>([]);
  const [showTemplateManager, setShowTemplateManager] = useState(false);

  const refreshCustomTemplates = () => {
    listCustomTemplates().then(setCustomTemplates).catch(() => {});
  };

  useEffect(() => {
    refreshCustomTemplates();
  }, []);

  const getSummaryText = useCallback(() => {
    if (!summary) return "";
    return summaryToPlainText(summary);
  }, [summary]);

  return (
    <section
      aria-label={t("summary.label")}
      className="flex min-h-0 flex-col"
    >
      {/* ── Header bar (never scrolls) ── */}
      <div className="flex flex-shrink-0 items-center justify-between px-1 py-1">
        <span className="type-section-header text-content-placeholder">
          {t("summary.label")}
        </span>
        <div className="flex items-center gap-1.5">
          <TemplateSelector
            value={selectedTemplate}
            onChange={setSelectedTemplate}
            disabled={loading || generating}
            customTemplates={customTemplates}
          />
          <button
            type="button"
            onClick={() => setShowTemplateManager(true)}
            className="rounded-md p-1 text-content-tertiary hover:bg-surface-inset hover:text-content-primary"
            title={t("templates.manage")}
          >
            <Settings className="h-3.5 w-3.5" />
          </button>
          <label className="flex cursor-pointer items-center gap-1 text-ui-xs text-content-tertiary">
            <input
              type="checkbox"
              checked={includeNotes}
              onChange={(e) => setIncludeNotes(e.target.checked)}
              disabled={loading || generating}
              className=""
            />
            {t("summary.includeNotes")}
          </label>
          {summary && <CopyButton getText={getSummaryText} title={t("meeting.copySummary")} />}
          <SummaryActions
            summary={summary}
            generating={generating}
            loading={loading}
            onGenerate={() => {
              void state.generate();
            }}
          />
        </div>
      </div>

      {showTemplateManager && (
        <TemplateManager
          onClose={() => setShowTemplateManager(false)}
          onChanged={refreshCustomTemplates}
        />
      )}

      {/* ── Scrollable body ── */}
      <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-subtle bg-surface-sunken p-3">
        {error && (
          <p className="text-ui-sm text-amber-700 dark:text-amber-400">{error}</p>
        )}

        {loading && (
          <div className="flex items-center gap-2">
            <LogoAnimated size={20} className="opacity-40" />
            <p className="text-ui-sm text-content-tertiary">{t("summary.loading")}</p>
          </div>
        )}
        {!loading && generating && streamingText && (
          <div className="prose prose-sm max-w-none text-ui-md leading-relaxed">
            <Markdown>{streamingText}</Markdown>
            <span className="ml-0.5 inline-block h-3 w-1.5 animate-pulse rounded-sm bg-emerald-500" />
          </div>
        )}
        {!loading && generating && !streamingText && (
          <div className="flex items-center gap-2">
            <LogoAnimated size={20} className="opacity-40" />
            <p className="text-ui-sm text-content-tertiary">{t("summary.generating")}</p>
          </div>
        )}
        {!loading && !generating && summary && <SummaryBody summary={summary} />}
        {!loading && !generating && !summary && (
          <div className="flex flex-col items-center justify-center gap-3 py-12 text-center">
            <FileText className="h-10 w-10 text-content-placeholder opacity-40" />
            <p className="text-ui-sm text-content-placeholder">{t("summary.empty")}</p>
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Template selector
// ---------------------------------------------------------------------------

function TemplateSelector({
  value,
  onChange,
  disabled,
  customTemplates,
}: Readonly<{
  value: SelectedTemplate;
  onChange: (t: SelectedTemplate) => void;
  disabled: boolean;
  customTemplates: CustomTemplate[];
}>) {
  const selectValue =
    value.kind === "builtin" ? value.id : `custom:${value.id}`;

  const { t } = useTranslation();
  const handleChange = (raw: string) => {
    if (raw.startsWith("custom:")) {
      const cid = raw.slice("custom:".length);
      const ct = customTemplates.find((t) => t.id === cid);
      if (ct) {
        onChange({ kind: "custom", id: ct.id, name: ct.name });
      }
    } else {
      onChange({ kind: "builtin", id: raw as TemplateId });
    }
  };

  return (
    <select
      value={selectValue}
      onChange={(e) => handleChange(e.target.value)}
      disabled={disabled}
      className="rounded-md border bg-surface-elevated px-1.5 py-1 text-ui-sm disabled:opacity-60"
    >
      {TEMPLATE_IDS.map((id) => (
        <option key={id} value={id}>
          {TEMPLATE_LABELS[id]}
        </option>
      ))}
      {customTemplates.length > 0 && (
        <optgroup label={t("templates.customGroup")}>
          {customTemplates.map((ct) => (
            <option key={ct.id} value={`custom:${ct.id}`}>
              {ct.name}
            </option>
          ))}
        </optgroup>
      )}
    </select>
  );
}

// ---------------------------------------------------------------------------
// Actions (Generate / Regenerate)
// ---------------------------------------------------------------------------

function SummaryActions({
  summary,
  generating,
  loading,
  onGenerate,
}: Readonly<{
  summary: Summary | null;
  generating: boolean;
  loading: boolean;
  onGenerate: () => void;
}>) {
  const { t } = useTranslation();
  const disabled = loading || generating;
  let label = t("summary.generate");
  if (generating) label = t("summary.generating");
  else if (summary) label = t("summary.regenerate");
  return (
    <button
      type="button"
      onClick={onGenerate}
      disabled={disabled}
      className="rounded-md border bg-surface-elevated px-2 py-1 text-ui-sm font-medium hover:bg-surface-sunken disabled:cursor-not-allowed disabled:opacity-60"
    >
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Summary body — dispatch on template
// ---------------------------------------------------------------------------

function SummaryBody({ summary }: Readonly<{ summary: Summary }>) {
  const { t } = useTranslation();
  return (
    <div className="flex flex-col gap-3 text-ui-md">
      {renderTemplateBody(summary, t)}
      <Footer summary={summary} />
    </div>
  );
}

function renderTemplateBody(summary: Summary, t: (key: string) => string) {
  switch (summary.template) {
    case "general":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          <SummarySection title={t("summary.sections.keyPoints")} items={summary.keyPoints} />
          <SummarySection title={t("summary.sections.decisions")} items={summary.decisions} />
          <ActionItemsSection items={summary.actionItems} title={t("summary.sections.actionItems")} />
          <SummarySection title={t("summary.sections.openQuestions")} items={summary.openQuestions} />
        </>
      );

    case "oneOnOne":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          <SummarySection title={t("summary.sections.wins")} items={summary.wins} />
          <SummarySection title={t("summary.sections.blockers")} items={summary.blockers} />
          <SummarySection title={t("summary.sections.growthFeedback")} items={summary.growthFeedback} />
          <ActionItemsSection items={summary.nextSteps} title={t("summary.sections.nextSteps")} />
          <SummarySection title={t("summary.sections.followUp")} items={summary.followUpTopics} />
        </>
      );

    case "sprintReview":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          <SummarySection title={t("summary.sections.completed")} items={summary.completedItems} />
          <SummarySection title={t("summary.sections.carryOver")} items={summary.carryOver} />
          <SummarySection title={t("summary.sections.risks")} items={summary.risks} />
          <SummarySection title={t("summary.sections.nextSprintPriorities")} items={summary.nextSprintPriorities} />
        </>
      );

    case "interview":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          <QuotesSection quotes={summary.quotes} />
          <SummarySection title={t("summary.sections.themes")} items={summary.themes} />
          <SummarySection title={t("summary.sections.painPoints")} items={summary.painPoints} />
          <SummarySection title={t("summary.sections.opportunities")} items={summary.opportunities} />
        </>
      );

    case "salesCall":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          {summary.customerContext && (
            <div className="flex flex-col gap-1">
              <h4 className="type-section-header">
                {t("summary.sections.customerContext")}
              </h4>
              <p className="text-content-primary">{summary.customerContext}</p>
            </div>
          )}
          <SummarySection title={t("summary.sections.painPoints")} items={summary.painPoints} />
          <SummarySection title={t("summary.sections.interestSignals")} items={summary.interestSignals} />
          <SummarySection title={t("summary.sections.objections")} items={summary.objections} />
          <ActionItemsSection items={summary.nextSteps} title={t("summary.sections.nextSteps")} />
          {summary.dealStageIndicator && (
            <p className="text-ui-sm text-content-tertiary">
              {t("summary.sections.dealStage")} <span className="font-medium">{summary.dealStageIndicator}</span>
            </p>
          )}
        </>
      );

    case "lecture":
      return (
        <>
          <p className="text-content-primary">{summary.summary}</p>
          <SummarySection title={t("summary.sections.conceptsCovered")} items={summary.conceptsCovered} />
          <DefinitionsSection definitions={summary.definitions} />
          <SummarySection title={t("summary.sections.examples")} items={summary.examples} />
          <SummarySection title={t("summary.sections.homeworkNext")} items={summary.homeworkOrNext} />
        </>
      );

    case "freeText":
      return (
        <div className="prose prose-sm max-w-none">
          <Markdown>{summary.text || t("summary.emptySummary")}</Markdown>
        </div>
      );

    case "custom":
      return (
        <div className="flex flex-col gap-2">
          <p className="text-ui-sm font-medium text-content-tertiary">
            {summary.templateName}
          </p>
          <div className="prose prose-sm max-w-none">
            <Markdown>{summary.text || t("summary.emptySummary")}</Markdown>
          </div>
        </div>
      );

    default:
      return (
        <p className="text-content-tertiary italic">
          {t("summary.unknown")}
        </p>
      );
  }
}

// ---------------------------------------------------------------------------
// Shared section components
// ---------------------------------------------------------------------------

function SummarySection({
  title,
  items,
}: Readonly<{
  title: string;
  items: string[];
}>) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="type-section-header">
        {title}
      </h4>
      <ul className="ml-4 list-disc space-y-1 text-content-primary">
        {items.map((it, i) => (
          <li key={`${title}-${i}`}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

function ActionItemsSection({
  items,
  title,
}: Readonly<{
  items: { task: string; owner?: string | null; due?: string | null }[];
  title: string;
}>) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="type-section-header">
        {title}
      </h4>
      <ul className="ml-4 list-disc space-y-1 text-content-primary">
        {items.map((it, i) => (
          <li key={`action-${i}`}>
            <span>{it.task}</span>
            {it.owner && (
              <span className="ml-2 text-ui-sm text-content-tertiary">— {it.owner}</span>
            )}
            {it.due && (
              <span className="ml-2 text-ui-sm text-content-tertiary">· {it.due}</span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

function QuotesSection({
  quotes,
}: Readonly<{
  quotes: { speaker: string; quote: string; context?: string | null }[];
}>) {
  const { t } = useTranslation();
  if (quotes.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="type-section-header">
        {t("summary.sections.notableQuotes")}
      </h4>
      <ul className="ml-4 space-y-2 text-content-primary">
        {quotes.map((q, i) => (
          <li key={`quote-${i}`}>
            <blockquote className="border-l-2 border-strong pl-2 italic">
              "{q.quote}"
            </blockquote>
            <p className="text-ui-sm text-content-tertiary">
              — {q.speaker}
              {q.context && <span> · {q.context}</span>}
            </p>
          </li>
        ))}
      </ul>
    </div>
  );
}

function DefinitionsSection({
  definitions,
}: Readonly<{
  definitions: { term: string; definition: string }[];
}>) {
  const { t } = useTranslation();
  if (definitions.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="type-section-header">
        {t("summary.sections.definitions")}
      </h4>
      <dl className="ml-4 space-y-1 text-content-primary">
        {definitions.map((d, i) => (
          <div key={`def-${i}`}>
            <dt className="font-medium">{d.term}</dt>
            <dd className="ml-4 text-content-secondary">{d.definition}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Footer({ summary }: Readonly<{ summary: Summary }>) {
  return (
    <p className="text-micro text-content-placeholder">
      {summary.model} · {formatDate(summary.createdAt)}
      {summary.language ? ` · ${summary.language}` : ""}
    </p>
  );
}

// ---------------------------------------------------------------------------
// Plain-text serialization for clipboard copy
// ---------------------------------------------------------------------------

function sectionText(title: string, items: string[]): string {
  if (items.length === 0) return "";
  const bullets = items.map((it) => "• " + it).join("\n");
  return "\n" + title + "\n" + bullets;
}

function actionItemsText(title: string, items: { task: string; owner?: string | null; due?: string | null }[]): string {
  if (items.length === 0) return "";
  const bullets = items.map((a) => {
    let line = "• " + a.task;
    if (a.owner) line += " (" + a.owner + ")";
    if (a.due) line += " — " + a.due;
    return line;
  }).join("\n");
  return "\n" + title + "\n" + bullets;
}

function summaryToPlainText(s: Summary): string {
  switch (s.template) {
    case "general":
      return [
        s.summary,
        sectionText("Key points", s.keyPoints),
        sectionText("Decisions", s.decisions),
        actionItemsText("Action items", s.actionItems),
        sectionText("Open questions", s.openQuestions),
      ].filter(Boolean).join("\n");

    case "oneOnOne":
      return [
        s.summary,
        sectionText("Wins", s.wins),
        sectionText("Blockers", s.blockers),
        sectionText("Growth feedback", s.growthFeedback),
        actionItemsText("Next steps", s.nextSteps),
        sectionText("Follow-up topics", s.followUpTopics),
      ].filter(Boolean).join("\n");

    case "sprintReview":
      return [
        s.summary,
        sectionText("Completed", s.completedItems),
        sectionText("Carry-over", s.carryOver),
        sectionText("Risks", s.risks),
        sectionText("Next sprint priorities", s.nextSprintPriorities),
      ].filter(Boolean).join("\n");

    case "interview":
      return [
        s.summary,
        s.quotes.length > 0 ? "\nQuotes\n" + s.quotes.map((q) => "• \"" + q.quote + "\" — " + q.speaker).join("\n") : "",
        sectionText("Themes", s.themes),
        sectionText("Pain points", s.painPoints),
        sectionText("Opportunities", s.opportunities),
      ].filter(Boolean).join("\n");

    case "salesCall":
      return [
        s.summary,
        s.customerContext ? `\nCustomer context\n${s.customerContext}` : "",
        sectionText("Pain points", s.painPoints),
        sectionText("Interest signals", s.interestSignals),
        sectionText("Objections", s.objections),
        actionItemsText("Next steps", s.nextSteps),
        s.dealStageIndicator ? `\nDeal stage: ${s.dealStageIndicator}` : "",
      ].filter(Boolean).join("\n");

    case "lecture":
      return [
        s.summary,
        sectionText("Concepts covered", s.conceptsCovered),
        s.definitions.length > 0 ? "\nDefinitions\n" + s.definitions.map((d) => "• " + d.term + ": " + d.definition).join("\n") : "",
        sectionText("Examples", s.examples),
        sectionText("Homework / next", s.homeworkOrNext),
      ].filter(Boolean).join("\n");

    case "freeText":
    case "custom":
      return s.text || "";

    default:
      return "";
  }
}
