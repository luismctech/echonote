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
  const { summary, loading, generating, error, selectedTemplate, setSelectedTemplate, includeNotes, setIncludeNotes } = state;
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
        <span className="text-xs font-medium uppercase tracking-wide text-zinc-400">
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
            className="rounded-md p-1 text-zinc-400 hover:bg-zinc-100 hover:text-zinc-700 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
            title={t("templates.manage")}
          >
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
              <path fillRule="evenodd" d="M6.455 1.45A.5.5 0 0 1 6.952 1h2.096a.5.5 0 0 1 .497.45l.186 1.858a4.996 4.996 0 0 1 1.466.848l1.703-.769a.5.5 0 0 1 .63.207l1.048 1.814a.5.5 0 0 1-.133.656l-1.517 1.09a5.026 5.026 0 0 1 0 1.694l1.517 1.09a.5.5 0 0 1 .133.656l-1.048 1.814a.5.5 0 0 1-.63.207l-1.703-.769a4.996 4.996 0 0 1-1.466.848l-.186 1.858a.5.5 0 0 1-.497.45H6.952a.5.5 0 0 1-.497-.45l-.186-1.858a4.993 4.993 0 0 1-1.466-.848l-1.703.769a.5.5 0 0 1-.63-.207L1.422 12.4a.5.5 0 0 1 .133-.656l1.517-1.09a5.026 5.026 0 0 1 0-1.694l-1.517-1.09a.5.5 0 0 1-.133-.656l1.048-1.814a.5.5 0 0 1 .63-.207l1.703.769a4.993 4.993 0 0 1 1.466-.848l.186-1.858ZM8 10.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" clipRule="evenodd" />
            </svg>
          </button>
          <label className="flex cursor-pointer items-center gap-1 text-[11px] text-zinc-500 dark:text-zinc-400">
            <input
              type="checkbox"
              checked={includeNotes}
              onChange={(e) => setIncludeNotes(e.target.checked)}
              disabled={loading || generating}
              className="h-3 w-3 rounded border-zinc-300 accent-emerald-600 dark:border-zinc-600"
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
      <div className="min-h-0 flex-1 overflow-y-auto rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-900 dark:bg-zinc-900/40">
        {error && (
          <p className="text-xs text-amber-700 dark:text-amber-400">{error}</p>
        )}

        {loading && (
          <div className="flex items-center gap-2">
            <LogoAnimated size={20} className="opacity-40" />
            <p className="text-xs text-zinc-500">{t("summary.loading")}</p>
          </div>
        )}
        {!loading && summary && <SummaryBody summary={summary} />}
        {!loading && !summary && (
          <p className="text-xs text-zinc-500">
            {t("summary.empty")}
          </p>
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
      className="rounded-md border border-zinc-200 bg-white px-1.5 py-1 text-xs disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-900"
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
      className="rounded-md border border-zinc-200 bg-white px-2 py-1 text-xs font-medium hover:bg-zinc-100 disabled:cursor-not-allowed disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-900 dark:hover:bg-zinc-800"
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
    <div className="flex flex-col gap-3 text-sm">
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
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title={t("summary.sections.keyPoints")} items={summary.keyPoints} />
          <SummarySection title={t("summary.sections.decisions")} items={summary.decisions} />
          <ActionItemsSection items={summary.actionItems} title={t("summary.sections.actionItems")} />
          <SummarySection title={t("summary.sections.openQuestions")} items={summary.openQuestions} />
        </>
      );

    case "oneOnOne":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
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
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title={t("summary.sections.completed")} items={summary.completedItems} />
          <SummarySection title={t("summary.sections.carryOver")} items={summary.carryOver} />
          <SummarySection title={t("summary.sections.risks")} items={summary.risks} />
          <SummarySection title={t("summary.sections.nextSprintPriorities")} items={summary.nextSprintPriorities} />
        </>
      );

    case "interview":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <QuotesSection quotes={summary.quotes} />
          <SummarySection title={t("summary.sections.themes")} items={summary.themes} />
          <SummarySection title={t("summary.sections.painPoints")} items={summary.painPoints} />
          <SummarySection title={t("summary.sections.opportunities")} items={summary.opportunities} />
        </>
      );

    case "salesCall":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          {summary.customerContext && (
            <div className="flex flex-col gap-1">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                {t("summary.sections.customerContext")}
              </h4>
              <p className="text-zinc-800 dark:text-zinc-200">{summary.customerContext}</p>
            </div>
          )}
          <SummarySection title={t("summary.sections.painPoints")} items={summary.painPoints} />
          <SummarySection title={t("summary.sections.interestSignals")} items={summary.interestSignals} />
          <SummarySection title={t("summary.sections.objections")} items={summary.objections} />
          <ActionItemsSection items={summary.nextSteps} title={t("summary.sections.nextSteps")} />
          {summary.dealStageIndicator && (
            <p className="text-xs text-zinc-500">
              {t("summary.sections.dealStage")} <span className="font-medium">{summary.dealStageIndicator}</span>
            </p>
          )}
        </>
      );

    case "lecture":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title={t("summary.sections.conceptsCovered")} items={summary.conceptsCovered} />
          <DefinitionsSection definitions={summary.definitions} />
          <SummarySection title={t("summary.sections.examples")} items={summary.examples} />
          <SummarySection title={t("summary.sections.homeworkNext")} items={summary.homeworkOrNext} />
        </>
      );

    case "freeText":
      return (
        <div className="prose prose-sm prose-zinc dark:prose-invert max-w-none">
          <Markdown>{summary.text || t("summary.emptySummary")}</Markdown>
        </div>
      );

    case "custom":
      return (
        <div className="flex flex-col gap-2">
          <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
            {summary.templateName}
          </p>
          <div className="prose prose-sm prose-zinc dark:prose-invert max-w-none">
            <Markdown>{summary.text || t("summary.emptySummary")}</Markdown>
          </div>
        </div>
      );

    default:
      return (
        <p className="text-zinc-500 italic">
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
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {title}
      </h4>
      <ul className="ml-4 list-disc space-y-1 text-zinc-800 dark:text-zinc-200">
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
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {title}
      </h4>
      <ul className="ml-4 list-disc space-y-1 text-zinc-800 dark:text-zinc-200">
        {items.map((it, i) => (
          <li key={`action-${i}`}>
            <span>{it.task}</span>
            {it.owner && (
              <span className="ml-2 text-xs text-zinc-500">— {it.owner}</span>
            )}
            {it.due && (
              <span className="ml-2 text-xs text-zinc-500">· {it.due}</span>
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
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {t("summary.sections.notableQuotes")}
      </h4>
      <ul className="ml-4 space-y-2 text-zinc-800 dark:text-zinc-200">
        {quotes.map((q, i) => (
          <li key={`quote-${i}`}>
            <blockquote className="border-l-2 border-zinc-300 pl-2 italic dark:border-zinc-600">
              "{q.quote}"
            </blockquote>
            <p className="text-xs text-zinc-500">
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
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {t("summary.sections.definitions")}
      </h4>
      <dl className="ml-4 space-y-1 text-zinc-800 dark:text-zinc-200">
        {definitions.map((d, i) => (
          <div key={`def-${i}`}>
            <dt className="font-medium">{d.term}</dt>
            <dd className="ml-4 text-zinc-600 dark:text-zinc-400">{d.definition}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function Footer({ summary }: Readonly<{ summary: Summary }>) {
  return (
    <p className="text-[10px] text-zinc-400">
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
