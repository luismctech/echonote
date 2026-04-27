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

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";
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
  const { summary, loading, generating, error, selectedTemplate, setSelectedTemplate } = state;
  const { t } = useTranslation();
  const [customTemplates, setCustomTemplates] = useState<CustomTemplate[]>([]);
  const [showTemplateManager, setShowTemplateManager] = useState(false);

  const refreshCustomTemplates = () => {
    listCustomTemplates().then(setCustomTemplates).catch(() => {});
  };

  useEffect(() => {
    refreshCustomTemplates();
  }, []);

  return (
    <section
      aria-label="Summary"
      className="flex flex-col gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-900 dark:bg-zinc-900/40"
    >
      <header className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">Summary</h3>
        <div className="flex items-center gap-2">
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
          <SummaryActions
            summary={summary}
            generating={generating}
            loading={loading}
            onGenerate={() => {
              void state.generate();
            }}
          />
        </div>
      </header>

      {showTemplateManager && (
        <TemplateManager
          onClose={() => setShowTemplateManager(false)}
          onChanged={refreshCustomTemplates}
        />
      )}

      {error && (
        <p className="text-xs text-amber-700 dark:text-amber-400">{error}</p>
      )}

      {loading && (
        <div className="flex items-center gap-2">
          <LogoAnimated size={20} className="opacity-40" />
          <p className="text-xs text-zinc-500">Loading summary…</p>
        </div>
      )}
      {!loading && summary && <SummaryBody summary={summary} />}
      {!loading && !summary && (
        <p className="text-xs text-zinc-500">
          No summary yet. Pick a template and click <em>Generate</em>.
        </p>
      )}
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
        <optgroup label="Custom">
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
  const disabled = loading || generating;
  let label = "Generate";
  if (generating) label = "Generating…";
  else if (summary) label = "Regenerate";
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
  return (
    <div className="flex flex-col gap-3 text-sm">
      {renderTemplateBody(summary)}
      <Footer summary={summary} />
    </div>
  );
}

function renderTemplateBody(summary: Summary) {
  switch (summary.template) {
    case "general":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title="Key points" items={summary.keyPoints} />
          <SummarySection title="Decisions" items={summary.decisions} />
          <ActionItemsSection items={summary.actionItems} />
          <SummarySection title="Open questions" items={summary.openQuestions} />
        </>
      );

    case "oneOnOne":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title="Wins" items={summary.wins} />
          <SummarySection title="Blockers" items={summary.blockers} />
          <SummarySection title="Growth feedback" items={summary.growthFeedback} />
          <ActionItemsSection items={summary.nextSteps} title="Next steps" />
          <SummarySection title="Follow-up topics" items={summary.followUpTopics} />
        </>
      );

    case "sprintReview":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title="Completed" items={summary.completedItems} />
          <SummarySection title="Carry-over" items={summary.carryOver} />
          <SummarySection title="Risks" items={summary.risks} />
          <SummarySection title="Next sprint priorities" items={summary.nextSprintPriorities} />
        </>
      );

    case "interview":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <QuotesSection quotes={summary.quotes} />
          <SummarySection title="Themes" items={summary.themes} />
          <SummarySection title="Pain points" items={summary.painPoints} />
          <SummarySection title="Opportunities" items={summary.opportunities} />
        </>
      );

    case "salesCall":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          {summary.customerContext && (
            <div className="flex flex-col gap-1">
              <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
                Customer context
              </h4>
              <p className="text-zinc-800 dark:text-zinc-200">{summary.customerContext}</p>
            </div>
          )}
          <SummarySection title="Pain points" items={summary.painPoints} />
          <SummarySection title="Interest signals" items={summary.interestSignals} />
          <SummarySection title="Objections" items={summary.objections} />
          <ActionItemsSection items={summary.nextSteps} title="Next steps" />
          {summary.dealStageIndicator && (
            <p className="text-xs text-zinc-500">
              Deal stage: <span className="font-medium">{summary.dealStageIndicator}</span>
            </p>
          )}
        </>
      );

    case "lecture":
      return (
        <>
          <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
          <SummarySection title="Concepts covered" items={summary.conceptsCovered} />
          <DefinitionsSection definitions={summary.definitions} />
          <SummarySection title="Examples" items={summary.examples} />
          <SummarySection title="Homework / next" items={summary.homeworkOrNext} />
        </>
      );

    case "freeText":
      return (
        <div className="whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
          {summary.text || "[empty summary]"}
        </div>
      );

    case "custom":
      return (
        <div className="flex flex-col gap-2">
          <p className="text-xs font-medium text-zinc-500 dark:text-zinc-400">
            {summary.templateName}
          </p>
          <div className="whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
            {summary.text || "[empty summary]"}
          </div>
        </div>
      );

    default:
      return (
        <p className="text-zinc-500 italic">
          Unknown template. Try regenerating with a known template.
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
  title = "Action items",
}: Readonly<{
  items: { task: string; owner?: string | null; due?: string | null }[];
  title?: string;
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
  if (quotes.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Notable quotes
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
  if (definitions.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Definitions
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
