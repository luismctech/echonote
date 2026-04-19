/**
 * `SummaryPanel` — render the LLM summary for a single meeting.
 *
 * Three view states, in this order:
 *
 *   1. **Loading**   the initial `getSummary` is in flight.
 *   2. **Empty**     the meeting has no summary yet → render a CTA
 *                    button that calls `summarize_meeting`.
 *   3. **Loaded**    a `Summary` exists; render the structured
 *                    sections (general template) or the fallback
 *                    free-text block, plus a "Regenerate" affordance.
 *
 * The `generating` spinner is shown additively in states 2 + 3 so the
 * user always sees that work is happening, regardless of whether they
 * had an old summary on screen.
 */

import { formatDate } from "../../lib/format";
import type { Summary } from "../../types/summary";

export function SummaryPanel({
  summary,
  loading,
  generating,
  error,
  onGenerate,
}: {
  summary: Summary | null;
  loading: boolean;
  generating: boolean;
  error: string | null;
  onGenerate: () => void;
}) {
  return (
    <section
      aria-label="Summary"
      className="flex flex-col gap-2 rounded-md border border-zinc-100 bg-zinc-50 p-3 dark:border-zinc-900 dark:bg-zinc-900/40"
    >
      <header className="flex items-center justify-between gap-2">
        <h3 className="text-sm font-medium">Summary</h3>
        <SummaryActions
          summary={summary}
          generating={generating}
          loading={loading}
          onGenerate={onGenerate}
        />
      </header>

      {/* Inline error from the initial load. Generation errors are
          handled by the toast layer (see `useMeetingSummary`). */}
      {error && (
        <p className="text-xs text-amber-700 dark:text-amber-400">{error}</p>
      )}

      {loading ? (
        <p className="text-xs text-zinc-500">Loading summary…</p>
      ) : summary ? (
        <SummaryBody summary={summary} />
      ) : (
        <p className="text-xs text-zinc-500">
          No summary yet. Click <em>Generate</em> to create one with the
          local LLM.
        </p>
      )}
    </section>
  );
}

/**
 * Right-side controls for the panel header. Disabled while a load /
 * generate is in flight so the user can't fire two requests in a row.
 * The label switches between "Generate" (no summary) and "Regenerate"
 * (existing summary) so the action is unambiguous.
 */
function SummaryActions({
  summary,
  generating,
  loading,
  onGenerate,
}: {
  summary: Summary | null;
  generating: boolean;
  loading: boolean;
  onGenerate: () => void;
}) {
  const disabled = loading || generating;
  const label = generating
    ? "Generating…"
    : summary
      ? "Regenerate"
      : "Generate";
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

function SummaryBody({ summary }: { summary: Summary }) {
  return (
    <div className="flex flex-col gap-3 text-sm">
      {summary.template === "general" ? (
        <GeneralBody summary={summary} />
      ) : (
        // Fallback the use case writes when the LLM keeps emitting
        // unparseable JSON. Show the raw text and trust the user to
        // either regenerate or copy what they need.
        <div className="whitespace-pre-wrap text-zinc-800 dark:text-zinc-200">
          {summary.text || "[empty summary]"}
        </div>
      )}
      <Footer summary={summary} />
    </div>
  );
}

function GeneralBody({
  summary,
}: {
  // Type-narrowed: the parent already guarded on `template === "general"`.
  summary: Extract<Summary, { template: "general" }>;
}) {
  return (
    <>
      <p className="text-zinc-800 dark:text-zinc-200">{summary.summary}</p>
      <SummarySection title="Key points" items={summary.keyPoints} />
      <SummarySection title="Decisions" items={summary.decisions} />
      <ActionItemsSection items={summary.actionItems} />
      <SummarySection title="Open questions" items={summary.openQuestions} />
    </>
  );
}

/**
 * Render a list-typed section (`keyPoints`, `decisions`,
 * `openQuestions`). Hidden entirely when empty so a sparse summary
 * doesn't waste vertical space with three "(none)" rows.
 */
function SummarySection({
  title,
  items,
}: {
  title: string;
  items: string[];
}) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        {title}
      </h4>
      <ul className="ml-4 list-disc space-y-1 text-zinc-800 dark:text-zinc-200">
        {items.map((it, i) => (
          // Same caveat as the segments list: stable order from the
          // backend means index keys are safe enough here.
          <li key={`${title}-${i}`}>{it}</li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Action items get their own section because each row carries
 * optional `owner` + `due` metadata that wouldn't fit a flat list.
 */
function ActionItemsSection({
  items,
}: {
  items: Extract<Summary, { template: "general" }>["actionItems"];
}) {
  if (items.length === 0) return null;
  return (
    <div className="flex flex-col gap-1">
      <h4 className="text-xs font-semibold uppercase tracking-wide text-zinc-500 dark:text-zinc-400">
        Action items
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

/**
 * Provenance footer. Shows the model + generation date so the user
 * can spot a stale summary at a glance ("oh, this is from yesterday's
 * recording, before I added the Q&A segments").
 */
function Footer({ summary }: { summary: Summary }) {
  return (
    <p className="text-[10px] text-zinc-400">
      {summary.model} · {formatDate(summary.createdAt)}
      {summary.language ? ` · ${summary.language}` : ""}
    </p>
  );
}
