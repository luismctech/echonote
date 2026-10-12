import { memo } from "react";
import { useTranslation } from "react-i18next";

import { formatDate } from "../../lib/format";
import type {
  MeetingId,
  MeetingSearchHit,
  MeetingSummary,
} from "../../types/meeting";

/** FTS5 hits rendered as a sidebar list with `<mark>`-highlighted snippets. */
export const SearchResults = memo(function SearchResults({
  query,
  hits,
  loading,
  error,
  activeId,
  onSelect,
}: {
  query: string;
  hits: MeetingSearchHit[];
  loading: boolean;
  error: string | null;
  activeId: MeetingId | null;
  onSelect: (m: MeetingSummary) => void;
}) {
  const { t } = useTranslation();
  if (error) {
    return (
        <p className="rounded-sm bg-rose-50 px-2 py-1 text-ui-sm text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
        {t("search.failed")} {error}
      </p>
    );
  }
  if (loading && hits.length === 0) {
    return <p className="text-ui-sm text-content-placeholder">{t("search.searching")}</p>;
  }
  if (hits.length === 0) {
    return (
      <p className="text-ui-sm text-content-placeholder">
        {t("search.noMatches", { query })}
      </p>
    );
  }
  return (
    <ul
      className="flex flex-col gap-1 overflow-y-auto"
      style={{ maxHeight: "60vh" }}
    >
      {hits.map((hit) => {
        const m = hit.meeting;
        const active = m.id === activeId;
        return (
          <li key={m.id}>
            <button
              type="button"
              onClick={() => onSelect(m)}
              className={`flex w-full flex-col items-start gap-1 rounded-md border px-2.5 py-2 text-left text-ui-sm ${
                active
                  ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"
                  : "border-transparent hover:bg-surface-sunken"
              }`}
            >
              <span className="line-clamp-1 font-medium text-content-primary">
                {m.title}
              </span>
              {/*
                The snippet is HTML-escaped server-side by `sanitize_fts_snippet`
                (echo-storage, sqlite.rs) before only the FTS5 `<mark>` tags are
                restored, so transcript text cannot inject markup here.
              */}
              <span
                className="line-clamp-2 text-ui-xs leading-snug text-content-secondary [&_mark]:rounded-sm [&_mark]:bg-amber-200/60 [&_mark]:px-0.5 [&_mark]:text-content-primary dark:[&_mark]:bg-amber-500/30"
                // biome-ignore lint/security/noDangerouslySetInnerHtml: escaped server-side by sanitize_fts_snippet; only FTS5 <mark> tags are restored
                dangerouslySetInnerHTML={{ __html: hit.snippet }}
              />
              <span className="text-micro tabular-nums text-content-placeholder">
                {formatDate(m.startedAt)} · {t("search.rank")} {hit.rank.toFixed(2)}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
});
