import { memo } from "react";

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
  if (error) {
    return (
      <p className="rounded bg-rose-50 px-2 py-1 text-xs text-rose-700 dark:bg-rose-950/40 dark:text-rose-300">
        Search failed: {error}
      </p>
    );
  }
  if (loading && hits.length === 0) {
    return <p className="text-xs text-zinc-400">Searching…</p>;
  }
  if (hits.length === 0) {
    return (
      <p className="text-xs text-zinc-400">
        No matches for <span className="font-medium">{query}</span>.
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
              className={`flex w-full flex-col items-start gap-1 rounded-md border px-2.5 py-2 text-left text-xs ${
                active
                  ? "border-emerald-300 bg-emerald-50 dark:border-emerald-800 dark:bg-emerald-950/40"
                  : "border-transparent hover:bg-zinc-50 dark:hover:bg-zinc-900"
              }`}
            >
              <span className="line-clamp-1 font-medium text-zinc-800 dark:text-zinc-100">
                {m.title}
              </span>
              {/*
                Snippet markers (`<mark>...</mark>`) are emitted by
                SQLite over text we ourselves indexed. The XSS surface
                is therefore identical to rendering the raw segment
                text, which the rest of the UI already does in plain
                strings — so `dangerouslySetInnerHTML` here is no more
                dangerous than e.g. a transcript line.
              */}
              <span
                className="line-clamp-2 text-[11px] leading-snug text-zinc-600 [&_mark]:rounded [&_mark]:bg-amber-200/60 [&_mark]:px-0.5 [&_mark]:text-zinc-900 dark:text-zinc-300 dark:[&_mark]:bg-amber-500/30 dark:[&_mark]:text-zinc-50"
                dangerouslySetInnerHTML={{ __html: hit.snippet }}
              />
              <span className="text-[10px] tabular-nums text-zinc-400">
                {formatDate(m.startedAt)} · rank {hit.rank.toFixed(2)}
              </span>
            </button>
          </li>
        );
      })}
    </ul>
  );
});
