/** Search input + clear button + loading indicator for the sidebar. */
export function MeetingsSearchBox({
  value,
  onChange,
  loading,
}: {
  value: string;
  onChange: (next: string) => void;
  loading: boolean;
}) {
  return (
    <div className="relative">
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder="Search transcripts…"
        aria-label="Search meeting transcripts"
        className="w-full rounded-md border border-zinc-200 bg-zinc-50 px-2.5 py-1.5 pr-7 text-xs text-zinc-800 placeholder:text-zinc-400 focus:border-emerald-400 focus:outline-none focus:ring-1 focus:ring-emerald-300 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-100"
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label="Clear search"
          className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded text-xs text-zinc-400 hover:text-zinc-700 dark:hover:text-zinc-200"
        >
          ×
        </button>
      )}
      {loading && (
        <span
          aria-live="polite"
          className="pointer-events-none absolute right-6 top-1/2 -translate-y-1/2 text-[10px] text-zinc-400"
        >
          …
        </span>
      )}
    </div>
  );
}
