import { useTranslation } from "react-i18next";

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
  const { t } = useTranslation();
  return (
    <div className="relative">
      <input
        type="search"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={t("sidebar.searchPlaceholder")}
        aria-label={t("sidebar.searchLabel")}
        className="w-full rounded-md border bg-surface-sunken px-2.5 py-1.5 pr-7 text-ui-sm text-content-primary placeholder:text-content-placeholder focus:border-accent-400 focus:outline-none focus:ring-1 focus:ring-accent-400"
      />
      {value && (
        <button
          type="button"
          onClick={() => onChange("")}
          aria-label={t("sidebar.clearSearch")}
          className="absolute right-1.5 top-1/2 -translate-y-1/2 rounded text-ui-sm text-content-placeholder hover:text-content-primary"
        >
          ×
        </button>
      )}
      {loading && (
        <span
          aria-live="polite"
          className="pointer-events-none absolute right-6 top-1/2 -translate-y-1/2 text-micro text-content-placeholder"
        >
          …
        </span>
      )}
    </div>
  );
}
