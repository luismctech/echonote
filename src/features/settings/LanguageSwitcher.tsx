/**
 * `<LanguageSwitcher />` — compact locale toggle for the app header.
 *
 * Renders the current language code as a small pill button. Clicking
 * cycles between `en` and `es`. Extending to more locales later just
 * means adding entries to the `LOCALES` array.
 */

import { useTranslation } from "react-i18next";

const LOCALES = ["en", "es"] as const;

export function LanguageSwitcher() {
  const { i18n, t } = useTranslation();

  const cycle = () => {
    const idx = LOCALES.indexOf(i18n.language as (typeof LOCALES)[number]);
    const next = LOCALES[(idx + 1) % LOCALES.length];
    void i18n.changeLanguage(next);
  };

  return (
    <button
      type="button"
      onClick={cycle}
      className="rounded-md border border-zinc-200 px-2 py-1 font-mono text-[11px] leading-none text-zinc-600 hover:bg-zinc-50 dark:border-zinc-700 dark:text-zinc-300 dark:hover:bg-zinc-800"
      title={t("settings.switchLanguage")}
    >
      {i18n.language.toUpperCase()}
    </button>
  );
}
