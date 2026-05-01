/**
 * `<ThemeSwitcher />` — compact theme toggle for the app header.
 *
 * Cycles through light → dark → system on each click.
 * Shows a sun, moon, or monitor icon respectively.
 */

import { useTranslation } from "react-i18next";
import { useTheme, type Theme } from "../../hooks/useTheme";

const CYCLE: Theme[] = ["light", "dark", "system"];

const ICONS: Record<Theme, JSX.Element> = {
  light: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
      <path d="M8 1a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 1ZM10.5 8a2.5 2.5 0 1 1-5 0 2.5 2.5 0 0 1 5 0ZM12.95 4.11a.75.75 0 1 0-1.06-1.06l-1.062 1.06a.75.75 0 0 0 1.061 1.06l1.06-1.06ZM15 8a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1 0-1.5h1.5A.75.75 0 0 1 15 8ZM11.888 12.95a.75.75 0 0 0 1.06-1.06l-1.06-1.062a.75.75 0 0 0-1.06 1.061l1.06 1.06ZM8 12a.75.75 0 0 1 .75.75v1.5a.75.75 0 0 1-1.5 0v-1.5A.75.75 0 0 1 8 12ZM5.172 11.888a.75.75 0 0 0-1.061-1.06l-1.06 1.06a.75.75 0 1 0 1.06 1.06l1.06-1.06ZM4 8a.75.75 0 0 1-.75.75h-1.5a.75.75 0 0 1 0-1.5h1.5A.75.75 0 0 1 4 8ZM4.11 3.05a.75.75 0 1 0-1.06 1.06l1.06 1.062a.75.75 0 0 0 1.06-1.061L4.11 3.05Z" />
    </svg>
  ),
  dark: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
      <path d="M14.438 10.148c.19-.425-.321-.787-.748-.601A5.5 5.5 0 0 1 6.453 2.31c.186-.427-.176-.938-.6-.748a6.501 6.501 0 1 0 8.585 8.586Z" />
    </svg>
  ),
  system: (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16" fill="currentColor" className="h-3.5 w-3.5">
      <path fillRule="evenodd" d="M2 4.25A2.25 2.25 0 0 1 4.25 2h7.5A2.25 2.25 0 0 1 14 4.25v5.5A2.25 2.25 0 0 1 11.75 12h-2.379l.435 1.088a.75.75 0 0 1-.696 1.037H6.89a.75.75 0 0 1-.696-1.037L6.629 12H4.25A2.25 2.25 0 0 1 2 9.75v-5.5Zm2.25-.75a.75.75 0 0 0-.75.75v5.5c0 .414.336.75.75.75h7.5a.75.75 0 0 0 .75-.75v-5.5a.75.75 0 0 0-.75-.75h-7.5Z" clipRule="evenodd" />
    </svg>
  ),
};

export function ThemeSwitcher() {
  const { t } = useTranslation();
  const { theme, setTheme } = useTheme();

  const cycle = () => {
    const idx = CYCLE.indexOf(theme);
    const next = CYCLE[(idx + 1) % CYCLE.length]!;
    setTheme(next);
  };

  return (
    <button
      type="button"
      onClick={cycle}
      className="rounded-md border px-2 py-1 text-content-secondary hover:bg-surface-sunken"
      title={t(`settings.theme.${theme}`)}
    >
      {ICONS[theme]}
    </button>
  );
}
