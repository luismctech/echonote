/**
 * `<ThemeSwitcher />` — compact theme toggle for the app header.
 *
 * Cycles through light → dark → system on each click.
 * Shows a sun, moon, or monitor icon respectively.
 */

import { useTranslation } from "react-i18next";
import { Sun, Moon, Monitor } from "lucide-react";
import { useTheme, type Theme } from "../../hooks/useTheme";

const CYCLE: Theme[] = ["light", "dark", "system"];

const ICONS: Record<Theme, JSX.Element> = {
  light: <Sun className="h-3.5 w-3.5" />,
  dark: <Moon className="h-3.5 w-3.5" />,
  system: <Monitor className="h-3.5 w-3.5" />,
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
