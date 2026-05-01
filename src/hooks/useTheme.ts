import { useCallback, useEffect, useSyncExternalStore } from "react";

export type Theme = "light" | "dark" | "system";

const STORAGE_KEY = "echonote-theme";

/* ── Tiny external store so every consumer stays in sync ── */

const listeners = new Set<() => void>();
function emit() { listeners.forEach((l) => l()); }

function getTheme(): Theme {
  const stored = localStorage.getItem(STORAGE_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return "system";
}

function setTheme(t: Theme) {
  if (t === "system") localStorage.removeItem(STORAGE_KEY);
  else localStorage.setItem(STORAGE_KEY, t);
  applyClass(t);
  emit();
}

/** Resolve "system" to the actual OS preference. */
function resolvedDark(theme: Theme): boolean {
  if (theme === "dark") return true;
  if (theme === "light") return false;
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function applyClass(theme: Theme) {
  const root = document.documentElement;
  if (resolvedDark(theme)) root.classList.add("dark");
  else root.classList.remove("dark");
}

/* Apply on load so the first paint is correct. */
applyClass(getTheme());

/* Listen for OS preference changes when in "system" mode. */
const mq = window.matchMedia("(prefers-color-scheme: dark)");
mq.addEventListener("change", () => {
  if (getTheme() === "system") {
    applyClass("system");
    emit();
  }
});

/* ── Hook ── */

export function useTheme() {
  const theme = useSyncExternalStore(
    useCallback((cb: () => void) => {
      listeners.add(cb);
      return () => listeners.delete(cb);
    }, []),
    getTheme,
    () => "system" as Theme,
  );

  // Keep class in sync when theme changes (covers SSR hydration edge).
  useEffect(() => applyClass(theme), [theme]);

  return { theme, setTheme } as const;
}
