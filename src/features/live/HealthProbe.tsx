import { useTranslation } from "react-i18next";

import type { Probe } from "../../types/view";

/**
 * Compact backend-health pill rendered in the app header.
 *
 * Single-line, font-mono pill that fits in the top-right of the
 * header. The full health payload (target, commit, …) is exposed via
 * a tooltip so it stays inspectable without consuming chrome space.
 */
export function HealthProbe({ probe }: Readonly<{ probe: Probe }>) {
  const { t } = useTranslation();
  const base =
    "flex items-center gap-1.5 rounded-md border px-2 py-1 font-mono text-[11px] leading-none whitespace-nowrap";
  switch (probe.kind) {
    case "idle":
      return (
        <span
          className={`${base} border-zinc-200 bg-zinc-50 text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-zinc-400" />
          {t("health.warmingUp")}
        </span>
      );
    case "loading":
      return (
        <span
          className={`${base} border-zinc-200 bg-zinc-50 text-zinc-500 dark:border-zinc-800 dark:bg-zinc-900`}
        >
          <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-amber-400" />
          {t("health.probing")}
        </span>
      );
    case "error":
      return (
        <span
          className={`${base} border-amber-300 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-300`}
          title={probe.message}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-amber-500" />
          {t("health.offline")}
        </span>
      );
    case "ok":
      return (
        <span
          className={`${base} border-emerald-300 bg-emerald-50 text-emerald-800 dark:border-emerald-900 dark:bg-emerald-950/40 dark:text-emerald-300`}
          title={`v${probe.status.version} · ${probe.status.target} · ${probe.status.commit}`}
        >
          <span className="h-1.5 w-1.5 rounded-full bg-emerald-500" />
          {t("health.ok", { version: probe.status.version })}
        </span>
      );
  }
}
