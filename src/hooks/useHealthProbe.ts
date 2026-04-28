/**
 * `useHealthProbe` — owns the backend health-check state.
 *
 * Wires the on-mount `health_check` IPC into a single `Probe`
 * discriminated union that the header pill renders. When the app is
 * running outside Tauri (e.g. `pnpm dev`), `invoke` is unavailable;
 * we surface that as an `error` probe with a friendly message rather
 * than letting the call throw.
 *
 * Caller responsibilities:
 *   - Render the returned `probe` via `<HealthProbe />`.
 *   - Read `probe.kind === "ok"` before allowing user actions that
 *     require the backend (Start button gating).
 *
 * The hook fires its check exactly once on mount. The `useToast` API
 * is referentially stable (see Toaster.tsx) so the effect deps are
 * effectively empty.
 */

import { useEffect, useState } from "react";
import { useTranslation } from "react-i18next";

import { useToast } from "../components/Toaster";
import { healthCheck } from "../ipc/client";
import { isTauri } from "../ipc/isTauri";
import type { Probe } from "../types/view";

export function useHealthProbe(): Probe {
  const { t } = useTranslation();
  const toast = useToast();
  const [probe, setProbe] = useState<Probe>({ kind: "idle" });

  useEffect(() => {
    if (!isTauri()) {
      setProbe({
        kind: "error",
        message: t("errors.outsideTauri"),
      });
      return;
    }
    setProbe({ kind: "loading" });
    healthCheck()
      .then((status) => setProbe({ kind: "ok", status }))
      .catch((err: unknown) => {
        const message = err instanceof Error ? err.message : String(err);
        setProbe({ kind: "error", message });
        toast.push({
          kind: "error",
          message: t("errors.healthFailed"),
          detail: message,
        });
      });
  }, [toast, t]);

  return probe;
}
