/**
 * `useAutoUpdate` — checks for app updates on mount and at a regular interval.
 *
 * Uses Tauri's updater plugin to check for updates via the configured
 * endpoint (GitHub Releases by default). When an update is found, it
 * surfaces a toast. The user can trigger install from the settings or
 * the toast action.
 *
 * Outside Tauri (plain `pnpm dev`) this hook is a no-op.
 */

import { useCallback, useEffect, useRef } from "react";

import { isTauri } from "../ipc/isTauri";

/** How often to check for updates (ms). Default: every 4 hours. */
const CHECK_INTERVAL_MS = 4 * 60 * 60 * 1000;

export type UpdateStatus =
  | { state: "idle" }
  | { state: "checking" }
  | { state: "available"; version: string; body?: string }
  | { state: "downloading"; progress: number }
  | { state: "ready"; version: string }
  | { state: "error"; message: string }
  | { state: "up-to-date" };

type OnUpdateFound = (version: string, body?: string) => void;

/**
 * Check for updates silently. Call `downloadAndInstall` separately.
 *
 * @param onUpdateFound — called when a newer version exists.
 */
export function useAutoUpdate(onUpdateFound?: OnUpdateFound) {
  const onUpdateFoundRef = useRef(onUpdateFound);
  onUpdateFoundRef.current = onUpdateFound;

  /** Tracks the last version we surfaced so we don't fire the callback twice. */
  const lastNotifiedRef = useRef<string | null>(null);

  const checkForUpdate = useCallback(async (): Promise<boolean> => {
    if (!isTauri()) return false;

    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (update) {
        if (lastNotifiedRef.current === update.version) return true;
        lastNotifiedRef.current = update.version;
        onUpdateFoundRef.current?.(update.version, update.body ?? undefined);
        return true;
      }
      return false;
    } catch {
      // Silently ignore — updater may not be configured yet (no pubkey).
      return false;
    }
  }, []);

  useEffect(() => {
    // Initial check after a short delay so the app can finish booting.
    const initial = setTimeout(checkForUpdate, 5_000);

    const interval = setInterval(checkForUpdate, CHECK_INTERVAL_MS);

    return () => {
      clearTimeout(initial);
      clearInterval(interval);
    };
  }, [checkForUpdate]);

  return { checkForUpdate };
}

/**
 * Download and install a pending update, then relaunch the app.
 * Call this from a UI button (e.g. toast action or settings panel).
 *
 * @param onProgress — called with a fraction 0→1 during download.
 * @param onPhase — called when the phase changes so the UI can
 *   surface "downloading…", "installing…", "restarting…" etc.
 */
export async function installUpdate(
  onProgress?: (fraction: number) => void,
  onPhase?: (phase: "downloading" | "installed" | "no-update") => void,
): Promise<void> {
  const { check } = await import("@tauri-apps/plugin-updater");
  const { relaunch } = await import("@tauri-apps/plugin-process");

  const update = await check();
  if (!update) {
    onPhase?.("no-update");
    return;
  }

  onPhase?.("downloading");

  let downloaded = 0;
  let contentLength = 1;

  await update.downloadAndInstall((event) => {
    switch (event.event) {
      case "Started":
        contentLength = event.data.contentLength ?? 1;
        break;
      case "Progress":
        downloaded += event.data.chunkLength;
        onProgress?.(downloaded / contentLength);
        break;
      case "Finished":
        onProgress?.(1);
        break;
    }
  });

  // At this point the new version is already on disk. Show the user
  // a clear "close & reopen" message, then *try* to relaunch as a
  // convenience — if it works the app closes instantly; if not, the
  // toast is already visible with instructions.
  onPhase?.("installed");

  // Give the user ~5 s to read the "installed" toast before the app
  // potentially closes via relaunch.
  await new Promise((r) => setTimeout(r, 5_000));

  // Best-effort restart. If it fails, the update is still installed
  // and will be active on the next manual launch.
  try {
    await relaunch();
  } catch {
    // Swallow — the "installed" toast already told the user what to do.
  }
}
