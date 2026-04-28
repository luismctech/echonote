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

  const checkForUpdate = useCallback(async (): Promise<boolean> => {
    if (!isTauri()) return false;

    try {
      const { check } = await import("@tauri-apps/plugin-updater");
      const update = await check();
      if (update) {
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
 */
export async function installUpdate(
  onProgress?: (fraction: number) => void,
): Promise<void> {
  const { check } = await import("@tauri-apps/plugin-updater");
  const { relaunch } = await import("@tauri-apps/plugin-process");

  const update = await check();
  if (!update) return;

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

  await relaunch();
}
