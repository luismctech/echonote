/**
 * Environment guard for Tauri vs plain browser.
 *
 * Lives in its own module so it can be imported by everything (the
 * client, hooks, stores) without dragging in the rest of `client.ts`.
 * The check matches Tauri 2.x: the runtime injects `__TAURI_INTERNALS__`
 * onto `window` at preload time. In `pnpm dev` (vanilla Vite) the
 * property is absent, which lets the app fall back to a "running
 * outside Tauri" probe instead of throwing on every `invoke()`.
 */
export const isTauri = (): boolean =>
  typeof window !== "undefined" && "__TAURI_INTERNALS__" in window;
