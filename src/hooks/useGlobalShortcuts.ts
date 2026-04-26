/**
 * Register global keyboard shortcuts that work even when the app
 * window is not focused (e.g. while in a Zoom/Meet call).
 *
 * Currently registers a single toggle shortcut:
 *   macOS:  Cmd+Shift+R
 *   other:  Ctrl+Shift+R
 *
 * The shortcut starts recording when idle and stops when recording.
 */

import { useEffect, useRef } from "react";
import {
  register,
  unregister,
} from "@tauri-apps/plugin-global-shortcut";

const SHORTCUT = "CommandOrControl+Shift+KeyR";

export function useGlobalShortcuts({
  canStart,
  canStop,
  onStart,
  onStop,
}: Readonly<{
  canStart: boolean;
  canStop: boolean;
  onStart: () => void;
  onStop: () => void;
}>) {
  const canStartRef = useRef(canStart);
  const canStopRef = useRef(canStop);
  const onStartRef = useRef(onStart);
  const onStopRef = useRef(onStop);

  canStartRef.current = canStart;
  canStopRef.current = canStop;
  onStartRef.current = onStart;
  onStopRef.current = onStop;

  useEffect(() => {
    let mounted = true;

    async function setup() {
      try {
        await register(SHORTCUT, (event) => {
          if (event.state !== "Pressed" || !mounted) return;
          if (canStopRef.current) {
            onStopRef.current();
          } else if (canStartRef.current) {
            onStartRef.current();
          }
        });
      } catch (e) {
        console.warn("Failed to register global shortcut:", e);
      }
    }

    setup();

    return () => {
      mounted = false;
      unregister(SHORTCUT).catch(() => {});
    };
  }, []);
}
