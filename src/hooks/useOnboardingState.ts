import { useCallback, useState } from "react";

const KEY = "echonote:onboarding-complete";

/**
 * Persists a boolean flag indicating whether the user has completed
 * onboarding. Uses localStorage for synchronous reads (no flash of
 * onboarding on reload). The flag can be reset for development via
 * `localStorage.removeItem("echonote:onboarding-complete")`.
 */
export function useOnboardingState() {
  const [completed, setCompleted] = useState(
    () => globalThis.localStorage?.getItem(KEY) === "true",
  );

  const markComplete = useCallback(() => {
    globalThis.localStorage?.setItem(KEY, "true");
    setCompleted(true);
  }, []);

  const reset = useCallback(() => {
    globalThis.localStorage?.removeItem(KEY);
    setCompleted(false);
  }, []);

  return { completed, markComplete, reset } as const;
}
